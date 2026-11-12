"""Pre-tool and post-tool hooks (L14). Every MCP call, by an agent or by code, goes through here.

    result = await guarded_call(ctx, "quant", "edgar.get_metric", {...})

pre-tool  1. permission: the tool must be in cfg.agents.permissions[agent] (least privilege);
             write tools additionally need an approval token in the arguments
          2. budget: tool calls per agent and tokens per note
          3. EDGAR rate limit (token bucket, cfg.budgets.edgar_requests_per_second)
          4. data-loss checks (defences): private watchlist names and held-out eval paths may not
             appear in tool arguments
post-tool 5. untrusted text: every passage is scanned for injection; flagged ones are quarantined
             (text withheld, id kept) before any model reads them
          6. scrub secrets / out-of-scope watchlist names from what is logged
          7. trace event + Prometheus metrics (latency, status, EDGAR cache hit/miss)
A blocked call returns {"ok": false, "blocked": true, "error": reason} to the caller instead of
raising, so a ReAct agent sees why and can adapt.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from ..observability import metrics as M
from ..observability.tracing import emit
from ..tools import mcp_client
from .context import RunContext
from .guardrails import is_injection, scrub, watchlist_leaks


class _RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0

    def wait_time(self, per_second: float) -> float:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + 1.0 / per_second
            return slot - now


EDGAR_LIMITER = _RateLimiter()


def pre_tool(ctx: RunContext, agent: str, qualified: str, args: dict[str, Any]) -> str | None:
    """Return a refusal reason, or None if the call may proceed."""
    cfg = ctx.cfg
    perms = cfg["agents"]["permissions"].get(agent, [])
    if qualified not in perms:
        return f"permission denied: agent '{agent}' may not call {qualified}"
    if qualified in cfg["agents"]["write_tools"] and not args.get("approval"):
        return f"{qualified} is a write tool and needs an approval token from the approval gate"
    limit = cfg["budgets"]["tool_calls_per_agent"]
    if agent not in ("publisher", "critic", "system") and ctx.tool_calls_for(agent) >= limit:
        return f"tool-call budget exhausted for {agent} ({limit} calls)"
    if ctx.tokens_total >= cfg["budgets"]["tokens_per_note"]:
        return f"token budget for this note exhausted ({cfg['budgets']['tokens_per_note']})"
    blob = json.dumps(args, default=str)
    if "heldout" in blob.lower():
        return "held-out evaluation data is not readable by agents"
    if cfg["guardrails"]["enabled"]:
        leaks = watchlist_leaks(blob, ctx.watchlist, ctx.allowed_tickers)
        if leaks:
            return "private watchlist entries may not leave the system in tool arguments"
    return None


def _quarantine(ctx: RunContext, result: dict[str, Any]) -> dict[str, Any]:
    """Scan passages / section text for injected instructions; withhold flagged text."""
    cfg = ctx.cfg
    flagged = []
    containers = []
    if isinstance(result.get("passages"), list):
        containers = result["passages"]
    elif isinstance(result.get("passage"), dict):
        containers = [result["passage"]]
    for p in containers:
        bad, signals = is_injection(p.get("text", ""), cfg)
        p["untrusted"] = True
        if bad:
            p["injection_suspected"] = signals[:3]
            flagged.append(p.get("passage_id"))
            if cfg["guardrails"]["enabled"] and cfg["guardrails"].get("quarantine_untrusted", True):
                p["text"] = "[QUARANTINED by guardrail: this passage contained instructions aimed at AI systems; it was treated as data and withheld.]"
    for key in ("added", "removed", "changed"):
        for item in result.get(key, []) or []:
            bad, signals = is_injection(item.get("summary", ""), cfg)
            if bad:
                flagged.append(item.get("passage_id"))
                if cfg["guardrails"]["enabled"]:
                    item["summary"] = "[QUARANTINED by guardrail]"
    if isinstance(result.get("text"), str) and "fiscal_year" in result:
        bad, signals = is_injection(result["text"], cfg)
        if bad:
            flagged.append(f"section:{result.get('item')}")
    if flagged:
        M.GUARD_BLOCKS.labels("quarantined_passage").inc(len(flagged))
        emit(ctx.run_id, "guard", stage="post_tool", action="quarantine" if cfg["guardrails"]["enabled"] else "flag_only",
             passages=flagged)
        result["quarantined"] = flagged
    return result


async def guarded_call(ctx: RunContext, agent: str, qualified: str, args: dict[str, Any]) -> dict[str, Any]:
    server, tool = qualified.split(".", 1)
    reason = pre_tool(ctx, agent, qualified, args)
    if reason:
        M.TOOL_CALLS.labels(server, tool, "blocked").inc()
        M.GUARD_BLOCKS.labels("tool_call").inc()
        entry = {"agent": agent, "tool": qualified, "reason": reason}
        ctx.blocked.append(entry)
        emit(ctx.run_id, "tool_blocked", **entry, args=scrub(json.dumps(args, default=str), ctx.watchlist, ctx.allowed_tickers)[:400])
        return {"ok": False, "blocked": True, "error": reason}
    if server == "edgar":
        delay = EDGAR_LIMITER.wait_time(ctx.cfg["budgets"]["edgar_requests_per_second"])
        if delay > 0:
            await asyncio.sleep(delay)
    ctx.tool_calls[agent] = ctx.tool_calls.get(agent, 0) + 1
    t0 = time.perf_counter()
    result = await mcp_client.call(server, tool, args)
    latency = time.perf_counter() - t0
    result = _quarantine(ctx, result)
    for fact in result.get("facts", []) or []:
        ctx.facts_seen[fact["fact_id"]] = fact
    for p in result.get("passages", []) or ([result["passage"]] if isinstance(result.get("passage"), dict) else []):
        ctx.passages_seen[p["passage_id"]] = p
    for key in ("added", "removed", "changed"):
        for item in result.get(key, []) or []:
            ctx.passages_seen.setdefault(item["passage_id"], {"passage_id": item["passage_id"], "text": item.get("summary", "")})
    for err in result.get("errors", []) or ([result["error"]] if result.get("error") else []):
        ctx.tool_errors.append(f"{qualified}: {err}")
    status = "ok" if result.get("ok", True) else "error"
    M.TOOL_CALLS.labels(server, tool, status).inc()
    M.TOOL_LATENCY.labels(server).observe(latency)
    if "cache" in result:
        M.EDGAR_CACHE.labels(result["cache"]).inc()
    log = {"agent": agent, "tool": qualified, "status": status, "latency_s": round(latency, 3),
           "args": json.loads(scrub(json.dumps(args, default=str), ctx.watchlist, ctx.allowed_tickers)),
           "error": result.get("error", "") if status == "error" else ""}
    ctx.tool_log.append(log)
    emit(ctx.run_id, "tool", **log, result_preview=scrub(json.dumps(result, default=str), ctx.watchlist, ctx.allowed_tickers)[:600])
    return result

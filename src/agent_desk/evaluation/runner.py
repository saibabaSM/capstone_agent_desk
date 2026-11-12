"""JSONL runner: one experiment config x one split x N repetitions -> eval/results/*.jsonl.

Every run gets a FRESH checkpointer and memory store (data/eval_runs/<run>/), so cases cannot leak
into each other through memory, and memory cases seed their own earlier note (setup_question).
Notes are auto-approved by the simulated approver only when the output checks pass. Failed and
timed-out cases are recorded, never dropped.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from ..agents.runtime import desk_runtime
from ..config import PROJECT_ROOT, config_fingerprint, data_dir, load_yaml, with_overrides
from ..memory import store as mem
from .checkers import check_case

RESULTS = PROJECT_ROOT / "eval" / "results"
CASE_TIMEOUT_S = 600


def experiment_overrides(path: str | None) -> tuple[str, dict]:
    if not path:
        return "default", {}
    data = load_yaml(path)
    name = data.pop("name", Path(path).stem)
    data.pop("description", None)
    return name, data


async def _drain(gen) -> None:
    async for _ in gen:
        pass


async def run_case(rt, case: dict, overrides: dict, tag: str) -> dict[str, Any]:
    ov = json.loads(json.dumps(overrides))
    ov.setdefault("orchestration", {}).update(auto_approve_plan=True, auto_approve_note=True)
    analyst = f"eval-{case['id']}"
    t0 = time.time()
    error = ""
    thread = f"{tag}-{case['id']}"
    try:
        if case.get("setup_question"):
            await asyncio.wait_for(_drain(rt.start(case["setup_question"], analyst, ov, thread_id=thread + "-setup")), CASE_TIMEOUT_S)
        await asyncio.wait_for(_drain(rt.start(case["question"], analyst, ov, thread_id=thread)), CASE_TIMEOUT_S)
    except asyncio.TimeoutError:
        error = f"timeout after {CASE_TIMEOUT_S}s"
    except Exception as exc:  # recorded as an error case
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    latency = time.time() - t0
    state = (await rt.state(thread))["values"] if not error or True else {}
    usage = state.get("usage") or {}
    result = {
        "case_id": case["id"], "type": case["type"], "split": case["split"], "thread": thread,
        "status": state.get("status") if not error else "error", "message": state.get("message", "") or error,
        "error": error, "latency_s": round(latency, 2), "tokens": usage.get("tokens", 0),
        "cost_usd": usage.get("cost_usd", 0.0), "llm_calls": usage.get("llm_calls", 0),
        "tool_calls": sum((usage.get("tool_calls") or {}).values()), "blocked_tool_calls": len(usage.get("blocked", [])),
        "tool_log": usage.get("tool_log", []), "note": state.get("note") or state.get("draft", ""),
        "refs": state.get("refs") or {}, "plan": state.get("plan") or {}, "plan_history": state.get("plan_history") or [],
        "guard": state.get("guard") or {}, "approval": state.get("approval") or {},
        "critic_rounds": state.get("critic_round", 0),
        "critic_rejections": sum(1 for r in state.get("critic_reports", []) if not r.get("passed")),
        "verified_share": (state.get("output_check") or {}).get("verified_share"),
        "watchlist": state.get("watchlist") or [],
    }
    result["check"] = check_case(case, result)
    return result


async def run(cases: list[dict], experiment: str | None, rep: int = 1, concurrency: int = 3,
              label: str = "", playbook: list[dict] | None = None, extra_overrides: dict | None = None) -> Path:
    name, overrides = experiment_overrides(experiment)
    if extra_overrides:
        from ..config import deep_merge
        overrides = deep_merge(overrides, extra_overrides)
    cfg = with_overrides(overrides)
    tag = f"{name}{'-' + label if label else ''}-r{rep}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = data_dir() / "eval_runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{tag}.jsonl"
    sem = asyncio.Semaphore(concurrency)
    async with desk_runtime(str(run_dir / "checkpoints.db"), str(run_dir / "store.db")) as rt:
        for lesson in playbook or []:
            await mem.put_lesson(rt.store, "playbook", lesson["key"], lesson)

        async def one(case: dict) -> dict:
            async with sem:
                res = await run_case(rt, case, overrides, tag)
                res.update(experiment=name, rep=rep, config_version=cfg["version"], config_hash=config_fingerprint(cfg),
                           model=cfg["llm"]["models"]["specialist"], mode=cfg["orchestration"]["mode"])
                mark = "PASS" if res["check"]["success"] else "FAIL"
                print(f"  {mark} {case['id']:<24} {res['status']:<10} {res['latency_s']:>6.1f}s ${res['cost_usd']:.4f} "
                      f"{'; '.join(res['check']['notes'][:2])[:120]}", flush=True)
                return res

        results = await asyncio.gather(*(one(c) for c in cases))
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f"wrote {out_path}")
    return out_path


def summarize(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    lat = sorted(r["latency_s"] for r in results)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else 0
    return {
        "cases": n,
        "success": sum(r["check"]["success"] for r in results),
        "success_rate": round(sum(r["check"]["success"] for r in results) / n, 3) if n else 0,
        "policy_violations": sum(len(r["check"]["violations"]) for r in results),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "refused": sum(1 for r in results if r["status"] == "refused"),
        # legitimate work blocked by a defence: a non-refusal case refused, or with a tool call blocked
        "false_blocks": sum(1 for r in results if r["type"] != "refusal"
                            and (r["status"] == "refused" or r.get("blocked_tool_calls", 0) > 0)),
        "cost_per_case_usd": round(sum(r["cost_usd"] for r in results) / n, 5) if n else 0,
        "tokens_per_case": round(sum(r["tokens"] for r in results) / n) if n else 0,
        "latency_p50_s": round(statistics.median(lat), 2) if lat else 0,
        "latency_p95_s": round(p95, 2),
        "tool_calls_per_case": round(sum(r["tool_calls"] for r in results) / n, 1) if n else 0,
        "critic_rejections": sum(r.get("critic_rejections", 0) for r in results),
        "verified_share_min": min((r["verified_share"] for r in results if r.get("verified_share") is not None), default=None),
    }


def load_results(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

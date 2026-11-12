"""Chat models per role, from the one versioned config, with usage/cost accounting and budgets.

    llm = chat("specialist", ctx)            # ChatOpenAI on OpenRouter, model from cfg.llm.models
    llm = chat("judge", ctx)                 # a different model family

Every call is counted into the RunContext (tokens, cost, llm_calls) and traced. Before a call,
the per-note token budget is checked; when it is used up BudgetExceeded is raised and the graph
finishes with what it has. Cost: OpenRouter's reported cost when present, else the price table in
config (USD per 1M tokens).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from ..config import api_key, is_offline, needs_api_key
from ..control.context import BudgetExceeded, RunContext
from ..observability import metrics as M
from ..observability.tracing import emit


FATAL_ERRORS = ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "APIConnectionError")


def is_fatal(exc: BaseException) -> bool:
    """Configuration errors (bad key, unknown model) must stop the run, never trigger a silent
    fallback: otherwise a 'live team' experiment quietly becomes the scripted C0 baseline."""
    return any(name in type(exc).__name__ for name in FATAL_ERRORS)


class UsageCallback(BaseCallbackHandler):
    raise_error = True

    def __init__(self, ctx: RunContext | None, role: str, model: str) -> None:
        self.ctx, self.role, self.model = ctx, role, model

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        ctx = self.ctx
        if ctx and ctx.tokens_total >= ctx.cfg["budgets"]["tokens_per_note"]:
            emit(ctx.run_id, "error", kind="budget", role=self.role, tokens=ctx.tokens_total)
            raise BudgetExceeded(f"token budget of {ctx.cfg['budgets']['tokens_per_note']} reached")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        prompt = completion = 0
        cost = None
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) or {}
                prompt += usage.get("input_tokens", 0)
                completion += usage.get("output_tokens", 0)
                meta = getattr(msg, "response_metadata", {}) or {}
                tu = meta.get("token_usage") or {}
                if tu.get("cost") is not None:
                    cost = (cost or 0.0) + float(tu["cost"])
        if not prompt and response.llm_output:
            tu = response.llm_output.get("token_usage") or {}
            prompt, completion = tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)
            if tu.get("cost") is not None:
                cost = float(tu["cost"])
        if cost is None:
            prices = ((self.ctx.cfg if self.ctx else {}).get("llm", {}).get("prices_per_million", {}) or {}).get(self.model, {"in": 0, "out": 0})
            cost = (prompt * prices["in"] + completion * prices["out"]) / 1e6
        M.TOKENS.labels(self.role, "prompt").inc(prompt)
        M.TOKENS.labels(self.role, "completion").inc(completion)
        M.COST.labels(self.role).inc(cost)
        if self.ctx:
            self.ctx.prompt_tokens += prompt
            self.ctx.completion_tokens += completion
            self.ctx.tokens += prompt + completion
            self.ctx.cost_usd += cost
            self.ctx.llm_calls += 1
            emit(self.ctx.run_id, "llm", agent=self.ctx.agent, role=self.role, model=self.model,
                 prompt_tokens=prompt, completion_tokens=completion, cost_usd=round(cost, 6))


def chat(role: str, ctx: RunContext | None, cfg: dict | None = None, max_tokens: int | None = 2000) -> ChatOpenAI:
    cfg = cfg or (ctx.cfg if ctx else None)
    if cfg is None:
        from ..config import get_config
        cfg = get_config()
    if is_offline(cfg):
        raise RuntimeError("LLM requested in offline mode; callers must use their deterministic path")
    model = cfg["llm"]["models"][role]
    temp_key = role if role in cfg["llm"]["temperature"] else "specialist"
    return ChatOpenAI(
        model=model,
        temperature=cfg["llm"]["temperature"].get(temp_key, 0.0),
        base_url=cfg["llm"]["base_url"],
        api_key=api_key(cfg),
        max_tokens=max_tokens,
        timeout=cfg["llm"]["request_timeout_s"],
        max_retries=cfg["llm"]["max_retries"],
        callbacks=[UsageCallback(ctx, role, model)],
        extra_body={"usage": {"include": True}} if needs_api_key(cfg) else None,
        default_headers={"X-Title": f"AnalystDesk {cfg.get('version', '')}"},
    )

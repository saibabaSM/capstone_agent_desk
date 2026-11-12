"""Per-run context shared by the hooks, the LLM wrapper and the nodes of one note.

A RunContext is rebuilt at the start of every graph node from the graph state (so it survives a
crash + resume through the checkpointer) and the node returns `ctx.usage_delta()` into state,
where a reducer adds it up. Parallel branches therefore never share a mutable object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised by hooks/LLM wrapper when a note's token or tool-call budget is used up."""


@dataclass
class RunContext:
    run_id: str
    cfg: dict
    agent: str = "system"
    allowed_tickers: list[str] = field(default_factory=list)
    watchlist: list[str] = field(default_factory=list)  # private: never sent to a model or tool
    tokens_before: int = 0  # tokens used by the run before this node started
    tool_calls_before: dict[str, int] = field(default_factory=dict)
    # deltas produced inside this node
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    # evidence the tools really returned in this node (agents may only cite these)
    facts_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    passages_seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_errors: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    @classmethod
    def from_state(cls, state: dict, cfg: dict, agent: str) -> "RunContext":
        usage = state.get("usage") or {}
        plan = state.get("plan") or {}
        return cls(run_id=state.get("run_id", "run"), cfg=cfg, agent=agent,
                   allowed_tickers=list(plan.get("companies") or state.get("scope_tickers") or []),
                   watchlist=list(state.get("watchlist") or []),
                   tokens_before=int(usage.get("tokens", 0)),
                   tool_calls_before=dict(usage.get("tool_calls", {})))

    def for_agent(self, agent: str) -> "RunContext":
        """Same budgets and logs, different identity for permission checks."""
        self.agent = agent
        return self

    @property
    def tokens_total(self) -> int:
        return self.tokens_before + self.tokens

    def tool_calls_for(self, agent: str) -> int:
        return self.tool_calls_before.get(agent, 0) + self.tool_calls.get(agent, 0)

    def usage_delta(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cost_usd": round(self.cost_usd, 6),
                "llm_calls": self.llm_calls, "tool_calls": dict(self.tool_calls),
                "tool_log": list(self.tool_log), "blocked": list(self.blocked)}


def merge_usage(current: dict | None, update: dict | None) -> dict:
    """LangGraph reducer for state['usage']: sums counters, concatenates logs."""
    current = dict(current or {})
    update = update or {}
    if update.get("__reset__"):
        return {}
    out = {
        "tokens": current.get("tokens", 0) + update.get("tokens", 0),
        "prompt_tokens": current.get("prompt_tokens", 0) + update.get("prompt_tokens", 0),
        "completion_tokens": current.get("completion_tokens", 0) + update.get("completion_tokens", 0),
        "cost_usd": round(current.get("cost_usd", 0.0) + update.get("cost_usd", 0.0), 6),
        "llm_calls": current.get("llm_calls", 0) + update.get("llm_calls", 0),
        "tool_calls": dict(current.get("tool_calls", {})),
        "tool_log": list(current.get("tool_log", [])) + list(update.get("tool_log", [])),
        "blocked": list(current.get("blocked", [])) + list(update.get("blocked", [])),
    }
    for agent, n in (update.get("tool_calls") or {}).items():
        out["tool_calls"][agent] = out["tool_calls"].get(agent, 0) + n
    return out

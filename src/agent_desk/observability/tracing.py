"""Local traces: one JSONL file per run with every node, LLM call, tool call, guard and decision.

    data/traces/<run_id>.jsonl   {"ts":..., "run_id":..., "type":"tool", "agent":"quant", ...}

Types: node, llm, tool, tool_blocked, guard, plan, replan, critic, approval, publish, memory, error.
LangSmith tracing is added on top when LANGSMITH_API_KEY is set (LangChain picks it up from the
environment); the local file is always written so evaluation never depends on a SaaS.
Traces are scrubbed by the post-tool hook before they are written (no watchlist, no secrets).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ..config import data_dir

_lock = threading.Lock()


def trace_path(run_id: str) -> Path:
    path = data_dir() / "traces"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{run_id}.jsonl"


def emit(run_id: str, type_: str, **fields: Any) -> dict[str, Any]:
    event = {"ts": round(time.time(), 3), "run_id": run_id, "type": type_, **fields}
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _lock, open(trace_path(run_id), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return event


def read_trace(run_id: str) -> list[dict[str, Any]]:
    path = trace_path(run_id)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

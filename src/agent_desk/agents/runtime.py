"""Open the persistent checkpointer + store, start / resume / inspect runs.

    async with desk_runtime() as rt:
        async for event in rt.start("Compare MSFT and GOOGL operating margins...", analyst="alice"):
            print(event)                         # {"node": ..., "update": ...} / {"interrupt": {...}}
        async for event in rt.resume(thread_id, {"action": "approve", "note_hash": h}):
            ...

The checkpointer (data/checkpoints.db) saves state after every node, so a run killed mid-way
(Ctrl-C, crash, container restart) continues from the last completed node with
`python -m agent_desk resume <thread_id>` - demo of L8's "stop the process mid-run and resume".
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command

from ..config import data_dir
from ..observability.tracing import emit
from .team_graph import build_graph


@dataclass
class Runtime:
    graph: Any
    store: Any

    def config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}

    async def _stream(self, inp: Any, thread_id: str) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self.graph.astream(inp, self.config(thread_id), stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield {"interrupt": update[0].value, "thread_id": thread_id}
                else:
                    yield {"node": node, "update": update or {}, "thread_id": thread_id}

    async def start(self, question: str, analyst: str = "analyst", overrides: dict | None = None,
                    thread_id: str | None = None) -> AsyncIterator[dict[str, Any]]:
        thread_id = thread_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        emit(thread_id, "node", node="start", question=question, analyst=analyst, overrides=overrides or {})
        state = {"run_id": thread_id, "question": question, "analyst": analyst, "cfg_overrides": overrides or {},
                 "started_at": time.time(), "replan_count": 0, "critic_round": 0}
        async for event in self._stream(state, thread_id):
            yield event

    async def resume(self, thread_id: str, value: Any = None) -> AsyncIterator[dict[str, Any]]:
        """Resume an interrupted run (value = the human's answer) or a crashed one (value None)."""
        snapshot = await self.graph.aget_state(self.config(thread_id))
        inp = Command(resume=value) if snapshot.interrupts else None
        emit(thread_id, "node", node="resume", pending=list(snapshot.next), with_answer=value is not None)
        async for event in self._stream(inp, thread_id):
            yield event

    async def state(self, thread_id: str) -> dict[str, Any]:
        snap = await self.graph.aget_state(self.config(thread_id))
        return {"values": snap.values, "next": list(snap.next),
                "interrupts": [i.value for i in snap.interrupts]}


@asynccontextmanager
async def desk_runtime(checkpoint_path: str | None = None, store_path: str | None = None):
    cp_path = checkpoint_path or str(data_dir() / "checkpoints.db")
    st_path = store_path or str(data_dir() / "memory" / "store.db")
    (data_dir() / "memory").mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(cp_path) as saver, AsyncSqliteStore.from_conn_string(st_path) as store:
        await store.setup()
        graph = build_graph().compile(checkpointer=saver, store=store)
        yield Runtime(graph=graph, store=store)

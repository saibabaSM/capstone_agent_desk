"""FastAPI service: streaming runs, plan edit, note approval, memory, metrics (L16).

    uvicorn agent_desk.api.server:app            (or: python -m agent_desk serve)

Auth: every /api call needs `Authorization: Bearer <token>`; ANALYST_TOKENS="alice:tok1,bob:tok2"
maps tokens to analysts (default dev token "dev-token" -> analyst "analyst"). A run belongs to the
analyst who started it; nobody else can see, resume or approve it.

  GET  /                      the research UI (static/index.html)
  POST /api/runs              {"question"} -> text/event-stream of node updates, ends at an interrupt or the end
  POST /api/runs/{id}/resume  {"action": "approve"|"edit"|"reject", ...} -> event stream
  GET  /api/runs/{id}         current state (plan, draft, note, approval, usage)
  GET  /api/runs/{id}/trace   trace events
  GET/PUT /api/memory/prefs   analyst preferences;  PUT /api/memory/watchlist (private);  DELETE /api/memory
  GET  /api/notes             published notes (research-portal)
  GET  /metrics               Prometheus;   GET /health
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from ..agents.runtime import desk_runtime
from ..config import get_config
from ..memory import store as mem
from ..observability.tracing import read_trace
from ..tools import mcp_client

STATIC = Path(__file__).parent / "static"
OWNERS: dict[str, str] = {}  # thread_id -> analyst (also persisted in the run state)


def _tokens() -> dict[str, str]:
    raw = os.getenv("ANALYST_TOKENS", "analyst:dev-token")
    return {tok: name for name, tok in (pair.split(":", 1) for pair in raw.split(",") if ":" in pair)}


def analyst(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    name = _tokens().get(token)
    if not name:
        raise HTTPException(401, "missing or invalid bearer token")
    return name


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with desk_runtime() as rt:
        app.state.rt = rt
        yield


app = FastAPI(title="Analyst Desk", version=get_config()["version"], lifespan=lifespan)


class StartRun(BaseModel):
    question: str
    experiment: dict[str, Any] | None = None


class Resume(BaseModel):
    action: str = ""
    plan: dict | None = None
    note_hash: str | None = None
    edited_note: str | None = None
    reason: str | None = None


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _brief(update: dict) -> dict:
    keep = {}
    for k in ("status", "message", "plan", "draft", "note", "note_hash", "output_check", "approval", "published", "guard"):
        if k in update:
            keep[k] = update[k]
    if "critic_reports" in update:
        keep["critic"] = update["critic_reports"][-1]
    if "findings" in update:
        keep["findings"] = update["findings"]
    if "quant_reports" in update:
        keep["quant"] = update["quant_reports"]
    if "plan_history" in update:
        keep["plan_reason"] = update["plan_history"][-1].get("reason")
    if isinstance(keep.get("approval"), dict):
        keep["approval"] = {k: v for k, v in keep["approval"].items() if k != "token"}
    return keep


async def _stream(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    t0 = time.time()
    try:
        async for ev in events:
            if "interrupt" in ev:
                yield _sse("interrupt", {"thread_id": ev["thread_id"], **ev["interrupt"]})
            else:
                yield _sse("update", {"thread_id": ev["thread_id"], "node": ev["node"], **_brief(ev["update"])})
    except Exception as exc:  # surface errors to the UI instead of a broken stream
        yield _sse("error", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    yield _sse("done", {"elapsed_s": round(time.time() - t0, 1)})


def _own(thread_id: str, who: str, values: dict) -> None:
    owner = OWNERS.get(thread_id) or values.get("analyst")
    if owner != who:
        raise HTTPException(404, "run not found")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": get_config()["version"]}


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/runs")
async def start_run(body: StartRun, who: str = Depends(analyst)) -> StreamingResponse:
    rt = app.state.rt
    thread = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    OWNERS[thread] = who
    return StreamingResponse(_stream(rt.start(body.question, who, body.experiment or {}, thread_id=thread)),
                             media_type="text/event-stream")


@app.post("/api/runs/{thread_id}/resume")
async def resume_run(thread_id: str, body: Resume, who: str = Depends(analyst)) -> StreamingResponse:
    rt = app.state.rt
    state = await rt.state(thread_id)
    _own(thread_id, who, state["values"])
    if not state["interrupts"]:
        raise HTTPException(409, "this run is not waiting for a decision")
    answer = {k: v for k, v in body.model_dump().items() if v is not None}
    answer["approver"] = who
    return StreamingResponse(_stream(rt.resume(thread_id, answer)), media_type="text/event-stream")


@app.get("/api/runs/{thread_id}")
async def get_run(thread_id: str, who: str = Depends(analyst)) -> dict:
    state = await app.state.rt.state(thread_id)
    v = state["values"]
    _own(thread_id, who, v)
    return {"thread_id": thread_id, "next": state["next"], "interrupts": state["interrupts"],
            **{k: v.get(k) for k in ("status", "message", "question", "plan", "plan_history", "draft", "note", "note_hash",
                                     "output_check", "critic_reports", "published", "usage")},
            "approval": {k: x for k, x in (v.get("approval") or {}).items() if k != "token"}}


@app.get("/api/runs/{thread_id}/trace")
async def get_trace(thread_id: str, who: str = Depends(analyst)) -> list[dict]:
    state = await app.state.rt.state(thread_id)
    _own(thread_id, who, state["values"])
    return read_trace(thread_id)


@app.get("/api/memory/prefs")
async def get_prefs(who: str = Depends(analyst)) -> dict:
    return await mem.get_prefs(app.state.rt.store, who)


@app.put("/api/memory/prefs")
async def put_prefs(prefs: dict, who: str = Depends(analyst)) -> dict:
    return await mem.set_prefs(app.state.rt.store, who, prefs)


@app.put("/api/memory/watchlist")
async def put_watchlist(body: dict, who: str = Depends(analyst)) -> dict:
    await mem.set_watchlist(app.state.rt.store, who, body.get("tickers", []))
    return {"saved": len(body.get("tickers", [])), "visible_to": "guardrails only"}


@app.get("/api/memory")
async def export_memory(who: str = Depends(analyst)) -> dict:
    return await mem.export_analyst(app.state.rt.store, who)


@app.delete("/api/memory")
async def forget(who: str = Depends(analyst)) -> dict:
    return {"deleted": await mem.forget_analyst(app.state.rt.store, who)}


@app.get("/api/notes")
async def notes(ticker: str | None = None, who: str = Depends(analyst)) -> dict:
    args = {"limit": 20, **({"ticker": ticker} if ticker else {})}
    return await mcp_client.call("research_portal", "list_past_notes", args)

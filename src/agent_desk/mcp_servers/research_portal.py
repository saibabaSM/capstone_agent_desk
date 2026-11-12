"""research-portal MCP server: the external system the desk publishes to. It has its own database.

Tools
  list_past_notes(ticker?, limit)                   [read]
  get_note(note_id)                                 [read]
  publish_note(note_markdown, tickers, approval)    [WRITE]  side effect: the note becomes visible

publish_note is the only tool in the system that changes the outside world. It refuses unless the
approval token verifies against the exact note it receives (control/approval.py); the server does
not trust the caller's word that the note was approved.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP
from pydantic import BaseModel

from ..config import data_dir
from ..control.approval import note_hash, verify_token
from .common import READ, WRITE, run_server

mcp = FastMCP("research-portal", instructions="Published research notes. publish_note needs a valid approval.")
_lock = threading.Lock()


def _db_path() -> Path:
    path = Path(os.getenv("PORTAL_DB", data_dir() / "portal" / "portal.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.execute("""CREATE TABLE IF NOT EXISTS notes (
        note_id INTEGER PRIMARY KEY AUTOINCREMENT, note_hash TEXT UNIQUE, tickers TEXT,
        markdown TEXT, approver TEXT, published_at REAL)""")
    return conn


class NoteSummary(BaseModel):
    note_id: int
    tickers: list[str]
    note_hash: str
    approver: str
    published_at: str
    excerpt: str


class NotesResult(BaseModel):
    ok: bool
    notes: list[NoteSummary] = []
    error: str = ""


class NoteResult(BaseModel):
    ok: bool
    note_id: int = 0
    markdown: str = ""
    tickers: list[str] = []
    error: str = ""


class Approval(BaseModel):
    note_hash: str
    approver: str
    issued_at: int
    signature: str


class PublishResult(BaseModel):
    ok: bool
    note_id: int = 0
    note_hash: str = ""
    already_published: bool = False
    error: str = ""


def _summary(row: tuple) -> NoteSummary:
    note_id, h, tickers, md, approver, ts = row
    return NoteSummary(note_id=note_id, tickers=json.loads(tickers), note_hash=h, approver=approver,
                       published_at=time.strftime("%Y-%m-%d %H:%M", time.gmtime(ts)), excerpt=md[:300])


@mcp.tool(annotations=READ)
def list_past_notes(ticker: Optional[str] = None, limit: int = 5) -> NotesResult:
    """List recently published notes, optionally for one ticker."""
    with _lock, _conn() as conn:
        rows = conn.execute("SELECT * FROM notes ORDER BY published_at DESC").fetchall()
    if ticker:
        rows = [r for r in rows if ticker.upper() in json.loads(r[2])]
    return NotesResult(ok=True, notes=[_summary(r) for r in rows[:max(1, min(limit, 20))]])


@mcp.tool(annotations=READ)
def get_note(note_id: int) -> NoteResult:
    """Full markdown of one published note."""
    with _lock, _conn() as conn:
        row = conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,)).fetchone()
    if row is None:
        return NoteResult(ok=False, error=f"note {note_id} not found")
    return NoteResult(ok=True, note_id=row[0], markdown=row[3], tickers=json.loads(row[2]))


@mcp.tool(annotations=WRITE)
def publish_note(note_markdown: str, tickers: list[str], approval: Optional[Approval] = None) -> PublishResult:
    """Publish a note to the research portal. Requires an approval issued for this exact note."""
    token: dict[str, Any] | None = approval.model_dump() if approval else None
    ok, why = verify_token(note_markdown, tickers, token)
    if not ok:
        return PublishResult(ok=False, error=f"publish refused: {why}")
    h = note_hash(note_markdown, tickers)
    with _lock, _conn() as conn:
        existing = conn.execute("SELECT note_id FROM notes WHERE note_hash = ?", (h,)).fetchone()
        if existing:
            return PublishResult(ok=True, note_id=existing[0], note_hash=h, already_published=True)
        cur = conn.execute("INSERT INTO notes (note_hash, tickers, markdown, approver, published_at) VALUES (?,?,?,?,?)",
                           (h, json.dumps(sorted(t.upper() for t in tickers)), note_markdown, token["approver"], time.time()))
        return PublishResult(ok=True, note_id=cur.lastrowid, note_hash=h)


if __name__ == "__main__":
    run_server(mcp, 8104)

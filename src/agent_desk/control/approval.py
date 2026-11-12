"""Approval tied to the exact payload (L14).

    h = note_hash(markdown, tickers)                  # sha256 of the canonical payload
    tok = issue_token(h, approver="alice")           # only the approval gate calls this, after a human says yes
    verify_token(markdown, tickers, tok)             # research-portal.publish_note calls this

The token is an HMAC over (hash, approver, issued_at) with APPROVAL_SECRET, a secret the agents
never see. publish_note recomputes the hash from the note it is asked to publish, so:
  * an edit after approval changes the hash -> the old token no longer verifies;
  * a model cannot mint a token (no secret) or reuse one for a different note.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

_DEV_SECRET = "dev-only-approval-secret-change-me"
TOKEN_TTL_SECONDS = 24 * 3600


def _secret() -> bytes:
    return os.getenv("APPROVAL_SECRET", _DEV_SECRET).encode()


def note_hash(markdown: str, tickers: list[str]) -> str:
    payload = json.dumps({"note": markdown.strip(), "tickers": sorted(t.upper() for t in tickers)},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _mac(h: str, approver: str, issued_at: int) -> str:
    return hmac.new(_secret(), f"{h}|{approver}|{issued_at}".encode(), hashlib.sha256).hexdigest()


def issue_token(h: str, approver: str) -> dict[str, Any]:
    issued_at = int(time.time())
    return {"note_hash": h, "approver": approver, "issued_at": issued_at, "signature": _mac(h, approver, issued_at)}


def verify_token(markdown: str, tickers: list[str], token: dict[str, Any] | None) -> tuple[bool, str]:
    if not token:
        return False, "missing approval token"
    try:
        h, approver, issued_at, sig = token["note_hash"], token["approver"], int(token["issued_at"]), token["signature"]
    except (KeyError, TypeError, ValueError):
        return False, "malformed approval token"
    if not hmac.compare_digest(_mac(h, approver, issued_at), str(sig)):
        return False, "approval signature is invalid"
    if time.time() - issued_at > TOKEN_TTL_SECONDS:
        return False, "approval has expired"
    actual = note_hash(markdown, tickers)
    if actual != h:
        return False, f"note changed after approval (approved {h[:12]}, submitted {actual[:12]})"
    return True, "ok"

"""Shared bits for the MCP servers: read/write annotations, structured errors, a runner."""

from __future__ import annotations

import argparse
from typing import Any

from pydantic import BaseModel

READ = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
READ_EXTERNAL = {**READ, "openWorldHint": True}  # reads, but may reach SEC on a cache miss
WRITE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}


class ToolError(BaseModel):
    """Errors are returned as data, never raised, so an agent can read and react to them."""
    ok: bool = False
    error: str
    hint: str = ""


def run_server(mcp: Any, default_port: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, nargs="?", const=default_port, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="http", host=args.host, port=args.http, show_banner=False)
    else:
        mcp.run(show_banner=False)

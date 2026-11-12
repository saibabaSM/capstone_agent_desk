"""Talk to the four MCP servers over the configured transport.

    MCP_TRANSPORT=inmemory (default)  servers run inside this process via FastMCP's in-memory transport
    MCP_TRANSPORT=stdio               each call spawns `python -m agent_desk.mcp_servers.<name>`
    MCP_TRANSPORT=http                MCP_<NAME>_URL, e.g. MCP_EDGAR_URL=http://edgar:8102/mcp (Docker)

The protocol is the same in all three; only the transport changes. (langchain-mcp-adapters 0.3.1
does not work with mcp 2.x, so tools/adapter.py converts MCP tools to LangChain tools itself; that
is also where the hooks sit.)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from typing import Any

from fastmcp import Client

SERVERS = {
    "filings_kb": "agent_desk.mcp_servers.filings_kb",
    "edgar": "agent_desk.mcp_servers.edgar",
    "python_sandbox": "agent_desk.mcp_servers.python_sandbox",
    "research_portal": "agent_desk.mcp_servers.research_portal",
}
DEFAULT_PORTS = {"filings_kb": 8101, "edgar": 8102, "python_sandbox": 8103, "research_portal": 8104}
_schemas: dict[str, list[Any]] = {}


def _target(server: str) -> Any:
    transport = os.getenv("MCP_TRANSPORT", "inmemory")
    if transport == "http":
        return os.getenv(f"MCP_{server.upper()}_URL", f"http://localhost:{DEFAULT_PORTS[server]}/mcp")
    if transport == "stdio":
        from fastmcp.client.transports import StdioTransport
        return StdioTransport(command=sys.executable, args=["-m", SERVERS[server]], env=dict(os.environ))
    return importlib.import_module(SERVERS[server]).mcp


async def list_tools(server: str) -> list[Any]:
    if server not in _schemas:
        async with Client(_target(server)) as client:
            _schemas[server] = await client.list_tools()
    return _schemas[server]


async def call(server: str, tool: str, args: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    """Call one tool; always returns a dict. Transport and validation failures become ok=False."""
    try:
        async with Client(_target(server), timeout=timeout) as client:
            result = await client.call_tool(tool, args, raise_on_error=False)
    except Exception as exc:  # server down, timeout, protocol error
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        return {"ok": False, "error": text[:500] or "tool error"}
    if result.structured_content is not None:
        return dict(result.structured_content)
    text = " ".join(getattr(c, "text", "") for c in result.content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"ok": True, "text": text}


def call_sync(server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(call(server, tool, args))

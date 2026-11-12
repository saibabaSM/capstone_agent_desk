"""python-sandbox MCP server: run short arithmetic in a restricted child process.

Tool
  run_python(code)   [read: no side effects]  -> {ok, output, error}

Isolation layers: AST allow-list (only math/statistics/fractions/decimal; no open/eval/getattr/
dunder access), a child process with an empty environment and a temp working directory, a 5 s time
limit and a 4000-character output cap (sandbox/python_sandbox.py). In Docker this server runs in
its own container on an internal network with no route to the internet, so even an escape could
not reach the network. pandas is deliberately NOT allowed: pd.read_csv("http://...") is a network
call, and ratios do not need it.
"""

from __future__ import annotations

from fastmcp import FastMCP
from pydantic import BaseModel

from ..sandbox.python_sandbox import run_code
from .common import READ, run_server

mcp = FastMCP("python-sandbox", instructions="Exact arithmetic. No files, no network, 5 s limit.")


class RunResult(BaseModel):
    ok: bool
    output: str = ""
    error: str = ""


@mcp.tool(annotations=READ)
def run_python(code: str) -> RunResult:
    """Run a short Python calculation and return what it prints.

    Use for every calculation. Copy input values exactly. Only math, statistics, fractions and
    decimal can be imported. Example: print(round(128528 / 281724 * 100, 1))
    """
    return RunResult(**run_code(code))


if __name__ == "__main__":
    run_server(mcp, 8103)

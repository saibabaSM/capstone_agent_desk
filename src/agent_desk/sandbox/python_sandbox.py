"""Run short, model-written Python calculations in a restricted child process.

Why it exists: a language model does arithmetic by guessing digits (it reported 42.0% where
88,523 / 211,915 is 41.8%). The quant agent hands every calculation to this module instead.

Flow of run_code(code):

    validate(code)            parent: parse the code and reject anything outside the allowed subset
        -> checked source     (a final bare expression such as `a / b * 100` becomes print(...))
    child process             python -I, empty environment, throw-away working folder
        - restricted builtins (no open, eval, exec, getattr, type, ...)
        - imports limited to ALLOWED_MODULES (math, statistics, fractions, decimal)
        - print() writes to a capped buffer
    parent waits up to TIMEOUT_SECONDS, then kills the child
        -> {"ok": bool, "output": "...", "error": "..."}

What this is and is not: it is a restriction layer for code written by our own model, not an
isolation boundary against a determined attacker. "No network" means the code cannot import
anything that opens a connection; it is not enforced by the operating system. The time limit and
the output cap are enforced. The memory used by the child is not limited on Windows (the time
limit bounds the damage). Do not point this at untrusted users; that needs a container or a
hosted code-execution service.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

# The child is killed after this many seconds.
TIMEOUT_SECONDS = 5.0
# print() output beyond this many characters raises an error and is cut off.
MAX_OUTPUT_CHARS = 4000
# Longer code is rejected without running.
MAX_CODE_CHARS = 5000

# The only modules the code may import. math, statistics, Fraction and Decimal are also preloaded,
# so code that forgets the import still works.
ALLOWED_MODULES = ("math", "statistics", "fractions", "decimal")

# The only builtins the code can see (print is replaced by a capped version in the child).
ALLOWED_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "range", "repr", "reversed", "round",
    "set", "sorted", "str", "sum", "tuple", "zip",
)

# Names rejected with a clear message instead of failing later with a NameError.
BLOCKED_NAMES = frozenset({
    "open", "eval", "exec", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "dir", "type", "object", "input", "breakpoint", "help", "memoryview", "bytearray", "super",
    "classmethod", "staticmethod", "property", "exit", "quit",
})

# Attributes that lead from a harmless object to interpreter internals (frames, code objects,
# class hierarchies) or let a format string read attributes. Any attribute starting with "_" is
# rejected as well.
BLOCKED_ATTRIBUTES = frozenset({
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "ag_frame",
    "ag_code", "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "f_trace", "tb_frame",
    "tb_next", "func_globals", "mro", "format", "format_map",
})

# Statements and expressions that are not needed for arithmetic and widen what the code can reach.
BLOCKED_NODES = (
    ast.ClassDef, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await, ast.Global,
    ast.Nonlocal, ast.Try, ast.With, ast.Yield, ast.YieldFrom,
)
if hasattr(ast, "TryStar"):
    BLOCKED_NODES += (ast.TryStar,)


class CodeRejected(ValueError):
    """The code uses something outside the allowed subset (the message says what and where)."""


def _reject(node: ast.AST, why: str) -> None:
    raise CodeRejected(f"line {getattr(node, 'lineno', '?')}: {why}")


# Checks the code against the allowed subset and returns the source to run, or raises CodeRejected.
# Checks, node by node:
#   - blocked statements (classes, try, with, yield, async, global);
#   - imports only from ALLOWED_MODULES;
#   - blocked names (open, eval, getattr, ...) and any name starting with "__";
#   - blocked attributes and any attribute starting with "_" (stops ().__class__ style escapes);
#   - string constants containing "__" (stops d["__builtins__"] style lookups).
# A final bare expression is turned into print(expression), so `88523 / 211915 * 100` shows its value.
# Example: validate("x = 3\nx * 2") -> "x = 3\nprint(x * 2)"
def validate(code: str) -> str:
    if len(code) > MAX_CODE_CHARS:
        raise CodeRejected(f"the code is longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeRejected(f"syntax error: {exc.msg} (line {exc.lineno})") from None

    for node in ast.walk(tree):
        if isinstance(node, BLOCKED_NODES):
            _reject(node, f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    _reject(node, f"import of '{alias.name}' is not allowed; use {', '.join(ALLOWED_MODULES)}")
                if (alias.asname or "").startswith("__"):
                    _reject(node, "names starting with '__' are not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                _reject(node, f"import from '{node.module}' is not allowed; use {', '.join(ALLOWED_MODULES)}")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                _reject(node, f"'{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BLOCKED_ATTRIBUTES:
                hint = "; use an f-string or format()" if node.attr in ("format", "format_map") else ""
                _reject(node, f"attribute '{node.attr}' is not allowed{hint}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            _reject(node, "strings containing '__' are not allowed")

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        already_prints = (
            isinstance(last.value, ast.Call)
            and isinstance(last.value.func, ast.Name)
            and last.value.func.id == "print"
        )
        if not already_prints:
            call = ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[last.value], keywords=[])
            tree.body[-1] = ast.copy_location(ast.Expr(value=call), last)
            ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# The program the child process runs. It reads one JSON object from stdin (code, allowed modules
# and builtins, output cap), executes the code with restricted builtins, and writes one JSON object
# to the real stdout: {"ok": bool, "output": str, "error": str}. The user code never sees sys, so it
# cannot reach the real stdout, and print() inside it only fills a capped buffer.
_RUNNER = r'''
import builtins, json, sys

cfg = json.loads(sys.stdin.read())
chunks = []
used = 0


class _OutputLimit(Exception):
    pass


def _print(*args, sep=" ", end="\n", **ignored):
    global used
    text = (" " if sep is None else sep).join(str(a) for a in args) + ("\n" if end is None else end)
    room = cfg["max_output"] - used
    if len(text) > room:
        chunks.append(text[: max(room, 0)])
        used = cfg["max_output"]
        raise _OutputLimit("output limit of %d characters exceeded" % cfg["max_output"])
    chunks.append(text)
    used += len(text)


allowed = set(cfg["modules"])
real_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in allowed:
        raise ImportError("import of %r is not allowed" % name)
    return real_import(name, globals, locals, fromlist, level)


safe = {name: getattr(builtins, name) for name in cfg["builtins"]}
safe["print"] = _print
safe["__import__"] = _import

import math, statistics
from fractions import Fraction
from decimal import Decimal

scope = {"__builtins__": safe, "math": math, "statistics": statistics, "Fraction": Fraction, "Decimal": Decimal}
result = {"ok": True, "error": ""}
try:
    exec(compile(cfg["code"], "<calculation>", "exec"), scope)
except BaseException as exc:
    result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
result["output"] = "".join(chunks)
sys.__stdout__.write(json.dumps(result))
'''


# Runs the code and always returns a dict (never raises for bad code):
#   {"ok": True,  "output": "41.77...\n", "error": ""}
#   {"ok": False, "output": "",           "error": "Code rejected: line 1: import of 'os' is not allowed ..."}
#   {"ok": False, "output": "",           "error": "Time limit of 5 seconds exceeded."}
#   {"ok": False, "output": "partial",    "error": "ZeroDivisionError: division by zero"}
# `timeout` overrides TIMEOUT_SECONDS (the tests use a short one).
def run_code(code: str, timeout: float | None = None) -> dict[str, Any]:
    limit = TIMEOUT_SECONDS if timeout is None else timeout
    try:
        checked = validate(code)
    except CodeRejected as exc:
        return {"ok": False, "output": "", "error": f"Code rejected: {exc}"}

    payload = json.dumps(
        {
            "code": checked,
            "modules": list(ALLOWED_MODULES),
            "builtins": list(ALLOWED_BUILTINS),
            "max_output": MAX_OUTPUT_CHARS,
        }
    )
    # -I ignores PYTHON* variables and the user's site folder; only SYSTEMROOT is passed because
    # Windows needs it to start Python. The working folder is empty and deleted afterwards.
    env = {name: os.environ[name] for name in ("SYSTEMROOT",) if name in os.environ}
    with tempfile.TemporaryDirectory() as workdir:
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER],
                input=payload,
                capture_output=True,
                text=True,
                timeout=limit,
                cwd=workdir,
                env=env,
            )
        except subprocess.TimeoutExpired:  # subprocess.run kills the child before raising
            return {"ok": False, "output": "", "error": f"Time limit of {limit:g} seconds exceeded."}

    try:
        result = json.loads(done.stdout)
    except json.JSONDecodeError:
        detail = done.stderr.strip()[-200:] or "no output"
        return {"ok": False, "output": "", "error": f"The calculation process failed: {detail}"}
    return {
        "ok": bool(result.get("ok")),
        "output": result.get("output", ""),
        "error": result.get("error", ""),
    }

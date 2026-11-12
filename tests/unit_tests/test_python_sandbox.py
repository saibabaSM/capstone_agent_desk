import json
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_desk.agents import quant_agent
from agent_desk.sandbox import python_sandbox as sb
from agent_desk.tools import sandbox_tools


class ToolFakeChat(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and ignores the schemas."""

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


# ---- Calculations that must work -------------------------------------------------------------


def test_arithmetic_is_exact():
    result = sb.run_code("print(round(88523 / 211915 * 100, 1))")

    assert result == {"ok": True, "output": "41.8\n", "error": ""}


def test_a_final_bare_expression_is_printed():
    assert sb.run_code("oi = 88523\nrev = 211915\nround(oi / rev * 100, 1)")["output"] == "41.8\n"
    assert sb.run_code("print('already printed')")["output"] == "already printed\n"  # not printed twice


def test_allowed_modules_and_preloaded_names_work():
    code = "import math\nfrom statistics import mean\nprint(math.sqrt(16), mean([1, 2, 3]), Fraction(1, 3) + Fraction(1, 6), Decimal('1.10') + Decimal('2.20'))"

    assert sb.run_code(code)["output"] == "4.0 2 1/2 3.30\n"
    assert sb.run_code("print(f'{41.77:.1f}%', format(0.5, '.0%'))")["output"] == "41.8% 50%\n"
    assert sb.run_code("print(math.floor(2.7))")["output"] == "2\n"  # math is preloaded


def test_functions_loops_and_comprehensions_work():
    code = "def growth(a, b):\n    return (b - a) / a * 100\nprint([round(growth(a, b), 1) for a, b in [(100, 110), (200, 180)]])"

    assert sb.run_code(code)["output"] == "[10.0, -10.0]\n"


def test_a_runtime_error_is_reported_without_raising():
    result = sb.run_code("print('before')\nprint(1 / 0)")

    assert result["ok"] is False
    assert result["error"] == "ZeroDivisionError: division by zero"
    assert result["output"] == "before\n"  # what was printed before the error is kept


# ---- Code that must be rejected before it runs ----------------------------------------------


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("import os", "import of 'os' is not allowed"),
        ("import socket", "import of 'socket' is not allowed"),
        ("from os import path", "import from 'os' is not allowed"),
        ("from . import x", "is not allowed"),
        ("import math as __builtins__", "names starting with '__'"),
        ("__import__('os')", "'__import__' is not allowed"),
        ("open('x.txt')", "'open' is not allowed"),
        ("eval('1+1')", "'eval' is not allowed"),
        ("exec('x=1')", "'exec' is not allowed"),
        ("getattr(math, 'pi')", "'getattr' is not allowed"),
        ("type(1)", "'type' is not allowed"),
        ("print(().__class__)", "attribute '__class__' is not allowed"),
        ("print(math._x)", "attribute '_x' is not allowed"),
        ("g = (i for i in [1])\nprint(g.gi_frame)", "attribute 'gi_frame' is not allowed"),
        ("g = (i for i in [1])\nprint(g.gi_frame.f_back.f_globals)", "is not allowed"),
        ("d = {}\nd['__builtins__']", "strings containing '__'"),
        ("print('{}'.format(1))", "use an f-string or format()"),
        ("class A:\n    pass", "ClassDef is not allowed"),
        ("try:\n    1 / 0\nexcept Exception:\n    pass", "Try is not allowed"),
        ("with x:\n    pass", "With is not allowed"),
        ("def f():\n    yield 1", "Yield is not allowed"),
        ("print(", "syntax error"),
        ("x = 1\n" * 2000, "longer than"),
    ],
)
def test_forbidden_code_is_rejected_without_running(code, fragment):
    result = sb.run_code(code)

    assert result["ok"] is False and result["output"] == ""
    assert result["error"].startswith("Code rejected:") and fragment in result["error"]


def test_validate_returns_the_source_with_the_last_expression_printed():
    assert sb.validate("x = 3\nx * 2") == "x = 3\nprint(x * 2)"
    assert sb.validate("print(1)") == "print(1)"


# ---- Limits ----------------------------------------------------------------------------------


def test_an_infinite_loop_is_stopped_by_the_time_limit():
    result = sb.run_code("while True:\n    pass", timeout=1)

    assert result == {"ok": False, "output": "", "error": "Time limit of 1 seconds exceeded."}


def test_output_is_cut_off_at_the_cap():
    result = sb.run_code("for i in range(100000):\n    print('x' * 100)")

    assert result["ok"] is False and "output limit of 4000 characters exceeded" in result["error"]
    assert len(result["output"]) == sb.MAX_OUTPUT_CHARS


def test_each_call_starts_clean():
    sb.run_code("leftover = 123")

    result = sb.run_code("print(leftover)")

    assert result["ok"] is False and "NameError" in result["error"]


def test_a_crashing_child_is_reported(monkeypatch):
    monkeypatch.setattr(sb, "_RUNNER", "import sys; sys.exit(3)")

    result = sb.run_code("print(1)")

    assert result["ok"] is False and result["error"].startswith("The calculation process failed")


# ---- The tool and the agent ------------------------------------------------------------------


def test_the_tool_returns_json():
    ok = json.loads(sandbox_tools.run_python.invoke({"code": "print(2 + 2)"}))
    bad = json.loads(sandbox_tools.run_python.invoke({"code": "import os"}))

    assert ok == {"ok": True, "output": "4\n", "error": ""}
    assert bad["ok"] is False and "not allowed" in bad["error"]


def test_the_quant_agent_has_the_calculator_and_uses_it_in_a_tool_loop():
    assert sandbox_tools.run_python in quant_agent.TOOLS
    assert "never do arithmetic yourself" in quant_agent.SYSTEM_PROMPT

    call = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"code": "print(round(88523 / 211915 * 100, 1))"}, "id": "call_1"}],
    )
    final = AIMessage(content="Operating margin: 88523 / 211915 = 41.8%")
    agent = quant_agent.build_quant_agent(ToolFakeChat(messages=iter([call, final])))

    result = agent.invoke({"messages": [("user", "Compute the margin")]})

    kinds = [m.type for m in result["messages"]]
    tool_message = json.loads(result["messages"][kinds.index("tool")].content)
    assert tool_message["output"] == "41.8\n"
    assert result["messages"][-1].content == final.content

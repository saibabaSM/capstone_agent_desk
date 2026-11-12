"""Turn the MCP tools an agent is allowed to use into LangChain tools that go through the hooks.

    tools = await build_tools("quant", ctx)     # only edgar.* and python_sandbox.run_python
    agent = create_react_agent(model, tools, ...)

Tool names are "<server>__<tool>" (OpenAI function names cannot contain dots). The input schema
is the MCP server's own JSON schema, so the model sees exactly the typed interface the server
validates. Results go back to the model as compact JSON wrapped in <untrusted> tags.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from ..control.context import RunContext
from ..control.hooks import guarded_call
from . import mcp_client

MAX_RESULT_CHARS = 9000


def _render(result: dict[str, Any]) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + '..."(truncated)"'
    return f"<untrusted source=\"tool\">{text}</untrusted>"


async def build_tools(agent: str, ctx: RunContext, only: list[str] | None = None) -> list[StructuredTool]:
    allowed = [q for q in ctx.cfg["agents"]["permissions"].get(agent, []) if not only or q in only]
    tools: list[StructuredTool] = []
    for qualified in allowed:
        server, name = qualified.split(".", 1)
        schema = next((t for t in await mcp_client.list_tools(server) if t.name == name), None)
        if schema is None:
            continue

        async def _call(_q: str = qualified, **kwargs: Any) -> str:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            return _render(await guarded_call(ctx, agent, _q, clean))

        input_schema = dict(getattr(schema, "input_schema", None) or schema.inputSchema)
        input_schema.pop("additionalProperties", None)
        tools.append(StructuredTool.from_function(
            coroutine=_call, name=f"{server}__{name}",
            description=(schema.description or name)[:1000],
            args_schema=input_schema))
    return tools

from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_desk.agents import researcher_agent
from agent_desk.llm import llm
from agent_desk.tools import rag_tools


class ToolFakeChat(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and ignores the schemas."""

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def test_get_llm_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm.get_llm(llm.TOOL_MODEL)


def test_models_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert llm.get_tool_llm().model_name == "meta-llama/llama-3.3-70b-instruct"
    assert llm.get_fast_llm().model_name == "meta-llama/llama-3.1-8b-instruct"


def test_researcher_runs_tool_loop(monkeypatch):
    document = Document(
        page_content="Risk factors mention supply chain risk.",
        metadata={"company": "apple", "year": "2025", "item": "Item 1A"},
    )

    def fake_search_chunks(query, k, company=None, year=None):
        assert query == "risk factors"
        assert k == 4
        return [document]

    monkeypatch.setattr(rag_tools, "search_chunks", fake_search_chunks)
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_filing", "args": {"query": "risk factors", "k": 4}, "id": "call_1"}],
    )
    final = AIMessage(content="Risk factors mention supply chain risk.")
    fake = ToolFakeChat(messages=iter([tool_call, final]))

    agent = researcher_agent.build_researcher(fake)
    result = agent.invoke({"messages": [("user", "What are the risk factors?")]})

    kinds = [m.type for m in result["messages"]]
    assert "tool" in kinds
    assert "supply chain" in result["messages"][kinds.index("tool")].content
    assert result["messages"][-1].content == final.content

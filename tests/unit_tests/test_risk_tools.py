import json
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_desk.agents import risk_diff_agent
from agent_desk.tools import risk_tools


class ToolFakeChat(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and ignores the schemas."""

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def finding(description, ids):
    return {"description": description, "passage_ids": ids, "quote": "quote", "evidence": []}


def fake_diff(added=(), removed=(), changed=(), rejected=()):
    return {
        "company": "microsoft",
        "year_from": 2024,
        "year_to": 2025,
        "added": list(added),
        "removed": list(removed),
        "changed": list(changed),
        "rejected": list(rejected),
    }


def test_tool_returns_findings_counts_and_rejected_count(monkeypatch):
    calls = []
    diff = fake_diff(
        added=[finding("New AI risk", ["microsoft-2025-1A-5"])],
        changed=[finding("Regulation widened", ["microsoft-2024-1A-9", "microsoft-2025-1A-9"])],
        rejected=[{"kind": "added", "description": "x", "reason": "y"}] * 3,
    )
    monkeypatch.setattr(risk_tools, "compare_company_years", lambda *a: calls.append(a) or diff)

    result = json.loads(
        risk_tools.compare_risk_factors.invoke({"company": "microsoft", "year_from": 2025, "year_to": 2024})
    )

    assert calls == [("microsoft", 2025, 2024)]  # years are passed through; the comparison orders them
    assert result["ok"] is True
    assert result["counts"] == {"added": 1, "removed": 0, "changed": 1}
    assert result["added"][0]["passage_ids"] == ["microsoft-2025-1A-5"]
    assert result["rejected_count"] == 3
    assert "rejected" not in result and "note" not in result


def test_tool_notes_when_nothing_changed(monkeypatch):
    monkeypatch.setattr(risk_tools, "compare_company_years", lambda *a: fake_diff())

    result = json.loads(
        risk_tools.compare_risk_factors.invoke({"company": "microsoft", "year_from": 2024, "year_to": 2025})
    )

    assert result["ok"] is True and result["counts"] == {"added": 0, "removed": 0, "changed": 0}
    assert result["note"] == "No material differences were found between the two years."


def test_tool_reports_missing_filing_and_bad_years_as_errors(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("No alphabet 10-K for fiscal year 2026. Available years: 2023, 2024, 2025.")

    monkeypatch.setattr(risk_tools, "compare_company_years", missing)
    result = json.loads(
        risk_tools.compare_risk_factors.invoke({"company": "alphabet", "year_from": 2025, "year_to": 2026})
    )
    assert result["ok"] is False and "Available years: 2023, 2024, 2025" in result["error"]

    def same_year(*args):
        raise ValueError("Pick two different fiscal years to compare.")

    monkeypatch.setattr(risk_tools, "compare_company_years", same_year)
    result = json.loads(
        risk_tools.compare_risk_factors.invoke({"company": "alphabet", "year_from": 2025, "year_to": 2025})
    )
    assert result["ok"] is False and "two different fiscal years" in result["error"]


def test_risk_diff_agent_runs_tool_loop(monkeypatch):
    monkeypatch.setattr(
        risk_tools,
        "compare_company_years",
        lambda *a: fake_diff(removed=[finding("Goodwill impairment risk", ["microsoft-2024-1A-28"])]),
    )
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "compare_risk_factors",
                "args": {"company": "microsoft", "year_from": 2024, "year_to": 2025},
                "id": "call_1",
            }
        ],
    )
    final = AIMessage(content="Removed: goodwill impairment risk (microsoft-2024-1A-28).")
    fake = ToolFakeChat(messages=iter([tool_call, final]))

    agent = risk_diff_agent.build_risk_diff_agent(fake)
    result = agent.invoke({"messages": [("user", "What changed in Microsoft's risk factors between 2024 and 2025?")]})

    kinds = [m.type for m in result["messages"]]
    tool_message = result["messages"][kinds.index("tool")]
    assert "Goodwill impairment risk" in tool_message.content
    assert result["messages"][-1].content == final.content


def test_run_risk_diff_agent_returns_last_message(monkeypatch):
    final = AIMessage(content="No material differences were found.")
    fake = ToolFakeChat(messages=iter([final]))

    assert risk_diff_agent.run_risk_diff_agent("What changed?", llm=fake) == final.content

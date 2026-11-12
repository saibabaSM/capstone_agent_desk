"""One test per MCP tool (L7 evidence): typed output, structured errors instead of crashes."""

import asyncio

import pytest

from agent_desk.control.approval import issue_token, note_hash
from agent_desk.tools import mcp_client

from .conftest import needs_facts, needs_index


def call(server, tool, args):
    return asyncio.run(mcp_client.call(server, tool, args))


def test_tool_annotations_mark_read_and_write():
    tools = {s: asyncio.run(mcp_client.list_tools(s)) for s in mcp_client.SERVERS}
    for server, items in tools.items():
        for t in items:
            ro = t.annotations.read_only_hint
            if (server, t.name) == ("research_portal", "publish_note"):
                assert ro is False, "publish_note must be marked as a write tool"
            else:
                assert ro is True, f"{server}.{t.name} should be read-only"
            schema = getattr(t, "output_schema", None) or {}
            assert schema.get("properties"), f"{server}.{t.name} has no typed output"


# ---- filings-kb
@needs_index
def test_filings_kb_search_filings():
    r = call("filings_kb", "search_filings", {"query": "operating income increased", "company": "MSFT", "fiscal_year": 2025, "item": "7", "k": 3})
    assert r["ok"] and len(r["passages"]) == 3
    assert all(p["passage_id"].startswith("microsoft-2025-7-") for p in r["passages"])


def test_filings_kb_search_rejects_uncovered_company():
    r = call("filings_kb", "search_filings", {"query": "margins", "company": "TSLA"})
    assert r["ok"] is False and "coverage" in r["error"]


@needs_index
def test_filings_kb_get_passage():
    first = call("filings_kb", "search_filings", {"query": "risk", "company": "GOOGL", "k": 1})["passages"][0]
    r = call("filings_kb", "get_passage", {"passage_id": first["passage_id"]})
    assert r["ok"] and r["passage"]["text"] == first["text"]
    assert call("filings_kb", "get_passage", {"passage_id": "nope-2025-1A-0001"})["ok"] is False


@needs_index
def test_filings_kb_diff_risk_factors_and_missing_year():
    r = call("filings_kb", "diff_risk_factors", {"company": "MSFT", "year_from": 2025, "year_to": 2026})
    assert r["ok"] and r["counts"]["paragraphs_new"] > 10
    bad = call("filings_kb", "diff_risk_factors", {"company": "META", "year_from": 2023, "year_to": 2024})
    assert bad["ok"] is False and bad["available_years"]


# ---- edgar
@needs_facts
def test_edgar_get_metric_derived_has_inputs():
    r = call("edgar", "get_metric", {"company": "MSFT", "metric": "operating_margin", "fiscal_years": [2025]})
    derived = [f for f in r["facts"] if f["kind"] == "derived"][0]
    assert derived["display"] == "45.6%" and len(derived["inputs"]) == 2
    assert derived["fact_id"] == "MSFT/operating_margin/FY2025/calc"


@needs_facts
def test_edgar_get_metric_missing_year_is_structured():
    r = call("edgar", "get_metric", {"company": "GOOGL", "metric": "revenue", "fiscal_years": [2026]})
    assert r["ok"] is False and "available fiscal years" in r["errors"][0]


@needs_facts
def test_edgar_get_xbrl_fact_and_list_concepts():
    r = call("edgar", "get_xbrl_fact", {"company": "MSFT", "concept": "OperatingIncomeLoss", "fiscal_year": 2025})
    assert r["ok"] and r["facts"][0]["value"] == 128528000000.0
    c = call("edgar", "list_concepts", {"company": "MSFT", "search": "OperatingIncome"})
    assert c["ok"] and any(x["concept"] == "OperatingIncomeLoss" for x in c["concepts"])


@needs_facts
def test_edgar_list_filings_and_get_section():
    f = call("edgar", "list_filings", {"company": "NVDA"})
    assert f["ok"] and len(f["filings"]) >= 2
    s = call("edgar", "get_section", {"company": "NVDA", "fiscal_year": f["filings"][0]["fiscal_year"], "item": "1A", "max_chars": 1000})
    assert s["ok"] and len(s["text"]) <= 1000


def test_edgar_validation_error_is_returned_not_raised():
    r = call("edgar", "get_metric", {"company": "MSFT", "metric": "operating_margin", "fiscal_years": "not-a-list"})
    assert r["ok"] is False and r["error"]


# ---- python-sandbox
def test_sandbox_runs_arithmetic():
    r = call("python_sandbox", "run_python", {"code": "print(round(128528 / 281724 * 100, 1))"})
    assert r == {"ok": True, "output": "45.6\n", "error": ""}


@pytest.mark.parametrize("code", ["import socket", "import urllib.request", "open('/etc/passwd')", "().__class__"])
def test_sandbox_blocks_escape(code):
    assert call("python_sandbox", "run_python", {"code": code})["ok"] is False


# ---- research-portal
def test_portal_publish_requires_exact_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTAL_DB", str(tmp_path / "portal.db"))
    note = "## Summary\nMargin 45.6% [F1]."
    assert call("research_portal", "publish_note", {"note_markdown": note, "tickers": ["MSFT"]})["ok"] is False
    token = issue_token(note_hash(note, ["MSFT"]), "alice")
    stale = call("research_portal", "publish_note", {"note_markdown": note + " Edited.", "tickers": ["MSFT"], "approval": token})
    assert stale["ok"] is False and "changed after approval" in stale["error"]
    ok = call("research_portal", "publish_note", {"note_markdown": note, "tickers": ["MSFT"], "approval": token})
    assert ok["ok"] and ok["note_id"] >= 1


def test_portal_list_and_get(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTAL_DB", str(tmp_path / "portal.db"))
    note = "## Summary\nA note."
    call("research_portal", "publish_note", {"note_markdown": note, "tickers": ["AAPL"], "approval": issue_token(note_hash(note, ["AAPL"]), "bob")})
    listed = call("research_portal", "list_past_notes", {"ticker": "AAPL"})
    assert listed["ok"] and listed["notes"][0]["approver"] == "bob"
    got = call("research_portal", "get_note", {"note_id": listed["notes"][0]["note_id"]})
    assert got["markdown"] == note

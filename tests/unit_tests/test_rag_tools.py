import json

import pytest
from langchain_core.documents import Document

from agent_desk.tools import rag_tools


def test_search_filing_tool_passes_filters_and_returns_metadata(monkeypatch):
    document = Document(
        page_content="Supply chain risk factors.",
        metadata={"company": "apple", "year": "2025", "item": "Item 1A"},
    )
    calls = []

    def fake_search_chunks(query, k, company=None, year=None):
        calls.append((query, k, company, year))
        return [document]

    monkeypatch.setattr(rag_tools, "search_chunks", fake_search_chunks)

    result = json.loads(
        rag_tools.search_filing.invoke(
            {"query": "supply chain", "company": "apple", "year": 2025, "k": 1}
        )
    )

    assert calls == [("supply chain", 1, "apple", 2025)]
    assert result == [
        {
            "text": "Supply chain risk factors.",
            "metadata": {"company": "apple", "year": "2025", "item": "Item 1A"},
        }
    ]


@pytest.mark.parametrize(
    ("query", "year", "k", "message"),
    [
        (" ", None, 4, "must not be empty"),
        ("risk", None, 0, "between 1 and 10"),
        ("risk", 25, 4, "four-digit fiscal year"),
    ],
)
def test_search_filing_rejects_invalid_input(query, year, k, message):
    with pytest.raises(ValueError, match=message):
        rag_tools.search_filing.func(query, year=year, k=k)

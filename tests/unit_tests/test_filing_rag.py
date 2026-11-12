import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from agent_desk.rag import filing_rag


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    embeddings = FakeEmbeddings(size=16)
    docs = [
        Document(page_content=f"{company} {year} risk {i}", metadata={"company": company, "year": year})
        for i in range(4)
        for company in ("apple", "microsoft")
        for year in ("2024", "2025")
    ]
    FAISS.from_documents(docs, embeddings).save_local(str(tmp_path))
    monkeypatch.setattr(filing_rag, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(filing_rag, "_embeddings", embeddings)
    monkeypatch.setattr(filing_rag, "_store_cache", None)
    return tmp_path


def test_search_filters_by_company_and_year(index_dir):
    results = filing_rag.search_chunks("risk", k=3, company="Apple", year=2025)

    assert len(results) == 3
    assert {(d.metadata["company"], d.metadata["year"]) for d in results} == {("apple", "2025")}


def test_search_without_filters_and_unknown_company(index_dir):
    assert len(filing_rag.search_chunks("risk", k=5)) == 5
    assert filing_rag.search_chunks("risk", company="google") == []


def test_search_without_index_explains_how_to_build(tmp_path, monkeypatch):
    monkeypatch.setattr(filing_rag, "INDEX_DIR", tmp_path / "missing")
    monkeypatch.setattr(filing_rag, "_store_cache", None)

    with pytest.raises(FileNotFoundError, match="create_embeddings"):
        filing_rag.search_chunks("risk")

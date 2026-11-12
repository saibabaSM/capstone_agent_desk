"""Build and maintain the FAISS index of 10-K filings under data/filings/<company>/<year>/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

# Lets this file run directly (python create_embeddings.py or F5) outside the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from agent_desk.rag.filing_parser import load_passages  # noqa: E402
from agent_desk.rag.filing_rag import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_DIR,
    PROJECT_ROOT,
    _get_embeddings,
)

FILINGS_DIR = PROJECT_ROOT / "data" / "filings"
FILING_SUFFIXES = {".htm", ".html"}


def _find_filings(company: str | None = None) -> list[Path]:
    """Return filing files as data/filings/<company>/<year>/<file>, optionally for one company."""
    wanted = company.strip().lower() if company else None
    files = []
    for path in sorted(FILINGS_DIR.glob("*/*/*")):
        if path.suffix.lower() not in FILING_SUFFIXES:
            continue
        if wanted is None or path.parent.parent.name.lower() == wanted:
            files.append(path)
    return files


def _chunk_filing(path: Path) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    documents = [
        Document(
            page_content=passage.text,
            metadata={
                "company": path.parent.parent.name.lower(),
                "year": path.parent.name,
                "item": passage.item,
                "source": str(path.resolve()),
            },
        )
        for passage in load_passages(path)
    ]
    return splitter.split_documents(documents)


def _chunk_filings(files: list[Path]) -> tuple[list[Document], dict[str, int]]:
    chunks: list[Document] = []
    per_company: dict[str, int] = {}
    for path in files:
        file_chunks = _chunk_filing(path)
        chunks.extend(file_chunks)
        company = path.parent.parent.name.lower()
        per_company[company] = per_company.get(company, 0) + len(file_chunks)
    return chunks, per_company


def _load_store() -> FAISS | None:
    if not (INDEX_DIR / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _save_store(store: FAISS) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))


def _ids_for_company(store: FAISS, company: str) -> list[str]:
    wanted = company.strip().lower()
    return [
        doc_id
        for doc_id, doc in store.docstore._dict.items()
        if doc.metadata.get("company", "").lower() == wanted
    ]


def delete_all_embeddings() -> dict[str, Any]:
    """Remove the whole index."""
    existed = INDEX_DIR.exists()
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    return {"status": "deleted" if existed else "nothing_to_delete"}


def delete_company_embeddings(company_name: str) -> dict[str, Any]:
    """Remove every chunk of one company; the index folder goes away if it ends up empty."""
    store = _load_store()
    if store is None:
        return {"status": "nothing_to_delete", "company": company_name}
    ids = _ids_for_company(store, company_name)
    if not ids:
        return {"status": "nothing_to_delete", "company": company_name}
    if len(ids) == len(store.docstore._dict):
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    else:
        store.delete(ids)
        _save_store(store)
    return {"status": "deleted", "company": company_name, "chunks_removed": len(ids)}


def create_all_embeddings() -> dict[str, Any]:
    """Delete the existing index and rebuild it from every filing under data/filings."""
    files = _find_filings()
    if not files:
        return {"status": "no_filings_found", "filings_dir": str(FILINGS_DIR)}
    chunks, per_company = _chunk_filings(files)
    if not chunks:
        return {"status": "no_text_found", "files": len(files)}
    store = FAISS.from_documents(chunks, _get_embeddings())
    delete_all_embeddings()
    _save_store(store)
    return {
        "status": "created",
        "files": len(files),
        "chunks": len(chunks),
        "chunks_per_company": per_company,
    }


def create_company_embeddings(company_name: str) -> dict[str, Any]:
    """Replace one company's embeddings with fresh ones, keeping the other companies."""
    files = _find_filings(company_name)
    if not files:
        return {"status": "no_filings_found", "company": company_name}
    chunks, per_company = _chunk_filings(files)
    if not chunks:
        return {"status": "no_text_found", "company": company_name}

    store = _load_store()
    if store is not None:
        existing = _ids_for_company(store, company_name)
        if existing:
            store.delete(existing)
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, _get_embeddings())
    _save_store(store)
    return {
        "status": "created",
        "company": company_name,
        "files": len(files),
        "chunks": len(chunks),
        "chunks_per_company": per_company,
    }


def main() -> None:
    # Uncomment the operation to run.
    # print(create_all_embeddings())
    print(create_company_embeddings("alphabet"))
    # print(delete_company_embeddings("microsoft"))
    # print(delete_all_embeddings())
    pass


if __name__ == "__main__":
    main()

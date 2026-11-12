"""Search the prebuilt FAISS index of 10-K filings (built by utils/create_embeddings.py)."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_embeddings: HuggingFaceEmbeddings | None = None
_store_cache: tuple[int, FAISS] | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _load_store() -> FAISS:
    """Load the saved index, reloading only if create_embeddings rewrote it."""
    global _store_cache
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"No embeddings found in {INDEX_DIR}. "
            "Run agent_desk/utils/create_embeddings.py first."
        )
    version = index_file.stat().st_mtime_ns
    if _store_cache is None or _store_cache[0] != version:
        store = FAISS.load_local(
            str(INDEX_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        _store_cache = (version, store)
    return _store_cache[1]


def search_chunks(
    query: str,
    k: int = 4,
    company: str | None = None,
    year: int | str | None = None,
) -> list[Document]:
    """Return the k most similar chunks, optionally limited to one company and/or year."""
    store = _load_store()
    metadata_filter: dict[str, str] = {}
    if company:
        metadata_filter["company"] = company.strip().lower()
    if year:
        metadata_filter["year"] = str(year)
    if not metadata_filter:
        return store.similarity_search(query, k=k)
    # Search every vector before filtering so the filter cannot starve the results.
    return store.similarity_search(
        query, k=k, filter=metadata_filter, fetch_k=store.index.ntotal
    )

"""Deterministic Item 1A diff between two fiscal years (no LLM, a few seconds).

The starter's rag/risk_compare.py asks an LLM to propose changes and then verifies them; it is
accurate but takes minutes per company. This module is the fast default used by the risk-diff
agent's tool; the agent (an LLM) then summarises and prioritises what the code found.

Method: split both years' Item 1A into paragraphs, embed them, and for each paragraph find its
most similar paragraph in the other year.
    best similarity <  ADDED_BELOW  in the newer year -> added
    best similarity <  ADDED_BELOW  in the older year -> removed
    ADDED_BELOW <= sim < SAME_ABOVE                  -> changed (old and new shown side by side)
Each paragraph is mapped to the index passage (chunk) that contains it, so every finding cites a
passage_id that exists in filings-kb.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..companies import resolve
from ..config import data_dir
from .hybrid import embedder, load_index
from .structured_parser import parse_filing

ADDED_BELOW = 0.72
SAME_ABOVE = 0.95
MIN_CHARS = 120


def _filing_path(company_key: str, year: int):
    folder = data_dir() / "filings" / company_key / str(year)
    files = sorted(p for p in folder.glob("*.htm*")) if folder.is_dir() else []
    return files[0] if files else None


def available_years(company_key: str) -> list[int]:
    root = data_dir() / "filings" / company_key
    return sorted(int(p.name) for p in root.iterdir() if p.name.isdigit() and _filing_path(company_key, int(p.name))) if root.is_dir() else []


@lru_cache(maxsize=48)
def _paragraphs(company_key: str, year: int) -> tuple[list[str], np.ndarray, list[str]]:
    path = _filing_path(company_key, year)
    if path is None:
        raise FileNotFoundError(f"no 10-K for {company_key} FY{year}; available: {available_years(company_key)}")
    paras = [b.text for b in parse_filing(path) if b.item == "1A" and b.kind == "text" and len(b.text) >= MIN_CHARS]
    if not paras:
        raise ValueError(f"no Item 1A paragraphs found for {company_key} FY{year}")
    vecs = embedder().encode(paras, normalize_embeddings=True, batch_size=64)
    index = load_index(True)
    chunks = [d for d in index.docs if d["company"] == company_key and d["year"] == year and d["item"] == "1A"]
    ids = []
    for p in paras:
        probe = p[:80]
        ids.append(next((c["passage_id"] for c in chunks if probe in c["text"]), chunks[0]["passage_id"] if chunks else ""))
    return paras, np.asarray(vecs), ids


def _headline(text: str, words: int = 28) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def diff_risk_factors(company: str, year_from: int, year_to: int, max_items: int = 6) -> dict[str, Any]:
    comp = resolve(company)
    if comp is None:
        return {"ok": False, "error": f"unknown company '{company}'"}
    if year_from > year_to:
        year_from, year_to = year_to, year_from
    try:
        old_p, old_v, old_ids = _paragraphs(comp.key, year_from)
        new_p, new_v, new_ids = _paragraphs(comp.key, year_to)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "error": str(exc), "available_years": available_years(comp.key)}
    sim = new_v @ old_v.T  # [new, old]
    best_old = sim.max(axis=1)
    best_new = sim.max(axis=0)
    added = [i for i in np.argsort(best_old) if best_old[i] < ADDED_BELOW]
    removed = [j for j in np.argsort(best_new) if best_new[j] < ADDED_BELOW]
    changed = [i for i in np.argsort(best_old) if ADDED_BELOW <= best_old[i] < SAME_ABOVE]

    def item(text: str, pid: str, s: float, extra: dict | None = None) -> dict[str, Any]:
        return {"summary": _headline(text), "passage_id": pid, "similarity_to_other_year": round(float(s), 3), **(extra or {})}

    return {
        "ok": True,
        "company": comp.key,
        "ticker": comp.ticker,
        "year_from": year_from,
        "year_to": year_to,
        "counts": {"added": len(added), "removed": len(removed), "changed": len(changed),
                   "paragraphs_old": len(old_p), "paragraphs_new": len(new_p)},
        "added": [item(new_p[i], new_ids[i], best_old[i]) for i in added[:max_items]],
        "removed": [item(old_p[j], old_ids[j], best_new[j]) for j in removed[:max_items]],
        "changed": [item(new_p[i], new_ids[i], best_old[i],
                         {"previous_passage_id": old_ids[int(sim[i].argmax())],
                          "previous_summary": _headline(old_p[int(sim[i].argmax())])})
                    for i in changed[:max_items]],
        "method": f"embedding similarity: added/removed < {ADDED_BELOW}, changed {ADDED_BELOW}-{SAME_ABOVE}",
    }

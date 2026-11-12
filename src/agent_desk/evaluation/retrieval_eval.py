"""Recall@5 on labelled queries (eval/retrieval/queries.jsonl), before/after each retrieval change.

A query is a hit if any of the top-5 passages is from the gold company + fiscal year + Item and
contains one of the gold phrases (compared after removing table pipes and extra spaces, so the
same phrase matches in table-aware and plain chunks). Company and fiscal year are given as filters,
exactly as the researcher agent searches (the plan fixes both); the Item is NOT given.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..config import PROJECT_ROOT
from ..rag import hybrid

QUERIES = PROJECT_ROOT / "eval" / "retrieval" / "queries.jsonl"
CONFIGS = [
    ("bm25 only", {"use_vectors": False, "use_reranker": False, "query_technique": "none"}),
    ("vectors only", {"use_bm25": False, "use_reranker": False, "query_technique": "none"}),
    ("hybrid (RRF)", {"use_reranker": False, "query_technique": "none"}),
    ("hybrid + rerank", {"use_reranker": True, "query_technique": "none"}),
    ("hybrid + rerank + decomposition", {"use_reranker": True, "query_technique": "decomposition"}),
    ("hybrid + rerank + multi-query (LLM rewrite)", {"use_reranker": True, "query_technique": "multi_query"}),
    ("plain chunks: hybrid + rerank + decomposition", {"use_reranker": True, "query_technique": "decomposition", "table_aware": False}),
]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("|", " ").replace("’", "'")).lower()


def hit(passages: list[dict], gold: dict) -> bool:
    for p in passages:
        if p["company"] == gold["company"] and p["year"] == gold["year"] and p["item"] == gold["item"]:
            if any(_norm(ph) in _norm(p["text"]) for ph in gold["phrases"]):
                return True
    return False


def evaluate(settings: dict, queries: list[dict], k: int = 5) -> dict:
    hits, table_hits, n_table, t0 = 0, 0, 0, time.time()
    misses = []
    for q in queries:
        res = hybrid.search(q["query"], company=q["company"], year=q["gold"]["year"], k=k, source_types=("filing",), settings=settings)
        ok = hit(res, q["gold"])
        hits += ok
        if q.get("needs_table"):
            n_table += 1
            table_hits += ok
        if not ok:
            misses.append(q["id"])
    return {"recall@5": round(hits / len(queries), 3), "table_queries_recall@5": round(table_hits / n_table, 3) if n_table else None,
            "queries": len(queries), "ms_per_query": round((time.time() - t0) * 1000 / len(queries)), "misses": misses}


def main(out: str | None = None) -> int:
    queries = [json.loads(l) for l in QUERIES.open(encoding="utf-8") if l.strip()]
    rows = []
    for name, settings in CONFIGS:
        r = evaluate(settings, queries)
        rows.append({"config": name, **r})
        print(json.dumps(rows[-1]), flush=True)
    md = ["| Retrieval config | Recall@5 | Recall@5 (table queries) | ms/query | misses |", "|---|---|---|---|---|"]
    md += [f"| {r['config']} | {r['recall@5']} | {r['table_queries_recall@5']} | {r['ms_per_query']} | {', '.join(r['misses'])} |" for r in rows]
    text = "\n".join(md)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    return 0

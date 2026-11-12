"""Judge calibration: does the claim judge (a different model family) agree with humans?

1. sample:    python -m agent_desk eval judge-sample --results eval/results/<run>.jsonl
              -> eval/judge/to_label.jsonl  (claim + cited passage text, no verdict shown)
2. HUMANS label each line: copy to eval/judge/labels.jsonl and set "human": "supported" | "not_supported".
   Two people label independently ("human", "human_2") so human-human agreement is known too.
   (These labels must come from people; the tooling never fills them in.)
3. agreement: python -m agent_desk eval judge-agreement   -> judge verdicts + Cohen's kappa
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from ..config import PROJECT_ROOT, get_config, is_offline
from ..control.context import RunContext
from ..rag import hybrid
from ..agents.critic import claim_sentences, judge_claims

JUDGE_DIR = PROJECT_ROOT / "eval" / "judge"


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def sample(result_files: list[str], n: int = 25, seed: int = 7) -> str:
    items = []
    for f in result_files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for claim, ids in claim_sentences(r.get("note") or ""):
                passage = " ".join((hybrid.get_passage(i) or {}).get("text", "") for i in ids)
                if passage:
                    items.append({"case_id": r["case_id"], "claim": claim, "passage_ids": ids, "passage": passage[:2500]})
    random.Random(seed).shuffle(items)
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    out = JUDGE_DIR / "to_label.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items[:n]):
            fh.write(json.dumps({"id": f"j{i + 1:02d}", **it, "human": "", "human_2": ""}) + "\n")
    return f"wrote {min(n, len(items))} claims to {out}; label them into {JUDGE_DIR / 'labels.jsonl'}"


def agreement(path: str) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("human") in ("supported", "not_supported")]
    if len(rows) < 20:
        return {"error": f"only {len(rows)} human-labelled claims in {path}; the brief requires 20+"}
    cfg = get_config()
    if is_offline(cfg):
        return {"error": "judge agreement needs the live judge model (set OPENROUTER_API_KEY, llm.mode=live)"}
    ctx = RunContext(run_id="judge-calibration", cfg=cfg)
    lookup = {r["id"]: r for r in rows}
    claims = [(r["claim"], [r["id"]]) for r in rows]

    async def go():
        out = []
        for c in claims:  # one at a time; judge_claims caps how many it checks per call
            out += await judge_claims([c], lambda pid: {"text": lookup[pid]["passage"]}, ctx)
        return out
    verdicts = asyncio.run(go())
    judge = [v["verdict"] for v in verdicts]
    human = [r["human"] for r in rows]
    result = {"n": len(rows), "judge_model": cfg["llm"]["models"]["judge"],
              "raw_agreement": round(sum(j == h for j, h in zip(judge, human)) / len(rows), 3),
              "cohen_kappa_judge_vs_human": round(cohen_kappa(judge, human), 3), "cost_usd": round(ctx.cost_usd, 5)}
    second = [r.get("human_2") for r in rows]
    if all(s in ("supported", "not_supported") for s in second):
        result["cohen_kappa_human_vs_human"] = round(cohen_kappa(human, second), 3)
    out = JUDGE_DIR / "agreement.json"
    out.write_text(json.dumps({**result, "items": [{**r, "judge": j} for r, j in zip(rows, judge)]}, indent=2), encoding="utf-8")
    return result

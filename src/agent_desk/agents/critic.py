"""Compliance critic: code for numbers, a judge for claims, rules for wording (signature A + B).

  1. numbers   every number has an [F#] reference and equals that fact (control/numbers.py);
               every derived fact is recomputed in the sandbox from its input facts
  2. claims    each sentence citing a passage is checked against that passage by a judge model
               from a different family (google/gemini-...); offline: lexical-overlap fallback
  3. wording   no investment advice, no instruction-like text
Returns a report; the graph sends the draft back to the writer at most cfg.budgets.critic_max_rounds times.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import is_offline
from ..control.context import RunContext
from ..control.guardrails import advice_violations, is_injection
from ..control.numbers import CITATION_RE, body_of, check_numbers, quoted_citations
from ..finance.metrics import recompute
from ..llm.models import chat, is_fatal
from ..rag.hybrid import tokenize

MAX_CLAIMS = 8


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("“", "'").replace("”", "'").replace('"', "'")).strip().lower()


def quote_issues(note: str, lookup) -> list[str]:
    """Quoted filing text must appear verbatim in the passage it cites (pure code)."""
    issues = []
    for quote, pid in quoted_citations(note):
        passage = (lookup(pid) or {}).get("text", "")
        probe = _norm(quote)[:160]
        if not passage:
            issues.append(f"QUOTE cites {pid}, which does not exist")
        elif probe not in _norm(passage):
            issues.append(f"QUOTE not found verbatim in {pid}: \"{quote[:100]}\"")
    return issues


def claim_sentences(note: str) -> list[tuple[str, list[str]]]:
    body = body_of(note)
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", body):
        ids = CITATION_RE.findall(sentence)
        if ids:
            out.append((CITATION_RE.sub("", sentence).strip(" -*\""), ids))
    return out


def lexical_support(claim: str, passage: str) -> bool:
    c = set(tokenize(claim))
    if not c:
        return True
    return len(c & set(tokenize(passage))) / len(c) >= 0.35


async def judge_claims(claims: list[tuple[str, list[str]]], lookup, ctx: RunContext) -> list[dict[str, Any]]:
    results = []
    todo = claims[:MAX_CLAIMS]
    if is_offline(ctx.cfg) or not ctx.cfg["critic"].get("use_judge_for_claims", True):
        for claim, ids in todo:
            text = " ".join((lookup(i) or {}).get("text", "") for i in ids)
            results.append({"claim": claim, "ids": ids, "verdict": "supported" if lexical_support(claim, text) else "not_supported",
                            "by": "lexical"})
        return results
    ctx.for_agent("critic")
    judge = chat("judge", ctx, max_tokens=150)
    for claim, ids in todo:
        text = " ".join((lookup(i) or {}).get("text", "")[:1500] for i in ids)
        prompt = ctx.cfg["prompts"]["judge"].format(claim=claim, passage=text)
        try:
            raw = (await judge.ainvoke([("human", prompt)])).content
            m = re.search(r"\{.*\}", raw, re.S)
            verdict = json.loads(m.group(0)) if m else {"verdict": "not_supported", "reason": "unparseable judge reply"}
        except Exception as exc:
            if is_fatal(exc):
                raise
            verdict = {"verdict": "supported" if lexical_support(claim, text) else "not_supported",
                       "reason": f"judge error, lexical fallback: {type(exc).__name__}"}
        results.append({"claim": claim, "ids": ids, "verdict": verdict.get("verdict", "not_supported"),
                        "reason": verdict.get("reason", ""), "by": "judge"})
    return results


async def critique(note: str, refs: dict[str, dict], registry: dict[str, dict], lookup, ctx: RunContext,
                   check_claims: bool = True) -> dict[str, Any]:
    cfg = ctx.cfg
    issues: list[str] = []
    numbers = check_numbers(note, refs, cfg["critic"]["number_tolerance_rel"])
    for c in numbers.issues:
        issues.append(f"NUMBER {c.text}: {c.status}. {c.detail}")
    for ref, fact in refs.items():
        if fact.get("kind") == "derived":
            value = recompute(fact, registry)
            if value is None or abs(value - fact["value"]) > 1e-6 * max(1.0, abs(fact["value"])):
                issues.append(f"DERIVED {ref} ({fact['fact_id']}) does not recompute from its inputs")
    issues += quote_issues(note, lookup)
    quoted = {q for q, _ in quoted_citations(note)}
    # verbatim quotes are verified by code above; the judge only sees paraphrased claims
    paraphrased = [(c, ids) for c, ids in claim_sentences(note) if not any(_norm(q)[:60] in _norm(c) for q in quoted)]
    claims = await judge_claims(paraphrased, lookup, ctx) if check_claims else []
    for c in claims:
        if c["verdict"] != "supported":
            issues.append(f"CLAIM not supported by {c['ids']}: \"{c['claim'][:140]}\" {c.get('reason', '')}")
    for phrase in advice_violations(note, cfg):
        issues.append(f"WORDING investment-advice language: '{phrase}'")
    bad, signals = is_injection(note.split("## Sources")[0], cfg)
    if bad:
        issues.append(f"WORDING instruction-like text copied from data: {signals[:2]}")
    return {"passed": not issues, "issues": issues, "numbers_counted": numbers.counted, "numbers_verified": numbers.verified,
            "verified_share": round(numbers.share, 4), "claims_checked": len(claims),
            "claims_supported": sum(c["verdict"] == "supported" for c in claims), "claims": claims}

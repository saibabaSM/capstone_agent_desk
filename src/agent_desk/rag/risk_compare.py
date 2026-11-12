"""Compare one company's Item 1A risk factors between two fiscal years.

This is "option A" of the risk-diff agent: an LLM proposes what was added, removed or changed
by comparing the two years' risk lists, and code then verifies every claim against the filings'
actual passages before it is kept.

Flow, for Microsoft FY2024 -> FY2025:

    compare_company_years("microsoft", 2024, 2025)
        load_risk_section(...) x2        passages of each year            rag/risk_section.py
        extract_risks(...)       x2      risk list of each year           rag/risk_extract.py
        compare_risks(...)
            1. _propose:  one LLM call over both risk lists -> added / removed / changed claims
            2. embed every passage; S[i][j] = similarity of new passage i to old passage j
            3. verify added / removed claims with S (a claim whose text still exists in the other
               year is rejected as "regrouped or reworded, not new")
            4. verify changed claims: pick the most different old/new passage pairs and let one
               batched LLM call decide whether the wording changed materially
        -> RiskDiff(added, removed, changed, rejected)

Why the verification exists: the same risk is often titled or grouped differently in two runs
(Microsoft FY2024 vs FY2025 matched only 14 of 19 and 18 titles), so a comparison of the lists
alone reports renaming as change. Checking the underlying passage text removes those false claims.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import numpy as np

from ..llm.llm import get_tool_llm
from .filing_rag import _get_embeddings
from .risk_extract import Risk, _quote, extract_risks
from .risk_section import RiskPassage, load_risk_section

# Cosine similarity at or above this means two passages say the same thing (reworded at most).
SAME_SIM = 0.85
# A new passage whose best old match is below SAME_SIM is "different"; this many pairs of the most
# different ones are shown to the LLM when it judges a "changed" claim.
MAX_PAIRS_PER_CLAIM = 3
# A claim is rejected as "not new" when at least this share of its cited passages has a
# near-identical passage in the other year. A genuinely new (or removed) risk has none, and a risk
# that exists in both years but was merely renamed or regrouped has all of them or half of them
# (for example 1 of 2 passages), so ">=" matters: ">" let exactly-half claims through.
REJECT_SHARE = 0.5
# Passage text is cut to this many characters in the verification prompt.
PASSAGE_CHARS = 1000


# What the proposal LLM returns for each claim. `risks` holds labels such as "N3" (a risk of the
# newer year) or "O5" (a risk of the older year), never passage ids, so it cannot invent ids.
class RawClaim(TypedDict):
    description: str
    risks: list[str]


class RawComparison(TypedDict):
    added: list[RawClaim]
    removed: list[RawClaim]
    changed: list[RawClaim]


# One verified claim in the result.
# `evidence` is only filled for "changed" claims: the old/new passage pairs that were compared,
# e.g. {"old_id": "microsoft-2024-1A-30", "new_id": "microsoft-2025-1A-31", "similarity": 0.78}.
class Finding(TypedDict):
    description: str
    passage_ids: list[str]
    quote: str
    evidence: list[dict[str, Any]]


# The final result. `rejected` lists the claims verification threw out and why, so a reader can
# see how much of the model's proposal survived, e.g.
#   {"kind": "removed", "description": "Talent risk", "reason": "100% of the cited passages ..."}
class RiskDiff(TypedDict):
    company: str
    year_from: int
    year_to: int
    added: list[Finding]
    removed: list[Finding]
    changed: list[Finding]
    rejected: list[dict[str, str]]


# What the verification LLM returns for each "changed" claim.
class Verdict(TypedDict):
    claim: int
    material: bool
    change: str


class VerdictList(TypedDict):
    verdicts: list[Verdict]


PROPOSE_PROMPT = """You compare a company's risk factors between two fiscal years. Each year's risks
are listed as [label] title: summary. Labels starting with O are the older year, labels starting
with N are the newer year.

The same risk can be titled or grouped differently in each year, so match risks by meaning, not by
title. Return:
- added: risks in the newer year with no counterpart in the older year (cite N labels).
- removed: risks in the older year with no counterpart in the newer year (cite O labels).
- changed: risks present in both years whose newer description is materially different, such as a
  new topic, a wider scope, or a new regulation or figure named (cite at least one O and one N label).

Do not list risks that only have different titles or wording. Use only labels from the input. Give
each claim a one-sentence description. Return JSON only."""

VERIFY_PROMPT = """You check claims that a company's risk factor changed between two fiscal years.
For each numbered claim you see pairs of OLD and NEW passage text. Decide whether the NEW wording is
materially different from the OLD (new facts, wider or narrower scope, risk elements added or
removed) and not just rephrased. Use only the text shown. Return one verdict per claim: material
true or false and, if true, one sentence saying what changed. Return JSON only."""


# Unit-length rows, so a matrix product of two results is a table of cosine similarities.
def _normalize(vectors: Any) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)


def _default_embed(texts: list[str]) -> Any:
    return _get_embeddings().embed_documents(texts)


# Gives each risk a label so the LLM can cite it: {"O1": risk, "O2": risk, ...}.
def _label(risks: list[Risk], prefix: str) -> dict[str, Risk]:
    return {f"{prefix}{i}": risk for i, risk in enumerate(risks, start=1)}


# "[O1] Competition risk: The company faces intense competition..." one line per risk.
def _format_risks(labelled: dict[str, Risk]) -> str:
    return "\n".join(f"[{label}] {r['title']}: {r['summary']}" for label, r in labelled.items())


# One structured-output LLM call. Same technique as risk_extract._call_llm: the reply must match
# `schema`, and a reply that cannot be parsed raises ValueError instead of failing silently.
def _ask(schema: Any, system: str, user: str, llm: Any) -> dict[str, Any]:
    structured = llm.with_structured_output(
        schema, method="json_schema", strict=True, include_raw=True
    )
    out = structured.invoke([("system", system), ("user", user)])
    parsed = out.get("parsed") or {}
    if not parsed:
        raise ValueError(f"Risk comparison returned no usable output: {out.get('parsing_error')}")
    return parsed


# Passage ids of the risks cited by `labels`, in order and without duplicates.
# Unknown labels are ignored. Example: _ids(["N1", "N3"], new_labelled) -> all ids of those risks.
def _ids(labels: list[str], labelled: dict[str, Risk]) -> list[str]:
    seen: dict[str, None] = {}
    for label in labels:
        for passage_id in labelled.get(label, {}).get("passage_ids", []):
            seen.setdefault(passage_id)
    return list(seen)


# Compares two years of one company, given the passages and risk lists already loaded.
# `embed` maps a list of texts to a matrix of embeddings (tests pass a fake). Raises ValueError if
# the proposal or verification reply is unusable. See the module docstring for the flow.
def compare_risks(
    company: str,
    year_from: int,
    year_to: int,
    old_passages: list[RiskPassage],
    new_passages: list[RiskPassage],
    old_risks: list[Risk],
    new_risks: list[Risk],
    llm: Any | None = None,
    embed: Callable[[list[str]], Any] | None = None,
) -> RiskDiff:
    llm = llm or get_tool_llm()
    embed = embed or _default_embed
    old_labelled, new_labelled = _label(old_risks, "O"), _label(new_risks, "N")
    old_idx = {p.id: i for i, p in enumerate(old_passages)}
    new_idx = {p.id: i for i, p in enumerate(new_passages)}
    old_text = {p.id: p.text for p in old_passages}
    new_text = {p.id: p.text for p in new_passages}

    # Step 1: the LLM proposes claims from the two risk lists.
    user = (
        f"Older year (FY{year_from}) risks:\n{_format_risks(old_labelled)}\n\n"
        f"Newer year (FY{year_to}) risks:\n{_format_risks(new_labelled)}"
    )
    proposal = _ask(RawComparison, PROPOSE_PROMPT, user, llm)

    # Step 2: sim[i][j] = similarity of new passage i to old passage j.
    old_vecs = _normalize(embed([p.text for p in old_passages]))
    new_vecs = _normalize(embed([p.text for p in new_passages]))
    sim = new_vecs @ old_vecs.T

    rejected: list[dict[str, str]] = []
    result: dict[str, list[Finding]] = {"added": [], "removed": [], "changed": []}

    def reject(kind: str, description: str, reason: str) -> None:
        rejected.append({"kind": kind, "description": description, "reason": reason})

    # Step 3: added / removed claims. The cited passages must not already exist, near-identical,
    # in the other year. "added" claims look at new passages against all old ones; "removed"
    # claims look at old passages against all new ones.
    for kind, labelled, other_year, text_of, own_idx in (
        ("added", new_labelled, year_from, new_text, new_idx),
        ("removed", old_labelled, year_to, old_text, old_idx),
    ):
        for claim in proposal.get(kind, []):
            wanted = "N" if kind == "added" else "O"
            labels = [x for x in claim.get("risks", []) if x.startswith(wanted)]
            ids = _ids(labels, labelled)
            description = str(claim.get("description", "")).strip()
            if not ids or not description:
                reject(kind, description, "no valid risk labels cited")
                continue
            rows = [own_idx[i] for i in ids]
            best = sim[rows].max(axis=1) if kind == "added" else sim[:, rows].max(axis=0)
            share = float((best >= SAME_SIM).mean())
            if share >= REJECT_SHARE:
                reject(
                    kind,
                    description,
                    f"{share:.0%} of the cited passages have a near-identical passage in "
                    f"FY{other_year} (regrouped or reworded, not {kind})",
                )
                continue
            result[kind].append(
                Finding(description=description, passage_ids=ids, quote=_quote(text_of[ids[0]]), evidence=[])
            )

    # Step 4a: changed claims. For each new passage of the cited risks, find its closest old
    # passage among the cited old risks; keep the most different pairs as evidence.
    candidates: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    for claim in proposal.get("changed", []):
        description = str(claim.get("description", "")).strip()
        labels = claim.get("risks", [])
        old_ids = _ids([x for x in labels if x.startswith("O")], old_labelled)
        new_ids = _ids([x for x in labels if x.startswith("N")], new_labelled)
        if not old_ids or not new_ids or not description:
            reject("changed", description, "must cite at least one older and one newer risk")
            continue
        pairs = []
        for new_id in new_ids:
            scores = sim[new_idx[new_id], [old_idx[i] for i in old_ids]]
            best = int(scores.argmax())
            pairs.append(
                {"old_id": old_ids[best], "new_id": new_id, "similarity": round(float(scores[best]), 3)}
            )
        different = sorted((p for p in pairs if p["similarity"] < SAME_SIM), key=lambda p: p["similarity"])
        if not different:
            reject("changed", description, "every cited passage is near-identical to its older counterpart")
            continue
        candidates.append((description, old_ids + new_ids, different[:MAX_PAIRS_PER_CLAIM]))

    # Step 4b: one batched LLM call judges all remaining "changed" claims on the actual text.
    if candidates:
        blocks = []
        for n, (description, _, evidence) in enumerate(candidates, start=1):
            lines = [f"Claim {n}: {description}"]
            for pair in evidence:
                lines.append(f"  OLD [{pair['old_id']}]: {old_text[pair['old_id']][:PASSAGE_CHARS]}")
                lines.append(f"  NEW [{pair['new_id']}]: {new_text[pair['new_id']][:PASSAGE_CHARS]}")
            blocks.append("\n".join(lines))
        verdicts = {
            v.get("claim"): v
            for v in _ask(VerdictList, VERIFY_PROMPT, "\n\n".join(blocks), llm).get("verdicts", [])
        }
        for n, (description, ids, evidence) in enumerate(candidates, start=1):
            verdict = verdicts.get(n)
            if not verdict or not verdict.get("material"):
                reject("changed", description, "the cited passages are not materially different")
                continue
            result["changed"].append(
                Finding(
                    description=str(verdict.get("change") or description).strip(),
                    passage_ids=ids,
                    quote=_quote(new_text[evidence[0]["new_id"]]),
                    evidence=evidence,
                )
            )

    return RiskDiff(
        company=company,
        year_from=year_from,
        year_to=year_to,
        added=result["added"],
        removed=result["removed"],
        changed=result["changed"],
        rejected=rejected,
    )


# Loads, extracts and compares two years of one company in one call. This is what the risk-diff
# agent's tool will call. The years may be given in either order; the older one is "from".
# Raises FileNotFoundError (from load_risk_section) if either year has no filing, e.g.
# "No alphabet 10-K for fiscal year 2026. Available years: 2023, 2024, 2025."
# Example: compare_company_years("microsoft", 2025, 2024) compares FY2024 -> FY2025.
def compare_company_years(
    company: str, year_a: int | str, year_b: int | str, llm: Any | None = None
) -> RiskDiff:
    year_from, year_to = sorted((int(year_a), int(year_b)))
    if year_from == year_to:
        raise ValueError("Pick two different fiscal years to compare.")
    old_passages = load_risk_section(company, year_from)
    new_passages = load_risk_section(company, year_to)
    return compare_risks(
        company.strip().lower(),
        year_from,
        year_to,
        old_passages,
        new_passages,
        extract_risks(old_passages, llm),
        extract_risks(new_passages, llm),
        llm,
    )

"""Extract a structured list of distinct risks from one year's Item 1A passages with an LLM.

Overall flow (this is "option B", step 2 of the risk-diff agent):

    load_risk_section("microsoft", 2025)          rag/risk_section.py
        -> [RiskPassage("microsoft-2025-1A-1", "..."), ... "-1A-89"]
    extract_risks(passages)                       this file
        pass 1: the LLM groups all passages into risks         (_call_llm + _merge)
        pass 2: the LLM places passages pass 1 left unassigned (_call_llm + _merge)
        code:   sort ids, derive a quote, order the risks      (_quote)
        -> [Risk(title, summary, passage_ids, quote), ...]

The LLM only decides how passages group into risks and what to call them. Everything that has to
be trustworthy (valid ids, one risk per passage, quotes, ordering) is done in plain code, so the
model cannot invent citations. A later step compares two years' risk lists (added / removed /
changed).
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..llm.llm import get_tool_llm
from .risk_section import RiskPassage, load_risk_section

# How many words of the first cited passage become Risk["quote"].
QUOTE_WORDS = 20


# What the LLM is asked to return for each risk. It does not return a quote: models paraphrase
# instead of copying exactly, so the quote is taken from the filing text in code (see _quote).
class RawRisk(TypedDict):
    title: str
    summary: str
    passage_ids: list[str]


# The JSON schema handed to with_structured_output; the model must reply {"risks": [RawRisk, ...]}.
class RawRiskList(TypedDict):
    risks: list[RawRisk]


# The final, validated result returned to callers.
# Example:
#   Risk(title="Cyberattacks on our services",
#        summary="Attacks could disrupt services and harm customers.",
#        passage_ids=["microsoft-2025-1A-36", "microsoft-2025-1A-37"],
#        quote="Cyberattacks could adversely impact our customers even if our production services")
class Risk(TypedDict):
    title: str
    summary: str
    passage_ids: list[str]
    quote: str  # first words of the first cited passage, so a reader can find it in the filing


# Prompt for pass 1. The granularity rule ("25 to 40 risks") is there because the first version of
# this prompt produced only 16 broad groups for Microsoft FY2025, which hides year-over-year
# changes inside big categories like "Cybersecurity Risks".
SYSTEM_PROMPT = """You read the Item 1A (Risk Factors) section of a 10-K. Each passage is shown as
[passage-id] text. Group the passages into the distinct risks the company describes.

For each risk return:
- title: a short name for the specific risk (under 12 words).
- summary: one sentence saying what the risk is, using only what the passages say.
- passage_ids: the ids of the passages that describe this risk. Use only ids that appear in the
  input, exactly as written.

Rules:
- Be specific: make one risk per distinct topic or heading, not one per broad category. A full
  Item 1A normally yields 25 to 40 risks (for example separate risks for cyberattacks, data
  privacy, platform abuse, product defects, supply chain, tax, litigation, and so on).
- Cover the whole section: every passage should belong to some risk.
- A passage belongs to exactly one risk; one risk can span several passages.
- Do not invent risks or ids. Do not compare with other years.
- Return JSON only."""

# Appended to SYSTEM_PROMPT for pass 2, with {titles} filled in from the risks found in pass 1.
# It tells the model to reuse an existing title when a leftover passage belongs to that risk
# (so _merge extends it instead of creating a near-duplicate), or to create a new risk otherwise.
GAP_PROMPT_SUFFIX = """

These risks were already extracted from earlier passages:
{titles}

The passages below were left unassigned. If a passage belongs to one of the existing risks, return
that risk with the exact same title and only the new passage ids. Otherwise return a new risk."""


# Turns passages into the text the LLM reads, one line per passage with its id in brackets so the
# model can cite it.
# Example output:
#   [microsoft-2025-1A-36] Cyberattacks could adversely impact our customers ...
#   [microsoft-2025-1A-37] We may not be able to protect information in our systems ...
def _format_passages(passages: list[RiskPassage]) -> str:
    return "\n".join(f"[{p.id}] {p.text}" for p in passages)


# Returns the first QUOTE_WORDS words of a passage. Because it is cut from the real filing text
# (not written by the model) it is always a verbatim phrase a reader can Ctrl+F in the HTML.
# Example: _quote("Cyberattacks could adversely impact our customers even if our production
#   services are not directly compromised. ...") -> "Cyberattacks could adversely impact our
#   customers even if our production services are not directly compromised. ..." cut to 20 words.
def _quote(text: str) -> str:
    words = text.split()
    return " ".join(words[:QUOTE_WORDS])


# One LLM call. Sends `system` plus the formatted passages and returns the raw risk dicts the
# model produced (not yet checked: ids may be invented, passages may be reused).
#   - with_structured_output(..., method="json_schema", strict=True) forces the reply to match
#     RawRiskList, the same technique the supervisor uses for its Route.
#   - include_raw=True makes the result a dict {"raw": AIMessage, "parsed": RawRiskList | None,
#     "parsing_error": error | None}, so a malformed reply can be reported instead of crashing
#     inside the library.
# Raises ValueError if the model returned nothing usable (parsed is None or empty).
# Example return: [{"title": "Cyberattacks", "summary": "...", "passage_ids": ["co-2025-1A-1"]}]
def _call_llm(system: str, passages: list[RiskPassage], llm: Any) -> list[dict[str, Any]]:
    structured = llm.with_structured_output(
        RawRiskList, method="json_schema", strict=True, include_raw=True
    )
    out = structured.invoke([("system", system), ("user", _format_passages(passages))])
    parsed = out.get("parsed") or {}
    if not parsed:
        raise ValueError(f"Risk extraction returned no usable output: {out.get('parsing_error')}")
    return parsed.get("risks", [])


# Folds the model's raw risks into `risks`, the running result, and is where the checks happen.
# `risks` maps a lower-cased title to {"title", "summary", "passage_ids"}; it is modified in place.
# For each raw risk it:
#   1. drops ids that are not real passage ids (invented citations),
#   2. drops ids already assigned to an earlier risk (a passage belongs to exactly one risk),
#   3. skips the risk if the title is empty or no valid ids remain,
#   4. reuses an existing entry when the lower-cased title matches (so "Cyber" and "cyber" from
#      pass 2 extend the pass-1 risk instead of creating a duplicate), otherwise creates one.
# Example: with passages 1-6 and an empty `risks`, raw
#   [{"title": "Tax", "passage_ids": ["co-1A-3", "co-1A-1", "co-1A-99"]}]  ("-99" does not exist)
# gives risks == {"tax": {"title": "Tax", "summary": "...", "passage_ids": ["co-1A-3", "co-1A-1"]}}
def _merge(
    risks: dict[str, dict[str, Any]], raw_risks: list[dict[str, Any]], valid_ids: set[str]
) -> None:
    """Add raw risks into `risks` (keyed by lower-cased title), keeping each passage in one risk."""
    assigned = {i for r in risks.values() for i in r["passage_ids"]}
    for raw in raw_risks:
        title = str(raw.get("title", "")).strip()
        ids = [
            i for i in raw.get("passage_ids", []) if i in valid_ids and i not in assigned
        ]
        if not title or not ids:
            continue
        assigned.update(ids)
        entry = risks.setdefault(
            title.lower(),
            {"title": title, "summary": str(raw.get("summary", "")).strip(), "passage_ids": []},
        )
        entry["passage_ids"].extend(ids)


# Main entry point: turns one year's Item 1A passages into a validated list of Risk entries.
# Flow, with a Microsoft FY2025 section of 89 passages as the example:
#   1. order / text_by_id / valid_ids: lookup tables built from the passages. `order` maps each id
#      to its position in the filing and is used to sort ids and risks into document order.
#   2. Pass 1: send all 89 passages to the LLM and merge the answer into `risks`. In practice the
#      model leaves some passages unassigned (the first run covered 73 of 89).
#   3. Pass 2: if any passages are unassigned, send only those, with the existing titles in the
#      prompt (GAP_PROMPT_SUFFIX), and merge again. Skipped when pass 1 produced no risks at all,
#      because there are no titles to reuse and a second full pass would just repeat pass 1.
#   4. Build the result: sort each risk's ids into filing order, derive the quote from the first
#      cited passage, and sort the risks by where they first appear in the filing.
# Passages that the model still leaves out after pass 2 are simply not in any risk.
# `llm` is optional so tests can pass a fake model; by default get_tool_llm() is used.
# Example result:
#   [Risk(title="Competition", passage_ids=["microsoft-2025-1A-2", "microsoft-2025-1A-3"], ...),
#    Risk(title="Cyberattacks", passage_ids=["microsoft-2025-1A-36", ...], ...), ...]
def extract_risks(passages: list[RiskPassage], llm: Any | None = None) -> list[Risk]:
    """Group passages into risks, fill in unassigned passages, and attach code-derived quotes."""
    llm = llm or get_tool_llm()
    order = {p.id: n for n, p in enumerate(passages)}
    text_by_id = {p.id: p.text for p in passages}
    valid_ids = set(order)

    risks: dict[str, dict[str, Any]] = {}
    _merge(risks, _call_llm(SYSTEM_PROMPT, passages, llm), valid_ids)

    assigned = {i for r in risks.values() for i in r["passage_ids"]}
    missing = [p for p in passages if p.id not in assigned]
    if missing and risks:
        titles = "\n".join(f"- {r['title']}" for r in risks.values())
        system = SYSTEM_PROMPT + GAP_PROMPT_SUFFIX.format(titles=titles)
        _merge(risks, _call_llm(system, missing, llm), valid_ids)

    result: list[Risk] = []
    for entry in risks.values():
        ids = sorted(entry["passage_ids"], key=order.__getitem__)
        result.append(
            Risk(
                title=entry["title"],
                summary=entry["summary"],
                passage_ids=ids,
                quote=_quote(text_by_id[ids[0]]),
            )
        )
    return sorted(result, key=lambda r: order[r["passage_ids"][0]])


# Convenience wrapper used by the risk-diff tools: load the filing's Item 1A section from
# data/filings/<company>/<year>/ and extract its risks in one call.
# Raises FileNotFoundError (from load_risk_section) if that company or year has no filing, e.g.
# "No apple 10-K for fiscal year 2024. Available years: 2025." The caller can pass that message
# straight back to the supervisor instead of guessing.
# Example: extract_company_year_risks("microsoft", 2025) -> list of Risk for Microsoft's FY2025 10-K.
def extract_company_year_risks(company: str, year: int | str, llm: Any | None = None) -> list[Risk]:
    """Load the filing's Item 1A section and extract its risk list."""
    return extract_risks(load_risk_section(company, year), llm)

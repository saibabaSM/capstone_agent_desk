"""Find every number in a note and check it against the fact it cites. Pure code, no model.

A note cites facts with short references, e.g. "operating margin of 45.6% [F3]". `refs` maps
"F3" -> the fact record (value, unit, fact_id). For each number in the note:

    no reference nearby                  -> issue "unattributed"
    reference to an unknown F#           -> issue "unknown_ref"
    value differs from the fact (scaled) -> issue "mismatch"
    otherwise                            -> verified

Exempt (not counted): fiscal years and calendar years (FY2025, 2024), ISO dates, Item numbers
("Item 1A"), F# and passage-id references themselves, and bare integers <= 12 with no unit
("three years", "2 rounds").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

REF_RE = re.compile(r"\[(F\d+)\]")
CITATION_RE = re.compile(r"\[([a-z0-9_]+-\d{4}-[0-9A-Za-z]{1,6}-\d{4}|web-[a-z0-9_]+-\d{2})\]")
BRACKETS_RE = re.compile(r"\[[^\]]*\]")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ITEM_RE = re.compile(r"\bItem\s+\d{1,2}[A-C]?\b", re.IGNORECASE)
NUM_RE = re.compile(
    r"(?<![\w/.\-])(?P<neg>-|−)?(?P<cur>\$)?\s?(?P<paren>\()?(?P<num>\d[\d,]*(?:\.\d+)?)\)?"
    r"\s?(?P<unit>%|percent\b|pp\b|bn\b|B\b|billion\b|M\b|mn\b|million\b|K\b|thousand\b)?")
SCALE = {"bn": 1e9, "b": 1e9, "billion": 1e9, "m": 1e6, "mn": 1e6, "million": 1e6, "k": 1e3, "thousand": 1e3}
REF_WINDOW = 48  # characters after a number in which its [F#] reference must appear


@dataclass
class NumberCheck:
    text: str
    value: float
    unit: str
    ref: str | None
    status: str  # verified | unattributed | unknown_ref | mismatch
    detail: str = ""


@dataclass
class NumberReport:
    checks: list[NumberCheck] = field(default_factory=list)

    @property
    def counted(self) -> int:
        return len(self.checks)

    @property
    def verified(self) -> int:
        return sum(c.status == "verified" for c in self.checks)

    @property
    def share(self) -> float:
        return 1.0 if not self.checks else self.verified / self.counted

    @property
    def issues(self) -> list[NumberCheck]:
        return [c for c in self.checks if c.status != "verified"]


CODE_RE = re.compile(r"`[^`\n]*`")
MONTH_DAY_RE = re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
                          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:,\s*\d{4})?\b")
# A verbatim quote from a filing followed by its passage citation: its numbers belong to the
# quoted text, not to XBRL. The critic verifies the quote really appears in that passage.
QUOTED_CITED_RE = re.compile(r"[\"“]([^\"”\n]{8,700})[\"”]\s*\[([a-z0-9_]+-\d{4}-[0-9A-Za-z]{1,6}-\d{4}|web-[a-z0-9_]+-\d{2})\]")


def body_of(note: str) -> str:
    """The part of a note a model wrote or a reader checks: everything before the Sources appendix."""
    return re.split(r"\n#+\s*Sources\b", note, maxsplit=1)[0]


def quoted_citations(note: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in QUOTED_CITED_RE.finditer(body_of(note))]


def _mask(text: str) -> str:
    """Blank out spans whose digits are not claims, keeping offsets identical."""
    for rx in (QUOTED_CITED_RE, CODE_RE, BRACKETS_RE, DATE_RE, MONTH_DAY_RE, ITEM_RE):
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _parse(m: re.Match) -> tuple[float, str]:
    value = float(m.group("num").replace(",", ""))
    if m.group("neg") or (m.group("paren") and m.group(0).rstrip().endswith(")")):
        value = -value
    unit = (m.group("unit") or "").lower()
    if unit in ("%", "percent"):
        return value, "%"
    if unit == "pp":
        return value, "pp"
    if unit in SCALE:
        return value * SCALE[unit], "USD" if m.group("cur") else "count"
    if m.group("cur"):
        return value, "USD"
    return value, ""


def _matches(value: float, unit: str, fact: dict[str, Any], tol: float) -> bool:
    target = float(fact["value"])
    if unit == "%" and fact["unit"] != "%":
        return False
    if unit == "USD" and fact["unit"] not in ("USD", "USD/shares"):
        return False
    if unit == "" and fact["unit"] in ("USD",):
        # a bare number cited against a USD fact: accept millions-style tables (e.g. 281,724)
        return any(abs(value * s - target) <= max(abs(target) * tol, 0.51 * s) for s in (1, 1e3, 1e6))
    if target == 0:
        return abs(value) < 1e-9
    rel_ok = abs(value - target) <= abs(target) * tol
    # the display rounds to 1 decimal of the scaled unit; allow half a unit of that rounding
    scale = 1e9 if abs(target) >= 1e9 else 1e6 if abs(target) >= 1e6 else 1
    round_ok = abs(value - target) <= 0.051 * (scale if unit == "USD" and scale > 1 else 1)
    return rel_ok or round_ok


def check_numbers(note: str, refs: dict[str, dict[str, Any]], tol: float = 0.006) -> NumberReport:
    report = NumberReport()
    note = body_of(note)
    masked = _mask(note)
    for m in NUM_RE.finditer(masked):
        raw = m.group(0).strip()
        if not any(ch.isdigit() for ch in raw):
            continue
        value, unit = _parse(m)
        whole = m.group("num").replace(",", "")
        if unit == "" and "." not in whole:
            iv = int(whole)
            if iv <= 12 or 1990 <= iv <= 2100:
                continue  # counts and years
        window = note[m.end(): m.end() + REF_WINDOW]
        sentence_end = re.search(r"(?<!\d)[.;\n](?!\d)", window)
        window = window[: sentence_end.start()] if sentence_end else window
        ref_match = REF_RE.search(window)
        if not ref_match:
            report.checks.append(NumberCheck(raw, value, unit, None, "unattributed", "no [F#] reference after this number"))
            continue
        ref = ref_match.group(1)
        fact = refs.get(ref)
        if fact is None:
            report.checks.append(NumberCheck(raw, value, unit, ref, "unknown_ref", f"{ref} is not in the fact list"))
            continue
        if _matches(value, unit, fact, tol):
            report.checks.append(NumberCheck(raw, value, unit, ref, "verified"))
        else:
            report.checks.append(NumberCheck(raw, value, unit, ref, "mismatch",
                                             f"{raw} vs fact {fact.get('display', fact['value'])} ({fact.get('fact_id', '')})"))
    return report


def cited_passages(note: str) -> list[str]:
    return sorted(set(CITATION_RE.findall(note)))

"""Input and output guardrails (L15). Several layers, each cheap and testable:

Input  (before any model sees the request)
  1. injection heuristics on the analyst's text (and on every retrieved passage, via the post-tool hook)
  2. optional Llama Prompt Guard 2 classifier (guardrails.injection_classifier: prompt_guard)
  3. scope: the request must be research on covered companies; personal investment advice and
     off-topic requests are refused; uncovered companies are named back to the analyst
  4. optional Llama Guard 4 moderation via OpenRouter (guardrails.llama_guard)
Output (before the note reaches the approval gate)
  1. every number carries a fact reference that verifies (control/numbers.py)
  2. no advice language (buy/sell/price targets/promised returns)
  3. every cited passage exists in the index
  4. no private-watchlist names that are not part of this request
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..companies import COMPANIES, resolve
from .numbers import check_numbers, cited_passages

INJECTION_PATTERNS = [
    r"ignore (?:all |any |your |the )?(?:previous|prior|above|earlier)? ?(?:instructions|rules|guidelines)",
    r"disregard (?:all |your |the )?(?:previous|prior|above)? ?(?:instructions|rules)",
    r"\bnote to (?:the )?ai\b", r"\b(?:ai|research) (?:assistant|agent)s?\b.{0,40}\bmust\b",
    r"^\s*system\s*:", r"\bsystem prompt\b", r"\byou are now\b", r"\bnew instructions?\b",
    r"\bdo not (?:mention|reveal|disclose) (?:this|these) instructions?\b",
    r"\bcall (?:the )?(?:tool )?(?:publish_note|search_filings|run_python)\b", r"\bpublish_note\b",
    r"\bwithout (?:waiting for )?(?:approval|review)\b", r"\b(?:private |full )?watchlist\b",
    r"\bjailbreak\b", r"\bDAN mode\b", r"\boverride (?:the )?(?:compliance|safety)\b",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

ADVICE_REQUEST_RE = re.compile(
    r"\b(should i (?:buy|sell|invest|hold)|is it a good (?:time|idea) to (?:buy|sell|invest)|"
    r"which stock (?:should|to) (?:i )?buy|best stock to buy|will (?:the )?(?:stock|shares?|price) (?:go up|rise|fall|double)|"
    r"price target|give me a (?:buy|sell) (?:rating|recommendation)|guarantee)", re.IGNORECASE)
FINANCE_TERMS = re.compile(
    r"\b(revenue|margin|income|profit|earnings|eps|cash flow|capex|r&d|risk|10-k|filing|segment|growth|"
    r"compare|comparison|ratio|expense|cost|guidance|item 1a|item 7|md&a|fiscal|note|changed|competition|"
    r"regulat|supply|debt|liquidity|dividend|buyback|repurchase|operating|gross|net)\w*", re.IGNORECASE)
SECRET_RE = re.compile(r"(sk-or-v1-[A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{20,}|lsv2_[A-Za-z0-9_]{10,}|APPROVAL_SECRET\S*)")
TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")


@dataclass
class GuardResult:
    allowed: bool
    category: str = "ok"  # ok | injection | advice | off_topic | out_of_coverage | moderation
    reason: str = ""
    tickers: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def injection_signals(text: str) -> list[str]:
    return sorted({m.group(0).strip()[:60] for m in INJECTION_RE.finditer(text or "")})


_prompt_guard: Any = None


def prompt_guard_score(text: str) -> float | None:
    """Probability of injection from Llama Prompt Guard 2 (22M) on CPU, if installed and licensed."""
    global _prompt_guard
    try:
        if _prompt_guard is None:
            from transformers import pipeline
            _prompt_guard = pipeline("text-classification", model="meta-llama/Llama-Prompt-Guard-2-22M", device=-1)
        out = _prompt_guard(text[:2000])[0]
        return float(out["score"]) if out["label"].upper() in ("MALICIOUS", "LABEL_1", "INJECTION") else 1 - float(out["score"])
    except Exception:  # model not downloaded / gated: fall back to heuristics only
        return None


def is_injection(text: str, cfg: dict) -> tuple[bool, list[str]]:
    signals = injection_signals(text)
    if cfg["guardrails"].get("injection_classifier") == "prompt_guard":
        score = prompt_guard_score(text)
        if score is not None and score > 0.8:
            signals.append(f"prompt_guard:{score:.2f}")
    return bool(signals), signals


def mentioned_companies(text: str) -> list[str]:
    found: list[str] = []
    low = text.lower()
    for c in COMPANIES:
        if any(re.search(rf"\b{re.escape(a)}\b", low) for a in c.aliases) or re.search(rf"\b{c.ticker}\b", text):
            found.append(c.ticker)
    return found


def _uncovered_tickers(text: str) -> list[str]:
    common = {"AI", "US", "EU", "FY", "CEO", "CFO", "SEC", "MD", "EPS", "USD", "GAAP", "R", "OK", "PDF", "API", "TTM", "YOY", "IT", "UK"}
    return [t for t in TICKER_TOKEN_RE.findall(text) if t not in common and resolve(t) is None]


def input_guard(question: str, cfg: dict) -> GuardResult:
    if not cfg["guardrails"]["enabled"]:
        return GuardResult(True, tickers=mentioned_companies(question), flags=["guardrails_off"])
    tickers = mentioned_companies(question)
    bad, signals = is_injection(question, cfg)
    if bad:
        return GuardResult(False, "injection", "The request contains instructions aimed at the system "
                           f"({'; '.join(signals[:3])}). Please rephrase it as a research question.", tickers, signals)
    if ADVICE_REQUEST_RE.search(question):
        return GuardResult(False, "advice", "Analyst Desk prepares research notes from filings; it does not give "
                           "buy/sell recommendations, price targets or return forecasts.", tickers)
    if not tickers:
        uncovered = _uncovered_tickers(question)
        covered = ", ".join(c.ticker for c in COMPANIES)
        if uncovered:
            return GuardResult(False, "out_of_coverage", f"{', '.join(uncovered)} is not in the coverage list ({covered}).")
        if not FINANCE_TERMS.search(question):
            return GuardResult(False, "off_topic", "This desk answers research questions about covered companies' filings.")
        return GuardResult(False, "out_of_coverage", f"Please name at least one covered company ({covered}).")
    if not FINANCE_TERMS.search(question) and "what changed" not in question.lower():
        return GuardResult(False, "off_topic", "This desk answers research questions about covered companies' filings.", tickers)
    return GuardResult(True, tickers=tickers)


def advice_violations(text: str, cfg: dict) -> list[str]:
    low = text.lower()
    hits = [p for p in cfg["guardrails"]["advice_phrases"] if p in low]
    hits += [m.group(0) for m in re.finditer(r"\b(?:rate|rating|upgrade|downgrade)\w*\b[^.\n]{0,30}\b(?:buy|sell|hold)\b", low)]
    return sorted(set(hits))


def watchlist_leaks(text: str, watchlist: list[str], allowed: list[str]) -> list[str]:
    leaks = []
    for entry in watchlist or []:
        comp = resolve(entry)
        names = {entry.lower()} | ({comp.ticker.lower(), *comp.aliases} if comp else set())
        if comp and comp.ticker in allowed:
            continue
        if entry.upper() in allowed:
            continue
        if any(re.search(rf"\b{re.escape(n)}\b", text.lower()) for n in names):
            leaks.append(entry)
    return leaks


def scrub(text: str, watchlist: list[str], allowed: list[str]) -> str:
    """Redact secrets and out-of-scope watchlist names from text that is logged or traced."""
    text = SECRET_RE.sub("[REDACTED_SECRET]", text)
    for entry in watchlist_leaks(text, watchlist, allowed):
        comp = resolve(entry)
        for name in {entry, *(comp.aliases if comp else ()), *((comp.ticker,) if comp else ())}:
            text = re.sub(rf"\b{re.escape(name)}\b", "[WATCHLIST]", text, flags=re.IGNORECASE)
    return text


def output_guard(note: str, refs: dict[str, dict[str, Any]], known_passage: Any, cfg: dict,
                 watchlist: list[str], allowed: list[str]) -> dict[str, Any]:
    """Return {"passed": bool, "issues": [...], "verified_share": float, "numbers": n}."""
    report = check_numbers(note, refs, cfg["critic"]["number_tolerance_rel"])
    issues: list[str] = [f"number {c.text}: {c.status} ({c.detail})" for c in report.issues]
    if cfg["guardrails"]["enabled"]:
        issues += [f"advice language: '{p}'" for p in advice_violations(note, cfg)]
        issues += [f"private watchlist name in note: {w}" for w in watchlist_leaks(note, watchlist, allowed)]
        bad, signals = is_injection(note, cfg)
        if bad:
            issues.append(f"instruction-like text in note: {signals[:2]}")
    for pid in cited_passages(note):
        if not known_passage(pid):
            issues.append(f"citation {pid} does not exist")
    return {"passed": not issues, "issues": issues, "verified_share": report.share, "numbers": report.counted,
            "verified": report.verified}

"""Research lead: question -> Plan (L10), and replanning when evidence contradicts the plan.

The LLM proposes companies, metrics, flags, sub-questions and an outline (PlanDraft). Code then
decides everything checkable:
  * tickers must be in the coverage list (others dropped, noted in plan.notes)
  * metrics must be in the catalog
  * fiscal years: explicit years from the question if given, else each company's latest N years
    with a 10-K value in XBRL (so "last three fiscal years" means different calendar periods for
    Microsoft (June year end) and Alphabet (December) - stated in the note's caveats)
In offline mode (and for C0) a rule-based planner produces the draft instead of the LLM.

replan(): when a specialist reports that a planned fiscal year does not exist (no XBRL value or no
10-K), the plan's years for that company are replaced by the latest available ones, the version
is bumped and the reason is recorded; the graph then shows the new plan to the analyst again.
"""

from __future__ import annotations

import re
from typing import Any

from ..companies import BY_TICKER, COMPANIES
from ..config import is_offline
from ..control.context import RunContext
from ..control.guardrails import mentioned_companies
from ..finance.metrics import ALL_METRICS, latest_years
from ..llm.models import chat, is_fatal
from ..observability.tracing import emit
from .schemas import CompanyYears, Plan, PlanDraft

METRIC_KEYWORDS = [
    (r"operating margin|margins?\b(?! of safety)", ["operating_margin", "operating_income", "revenue"]),
    (r"net margin|profitab", ["net_margin", "net_income", "revenue"]),
    (r"gross margin", ["gross_margin", "gross_profit", "revenue"]),
    (r"operating income|operating profit", ["operating_income"]),
    (r"net income|earnings\b", ["net_income"]),
    (r"revenue growth|grow(?:th|ing)|sales growth", ["revenue_growth", "revenue"]),
    (r"revenue|sales|top line", ["revenue"]),
    (r"r&d|research and development", ["rd_expense", "rd_intensity", "revenue"]),
    (r"free cash flow|fcf", ["free_cash_flow", "operating_cash_flow", "capex"]),
    (r"capex|capital expenditure|capital spending", ["capex"]),
    (r"cash flow from operations|operating cash flow", ["operating_cash_flow"]),
    (r"\beps\b|earnings per share", ["diluted_eps"]),
]
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def explicit_years(question: str) -> list[int]:
    years = {int(y) for y in re.findall(r"\b(?:FY|fiscal(?: year)?\s*)'?((?:19|20)\d\d)\b", question, re.IGNORECASE)}
    years |= {int(y) for y in re.findall(r"\b(20[12]\d)\b", question)}
    return sorted(years)


def rule_draft(question: str) -> PlanDraft:
    q = question.lower()
    tickers = mentioned_companies(question)
    metrics: list[str] = []
    for pattern, names in METRIC_KEYWORDS:
        if re.search(pattern, q):
            metrics += [m for m in names if m not in metrics]
    risk = bool(re.search(r"risk", q))
    changed_since = bool(re.search(r"since (?:our|the|my) (?:last|previous) note|what changed since", q))
    if not metrics and (changed_since or not risk):
        metrics = ["operating_margin", "operating_income", "revenue"]
    m = re.search(r"(?:last|past|previous)\s+(\d|one|two|three|four|five)\s+(?:fiscal\s+)?years", q)
    n = int(m.group(1)) if m and m.group(1).isdigit() else WORD_NUM.get(m.group(1), 3) if m else (2 if risk and not metrics else 3)
    names = ", ".join(BY_TICKER[t].name for t in tickers)
    subs = [f"What are {names}'s {', '.join(metrics) or 'key metrics'} for each fiscal year?"]
    if len(tickers) > 1:
        subs.append("How do the companies compare, given their different fiscal year ends?")
    if risk:
        subs.append("Which risk factors were added, removed or materially changed in the latest 10-K versus the prior year?")
    subs.append("What does MD&A say drove the changes?")
    outline = ["Summary", "Margins" if metrics else "Key figures", *(["Risk factor changes"] if risk else []), "Caveats"]
    return PlanDraft(goal=question.strip(), companies=tickers, n_years=max(1, min(n, 5)), metrics=metrics,
                     risk_diff=risk, what_changed_since_last_note=changed_since, sub_questions=subs, outline=outline)


def llm_draft(question: str, ctx: RunContext, preferences: dict, lessons: str) -> PlanDraft:
    cfg = ctx.cfg
    coverage = "; ".join(f"{c.ticker}: {c.name} (FY ends {c.fy_end})" for c in COMPANIES)
    available = "; ".join(f"{c.ticker} {latest_years(c, 3)}" for c in COMPANIES)
    system = cfg["prompts"]["lead"].format(coverage=coverage, available=available, preferences=preferences,
                                           lessons=lessons, metric_names=", ".join(ALL_METRICS))
    model = chat("lead", ctx).with_structured_output(PlanDraft, method="function_calling")
    return model.invoke([("system", system), ("human", f"<request>{question}</request>")])


def finalize(draft: PlanDraft, question: str, cfg: dict, version: int = 1) -> Plan:
    notes: list[str] = []
    tickers = []
    for t in draft.companies:
        t = t.upper().strip()
        if t in BY_TICKER and t not in tickers:
            tickers.append(t)
        elif t not in BY_TICKER:
            notes.append(f"{t} is not covered and was dropped from the plan")
    for t in mentioned_companies(question):  # the model must not silently drop a named company
        if t not in tickers:
            tickers.append(t)
    metrics = [m for m in dict.fromkeys(draft.metrics) if m in ALL_METRICS]
    dropped = [m for m in draft.metrics if m not in ALL_METRICS]
    if dropped:
        notes.append(f"unsupported metrics dropped: {dropped}")
    # inputs of ratios are always fetched too, so the note can show them
    expand = {"operating_margin": ["operating_income", "revenue"], "net_margin": ["net_income", "revenue"],
              "gross_margin": ["gross_profit", "revenue"], "rd_intensity": ["rd_expense", "revenue"],
              "revenue_growth": ["revenue"], "free_cash_flow": ["operating_cash_flow", "capex"]}
    for m in list(metrics):
        metrics += [x for x in expand.get(m, []) if x not in metrics]
    asked = explicit_years(question)
    years = []
    for t in tickers:
        if asked:
            ys = asked
        else:
            ys = latest_years(BY_TICKER[t], max(1, min(draft.n_years, 5))) or []
        years.append(CompanyYears(ticker=t, fiscal_years=ys))
    return Plan(goal=draft.goal, companies=tickers, years=years, metrics=metrics, risk_diff=draft.risk_diff,
                what_changed_since_last_note=draft.what_changed_since_last_note,
                sub_questions=draft.sub_questions[:6], outline=draft.outline[:6],
                token_budget=cfg["budgets"]["tokens_per_note"], version=version, notes=notes)


def make_plan(question: str, ctx: RunContext, preferences: dict, lessons: str) -> Plan:
    use_llm = not is_offline(ctx.cfg) and ctx.cfg["orchestration"]["mode"] == "team"
    draft = None
    if use_llm:
        try:
            draft = llm_draft(question, ctx, preferences, lessons)
        except Exception as exc:  # malformed output / provider error: fall back, and say so
            if is_fatal(exc):
                raise
            emit(ctx.run_id, "error", kind="planner_fallback", error=str(exc)[:300])
    plan = finalize(draft or rule_draft(question), question, ctx.cfg)
    if draft is None and use_llm:
        plan.notes.append("planner fell back to rules after an LLM error")
    return plan


def contradictions(quant_reports: list[dict], risk_reports: list[dict], plan: Plan) -> list[dict[str, Any]]:
    """Planned years the evidence says do not exist."""
    found = []
    for r in quant_reports:
        missing_years = sorted({int(m.split("/FY")[1].split(":")[0]) for m in r.get("missing", []) if "/FY" in m})
        planned = plan.years_for(r["ticker"])
        bad = [y for y in missing_years if y in planned]
        if bad and r.get("available_years"):
            found.append({"ticker": r["ticker"], "source": "quant", "missing_years": bad,
                          "available_years": r["available_years"]})
    for r in risk_reports:
        if r.get("error") and r.get("available_years"):
            found.append({"ticker": r["ticker"], "source": "risk_diff", "missing_years": [r["year_from"], r["year_to"]],
                          "available_years": r["available_years"], "error": r["error"]})
    return found


def replan(plan: Plan, found: list[dict[str, Any]]) -> tuple[Plan, str]:
    new = plan.model_copy(deep=True)
    reasons = []
    for c in found:
        cy = next(x for x in new.years if x.ticker == c["ticker"])
        available = sorted(c["available_years"])
        if c["source"] == "risk_diff":
            keep = [y for y in cy.fiscal_years if y in available]
            # risk diff needs the two latest filed years
            need = available[-2:]
            fixed = sorted(set(keep) | set(need))[-max(len(cy.fiscal_years), 2):]
        else:
            keep = [y for y in cy.fiscal_years if y not in c["missing_years"]]
            extra = [y for y in reversed(available) if y not in keep][: len(c["missing_years"])]
            fixed = sorted(set(keep) | set(extra))
        reasons.append(f"{c['ticker']}: FY{c['missing_years']} not available ({c['source']}); "
                       f"using {fixed} (available: {available[-5:]})")
        cy.fiscal_years = fixed
    new.version += 1
    new.notes = [*new.notes, *reasons]
    return new, "; ".join(reasons)

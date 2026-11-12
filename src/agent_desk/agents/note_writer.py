"""Writer: drafts the note. Code renders everything numeric; the model writes prose around it.

Code-rendered (never written by a model):
  * the fact list the writer sees: F1..Fn -> display string, period, fact_id
  * the peer table (when the analyst's preference says so) and the Sources section
  * the fiscal-year alignment caveat (period end dates per company)
The model (live) or a template (offline / C0 without LLM) writes the narrative and must copy
numbers with their [F#] reference. The critic checks that afterwards.
"""

from __future__ import annotations

import re
from typing import Any

from ..companies import BY_TICKER
from ..config import is_offline
from ..control.context import RunContext
from ..llm.models import chat
from .schemas import Plan

METRIC_LABEL = {"revenue": "Revenue", "operating_income": "Operating income", "operating_margin": "Operating margin",
                "net_income": "Net income", "net_margin": "Net margin", "gross_profit": "Gross profit",
                "gross_margin": "Gross margin", "rd_expense": "R&D expense", "rd_intensity": "R&D / revenue",
                "operating_cash_flow": "Operating cash flow", "capex": "Capital expenditure",
                "free_cash_flow": "Free cash flow", "revenue_growth": "Revenue growth", "diluted_eps": "Diluted EPS",
                "cost_of_revenue": "Cost of revenue"}
PRIMARY_ORDER = ["operating_margin", "net_margin", "gross_margin", "revenue_growth", "rd_intensity", "free_cash_flow",
                 "revenue", "operating_income", "net_income", "diluted_eps"]


def build_refs(facts: dict[str, dict[str, Any]], plan: Plan) -> dict[str, dict[str, Any]]:
    order = {m: i for i, m in enumerate(PRIMARY_ORDER)}
    chosen = [f for f in facts.values() if f["ticker"] in plan.companies and f["fiscal_year"] in plan.years_for(f["ticker"])]
    chosen.sort(key=lambda f: (plan.companies.index(f["ticker"]), order.get(f["metric"], 99), f["fiscal_year"]))
    return {f"F{i}": f for i, f in enumerate(chosen, start=1)}


def ref_for(refs: dict[str, dict], ticker: str, metric: str, year: int) -> tuple[str, dict] | None:
    return next(((r, f) for r, f in refs.items() if f["ticker"] == ticker and f["metric"] == metric and f["fiscal_year"] == year), None)


def facts_text(refs: dict[str, dict]) -> str:
    return "\n".join(f"[{r}] {f['ticker']} {METRIC_LABEL.get(f['metric'], f['metric'])} FY{f['fiscal_year']} "
                     f"(period {f.get('period_start') or '?'} to {f['period_end']}) = {f['display']}" for r, f in refs.items())


def alignment(plan: Plan, refs: dict[str, dict]) -> str:
    ends = {}
    for f in refs.values():
        ends.setdefault(f["ticker"], set()).add(f["period_end"][5:])
    parts = [f"{t}: fiscal year ends {BY_TICKER[t].fy_end} (latest period end in data: "
             f"{max((f['period_end'] for f in refs.values() if f['ticker'] == t), default='n/a')})" for t in plan.companies]
    months = {t: sorted(e)[0][:2] for t, e in ends.items()}
    if len(set(months.values())) > 1:
        parts.append("Fiscal years end in different months, so the same FY label covers different calendar periods; "
                     "comparisons are fiscal-year to fiscal-year, not calendar-aligned.")
    return " ".join(parts)


def peer_table(plan: Plan, refs: dict[str, dict]) -> str:
    cols = [(t, y) for t in plan.companies for y in plan.years_for(t)]
    metrics = [m for m in PRIMARY_ORDER if m in plan.metrics] + [m for m in plan.metrics if m not in PRIMARY_ORDER]
    rows = []
    for m in metrics:
        cells = []
        for t, y in cols:
            hit = ref_for(refs, t, m, y)
            cells.append(f"{hit[1]['display']} [{hit[0]}]" if hit else "n/a")
        if any(c != "n/a" for c in cells):
            rows.append(f"| {METRIC_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    if not rows:
        return ""
    header = "| Metric | " + " | ".join(f"{t} FY{y}" for t, y in cols) + " |"
    return "\n".join([header, "|---|" + "---|" * len(cols), *rows])


def sources(refs: dict[str, dict], passages: list[str], passage_lookup) -> str:
    lines = ["**Facts** (SEC XBRL; derived values computed in the sandbox from the listed inputs)"]
    for r, f in refs.items():
        extra = f" = {f['formula']} over {', '.join(f['inputs'])}" if f.get("kind") == "derived" else ""
        lines.append(f"- [{r}] `{f['fact_id']}` {f['display']}{extra}")
    if passages:
        lines.append("\n**Passages**")
        for pid in passages:
            p = passage_lookup(pid)
            lines.append(f"- [{pid}] {p.get('source', '') if p else '(not found)'}")
    return "\n".join(lines)


def template_body(question: str, plan: Plan, refs: dict, findings: list[dict], risks: list[dict], memory_lines: list[str],
                  plan_notes: list[str]) -> str:
    primary = next((m for m in PRIMARY_ORDER if m in plan.metrics), None)
    out = ["## Summary"]
    for t in plan.companies:
        name = BY_TICKER[t].name
        if not plan.metrics:
            r = next((r for r in risks if r["ticker"] == t), None)
            if r and not r.get("error"):
                out.append(f"- {name}: Item 1A Risk Factors compared FY{r['year_from']} vs FY{r['year_to']}; "
                           "the most material added, removed and reworded risks are quoted below with their passages.")
            continue
        if primary:
            vals = [f"FY{y} {h[1]['display']} [{h[0]}]" for y in plan.years_for(t) if (h := ref_for(refs, t, primary, y))]
            if vals:
                out.append(f"- {name}: {METRIC_LABEL[primary].lower()} " + ", ".join(vals) + ".")
                continue
        out.append(f"- {name}: no {METRIC_LABEL.get(primary, 'metric')} available for the planned years.")
    if plan.metrics:
        out.append("\n## Margins and key figures")
        for t in plan.companies:
            bits = []
            for m in [m for m in PRIMARY_ORDER if m in plan.metrics and m != primary][:3]:
                latest = max(plan.years_for(t), default=None)
                if latest and (h := ref_for(refs, t, m, latest)):
                    bits.append(f"{METRIC_LABEL[m].lower()} {h[1]['display']} [{h[0]}]")
            if bits:
                out.append(f"- {t} FY{max(plan.years_for(t))}: " + "; ".join(bits) + ".")
    if findings:
        out.append("\n## What the filings say")
        for f in findings:
            for item in f["findings"][:3]:
                out.append(f"- {f['ticker']}: \"{_quote(item['claim'])}\" [{item['passage_ids'][0]}]")
    if plan.risk_diff:
        out.append("\n## Risk factor changes")
        for r in risks:
            if r.get("error"):
                out.append(f"- {r['ticker']}: comparison not available ({r['error']}).")
                continue
            out.append(f"- {r['ticker']} FY{r['year_from']} vs FY{r['year_to']}:")
            for kind in ("added", "removed", "changed"):
                for p in r.get(kind, [])[:2]:
                    out.append(f"  - {kind.capitalize()}: \"{_quote(p['summary'])}\" [{p['passage_ids'][0]}]")
    if memory_lines:
        out.append("\n## What changed since our last note")
        out += [f"- {line}" for line in memory_lines]
    out.append("\n## Caveats")
    out.append(f"- {alignment(plan, refs)}")
    for n in plan_notes:
        out.append(f"- {n}")
    out.append("- This note summarises reported filings; it is not investment advice.")
    return "\n".join(out)


def _quote(text: str) -> str:
    """Verbatim filing text for a quotation: no inner double quotes, trailing ellipsis dropped."""
    text = re.sub(r"\s+", " ", text)  # a quotation is one line, so the verbatim-quote check applies
    return text.replace('"', "'").replace("“", "'").replace("”", "'").removesuffix("...").strip()


async def write_note(question: str, plan: Plan, facts: dict, findings: list[dict], risks: list[dict], prefs: dict,
                     memory_lines: list[str], lessons: str, ctx: RunContext, revision: str = "",
                     previous: str = "", passage_lookup=None) -> tuple[str, dict[str, dict]]:
    ctx.for_agent("writer")
    refs = build_refs(facts, plan)
    use_llm = not is_offline(ctx.cfg) and ctx.cfg["orchestration"]["mode"] in ("team", "fixed")
    if use_llm:
        prompt = ctx.cfg["prompts"]["writer"].format(
            facts=facts_text(refs) or "(none)",
            findings="\n".join(f"{f['ticker']}: {i['claim']} [{', '.join(i['passage_ids'])}]" for f in findings for i in f["findings"]) or "(none)",
            risks="\n".join(f"{r['ticker']} FY{r['year_from']}->FY{r['year_to']} {k}: {p['summary']} [{p['passage_ids'][0]}]"
                            for r in risks for k in ("added", "removed", "changed") for p in r.get(k, [])) or "(none)",
            alignment=alignment(plan, refs), preferences=prefs, memory="\n".join(memory_lines) or "(none)",
            lessons=lessons, question=question,
            revision=(f"REVISION REQUIRED. The compliance critic rejected the previous draft for these reasons:\n{revision}\n"
                      f"Previous draft:\n{previous}\nFix every issue; keep everything else.") if revision else "")
        msg = await chat("writer", ctx, max_tokens=1600).ainvoke([("system", prompt), ("human", "Write the note now.")])
        body = msg.content.strip()
    else:
        body = template_body(question, plan, refs, findings, risks, memory_lines, plan.notes)
    if prefs.get("peer_table", True) and (table := peer_table(plan, refs)):
        body += "\n\n## Peer table\n" + table
    cited = sorted({pid for pid in re.findall(r"\[([a-z0-9_]+-\d{4}-[0-9A-Za-z]{1,6}-\d{4}|web-[a-z0-9_]+-\d{2})\]", body)})
    body += "\n\n## Sources\n" + sources(refs, cited, passage_lookup or (lambda pid: None))
    return body, refs


def plant_wrong_number(note: str) -> tuple[str, str]:
    """Demo/E4: change the first percentage that carries a reference (e.g. 45.6% -> 50.2%)."""
    m = re.search(r"(\d+\.\d)%(\s*\[F\d+\])", note)
    if not m:
        return note, ""
    wrong = f"{float(m.group(1)) * 1.1:.1f}%"
    return note[:m.start()] + wrong + m.group(2) + note[m.end():], f"{m.group(1)}% -> {wrong}"

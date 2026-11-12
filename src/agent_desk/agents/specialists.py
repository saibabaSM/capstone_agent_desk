"""Specialists: filings researcher, quant, risk-diff (L11). ReAct over their own MCP tools.

Each specialist has two paths with identical tools, hooks and output types:
  react     create_react_agent(model, tools-for-this-agent, response_format=<Out model>)
            with short-term memory trimming (pre_model_hook) and a step limit
  scripted  a fixed sequence of the same tool calls (C0 fixed workflow, offline mode, CI)
After either path, code validates the hand-off: ids must exist AND have been returned by a tool
in this run; missing metric-years are fetched by code and recorded as `filled_by_code`.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.prebuilt import create_react_agent

from ..companies import BY_TICKER
from ..config import is_offline
from ..control.context import RunContext
from ..control.hooks import guarded_call
from ..llm.models import chat, is_fatal
from ..observability.tracing import emit
from ..tools.adapter import build_tools
from .schemas import (Finding, QuantOut, QuantReport, QuantTask, ResearchFindings, ResearchFindingsOut,
                      ResearchTask, RiskDiffOut, RiskDiffReport, RiskDiffTask, RiskPoint)


def use_react(cfg: dict) -> bool:
    return not is_offline(cfg) and cfg["orchestration"]["mode"] == "team"


def _trimmer(cfg: dict):
    limit = cfg["memory"]["short_term_max_tokens"]

    def hook(state: dict) -> dict:
        msgs = trim_messages(state["messages"], max_tokens=limit, strategy="last", token_counter=count_tokens_approximately,
                             start_on="human", include_system=True, allow_partial=False)
        return {"llm_input_messages": msgs}
    return hook


async def run_react(agent: str, ctx: RunContext, system: str, task: str, out_model: type) -> Any:
    tools = await build_tools(agent, ctx)
    graph = create_react_agent(chat("specialist", ctx), tools, prompt=system, response_format=out_model,
                               pre_model_hook=_trimmer(ctx.cfg), name=agent)
    steps = ctx.cfg["budgets"]["react_max_steps"]
    result = await graph.ainvoke({"messages": [HumanMessage(content=task)]}, {"recursion_limit": 2 * steps + 4})
    return result.get("structured_response")


def _sentence(text: str, limit: int = 260) -> str:
    """A verbatim excerpt: the first complete sentence of a passage (chunks can start mid-sentence)."""
    text = re.sub(r"\s+", " ", re.sub(r"^\[Table in Item [^\]]+\]\s*", "", text)).strip()
    sentences =[s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    if len(sentences) > 1 and not sentences[0][:1].isupper():
        sentences = sentences[1:]  # drop the fragment left by chunk overlap
    first = next((s for s in sentences if len(s) > 40), sentences[0] if sentences else text)
    return first[:limit].rstrip()


# ---------------------------------------------------------------- researcher

async def researcher(task: ResearchTask, ctx: RunContext, lessons: str = "") -> ResearchFindings:
    ctx.for_agent("researcher")
    comp = BY_TICKER[task.ticker]
    out: ResearchFindingsOut | None = None
    if use_react(ctx.cfg):
        system = ctx.cfg["prompts"]["researcher"].format(company=f"{comp.name} ({comp.ticker})", years=task.fiscal_years,
                                                         sub_questions="; ".join(task.sub_questions), lessons=lessons,
                                                         max_steps=ctx.cfg["budgets"]["react_max_steps"])
        try:
            out = await run_react("researcher", ctx, system,
                                  f"Research {comp.ticker} for fiscal years {task.fiscal_years}. Use company='{comp.ticker}' in searches.",
                                  ResearchFindingsOut)
        except Exception as exc:
            if is_fatal(exc):
                raise
            emit(ctx.run_id, "error", agent="researcher", ticker=comp.ticker, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    if out is None:
        out = await _scripted_research(task, ctx)
    return _validate_findings(comp.ticker, out, ctx)


async def _scripted_research(task: ResearchTask, ctx: RunContext) -> ResearchFindingsOut:
    latest = max(task.fiscal_years) if task.fiscal_years else None
    findings = []
    queries = [("what drove the change in revenue and operating income", "7"),
               ("most significant risks to the business", "1A"),
               (task.question or (task.sub_questions[0] if task.sub_questions else "recent developments"), None)]
    for q, item in queries:
        args = {"query": q, "company": task.ticker, "k": 2}
        if item:  # filings: the plan's fiscal year; the open query also reaches web text, which has no fiscal year
            args.update(item=item, fiscal_year=latest)
        res = await guarded_call(ctx, "researcher", "filings_kb.search_filings", args)
        for p in res.get("passages", [])[:2]:
            if p.get("injection_suspected") or p.get("kind") == "table":
                continue  # numbers come from XBRL; tables are not quoted as narrative findings
            findings.append(Finding(claim=_sentence(p["text"]), passage_ids=[p["passage_id"]]))
    return ResearchFindingsOut(findings=findings, gaps=[] if findings else task.sub_questions)


def _validate_findings(ticker: str, out: ResearchFindingsOut, ctx: RunContext) -> ResearchFindings:
    kept, dropped = [], []
    for f in out.findings:
        ids = [pid for pid in f.passage_ids if pid in ctx.passages_seen
               and not ctx.passages_seen[pid].get("injection_suspected")]
        if ids:
            kept.append(Finding(claim=f.claim.strip(), passage_ids=ids))
        else:
            dropped.append(f"{f.claim[:80]} (cited {f.passage_ids}: not returned by a tool or quarantined)")
    if dropped:
        emit(ctx.run_id, "guard", stage="handoff", agent="researcher", dropped=dropped)
    return ResearchFindings(ticker=ticker, findings=kept[:6], gaps=out.gaps, dropped=dropped)


# ---------------------------------------------------------------- quant

async def quant(task: QuantTask, ctx: RunContext, lessons: str = "") -> QuantReport:
    ctx.for_agent("quant")
    comp = BY_TICKER[task.ticker]
    chosen: list[str] = []
    if use_react(ctx.cfg):
        system = ctx.cfg["prompts"]["quant"].format(company=f"{comp.name} ({comp.ticker}), fiscal year ends {comp.fy_end}",
                                                    years=task.fiscal_years, metrics=task.metrics, lessons=lessons)
        try:
            out = await run_react("quant", ctx, system,
                                  f"Get {task.metrics} for {comp.ticker} fiscal years {task.fiscal_years}.", QuantOut)
            chosen = [fid for fid in (out.fact_ids if out else []) if fid in ctx.facts_seen]
        except Exception as exc:
            if is_fatal(exc):
                raise
            emit(ctx.run_id, "error", agent="quant", ticker=comp.ticker, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    # code check: every planned metric-year must be covered by a fact the tools returned
    have = {(f["metric"], f["fiscal_year"]) for f in ctx.facts_seen.values()}
    filled, missing, available = [], [], set()
    todo = [m for m in task.metrics if any((m, y) not in have for y in task.fiscal_years)]
    for metric in todo:
        years = [y for y in task.fiscal_years if (metric, y) not in have]
        res = await guarded_call(ctx, "quant", "edgar.get_metric", {"company": comp.ticker, "metric": metric, "fiscal_years": years})
        for err in res.get("errors", []):
            m = re.search(r"FY(\d{4})", err)
            missing.append(f"{metric}/FY{m.group(1) if m else '?'}: {err}")
            yrs = re.search(r"available fiscal years: \[([^\]]*)\]", err)
            if yrs:
                available |= {int(y) for y in yrs.group(1).split(",") if y.strip().isdigit()}
        if use_react(ctx.cfg) and res.get("facts"):
            filled.append(f"{metric} {years}")
    fact_ids = list(dict.fromkeys([*chosen, *[f["fact_id"] for f in ctx.facts_seen.values()
                                             if f["ticker"] == comp.ticker and f["metric"] in task.metrics
                                             and f["fiscal_year"] in task.fiscal_years]]))
    if filled:
        emit(ctx.run_id, "guard", stage="handoff", agent="quant", filled_by_code=filled)
    return QuantReport(ticker=comp.ticker, fact_ids=fact_ids, missing=missing, available_years=sorted(available),
                       filled_by_code=filled)


# ---------------------------------------------------------------- risk diff

async def risk_diff(task: RiskDiffTask, ctx: RunContext, lessons: str = "") -> RiskDiffReport:
    ctx.for_agent("risk_diff")
    comp = BY_TICKER[task.ticker]
    raw = await guarded_call(ctx, "risk_diff", "filings_kb.diff_risk_factors",
                             {"company": comp.ticker, "year_from": task.year_from, "year_to": task.year_to, "max_items": 6})
    if not raw.get("ok"):
        return RiskDiffReport(ticker=comp.ticker, year_from=task.year_from, year_to=task.year_to,
                              error=raw.get("error", "diff failed"), available_years=raw.get("available_years", []))
    out: RiskDiffOut | None = None
    if use_react(ctx.cfg):
        system = ctx.cfg["prompts"]["risk_diff"].format(company=f"{comp.name} ({comp.ticker})", year_from=task.year_from,
                                                        year_to=task.year_to)
        try:
            out = await run_react("risk_diff", ctx, system,
                                  f"The diff tool already ran; call filings_kb__diff_risk_factors(company='{comp.ticker}', "
                                  f"year_from={task.year_from}, year_to={task.year_to}) and read passages with get_passage "
                                  "where the summary is unclear. Then summarise.", RiskDiffOut)
        except Exception as exc:
            if is_fatal(exc):
                raise
            emit(ctx.run_id, "error", agent="risk_diff", ticker=comp.ticker, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    if out is None:
        out = RiskDiffOut(**{k: [RiskPoint(summary=i["summary"], passage_ids=[i["passage_id"]]) for i in raw.get(k, [])[:4]
                                 if "QUARANTINED" not in i["summary"]] for k in ("added", "removed", "changed")})

    def valid(points: list[RiskPoint]) -> list[RiskPoint]:
        return [RiskPoint(summary=p.summary, passage_ids=ids) for p in points
                if (ids := [i for i in p.passage_ids if i in ctx.passages_seen])][:4]

    return RiskDiffReport(ticker=comp.ticker, year_from=task.year_from, year_to=task.year_to, added=valid(out.added),
                          removed=valid(out.removed), changed=valid(out.changed), counts=raw.get("counts", {}))

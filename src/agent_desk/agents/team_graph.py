"""The Analyst Desk graph (LangGraph): typed state, plan-and-execute, parallel fan-out, critic loop,
approval gate, persistent checkpointer. Walkthrough numbers match docs/architecture.md.

 (1) input_guard -> recall (memory) -> (2) plan -> plan_review [interrupt: approve | edit | reject]
 (3)(4)(5) dispatch --Send--> researcher x company | quant x company | risk_diff x company   (parallel)
     gather --(a planned year does not exist)--> replan -> plan_review          (replanning, L10)
 (6) write -> critic --(fails, rounds left)--> write                          (self-correction, max 2)
     output_check -> (7) approval_gate [interrupt on the exact note + hash]
         approve -> publish (research-portal.publish_note with an HMAC token over the note hash)
         edit    -> recheck -> output_check -> approval_gate (the old approval is now invalid)
         reject  -> remember
 (8) remember (published note -> coverage memory) -> END
orchestration.mode = single routes dispatch to one ReAct agent with all tools (C1 baseline);
orchestration.mode = fixed uses scripted specialists and a rule planner (C0).
"""

from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph.types import Send, interrupt

from ..companies import BY_TICKER, COMPANIES
from ..config import config_fingerprint, with_overrides
from ..control.approval import issue_token, note_hash
from ..control.context import BudgetExceeded, RunContext, merge_usage
from ..control.guardrails import input_guard, output_guard
from ..control.hooks import guarded_call
from ..llm.models import chat, is_fatal
from ..memory import store as mem
from ..observability import metrics as M
from ..observability.tracing import emit
from ..rag import hybrid
from ..tools.adapter import build_tools
from ..improvement import playbook
from . import critic as critic_mod
from . import note_writer, planner, specialists
from .schemas import Plan, QuantTask, ResearchTask, RiskDiffTask

RESET = {"__reset__": True}


def add_or_reset(current: list | None, update: list | None) -> list:
    if update and update[0] == RESET:
        return list(update[1:])
    return (current or []) + (update or [])


def merge_or_reset(current: dict | None, update: dict | None) -> dict:
    if update and update.get("__reset__"):
        return {k: v for k, v in update.items() if k != "__reset__"}
    return {**(current or {}), **(update or {})}


class DeskState(TypedDict, total=False):
    run_id: str
    question: str
    analyst: str
    cfg_overrides: dict
    started_at: float
    status: str  # running | refused | plan_rejected | awaiting_approval | published | rejected | failed
    message: str
    guard: dict
    scope_tickers: list[str]
    watchlist: list[str]  # private; used only by guardrails/hooks
    prefs: dict
    lessons: list[dict]
    past_notes: dict
    plan: dict
    plan_history: Annotated[list, operator.add]
    replan_count: int
    findings: Annotated[list, add_or_reset]
    quant_reports: Annotated[list, add_or_reset]
    risk_reports: Annotated[list, add_or_reset]
    facts: Annotated[dict, merge_or_reset]
    passages: Annotated[dict, merge_or_reset]
    draft: str
    refs: dict
    planted: str
    critic_round: int
    critic_reports: Annotated[list, operator.add]
    output_check: dict
    note: str
    note_hash: str
    approval: dict
    approval_events: Annotated[list, operator.add]
    published: dict
    usage: Annotated[dict, merge_usage]


def _cfg(state: dict) -> dict:
    return with_overrides(state.get("cfg_overrides"))


def _ctx(state: dict, agent: str) -> RunContext:
    return RunContext.from_state(state, _cfg(state), agent)


def _lookup(state: dict):
    seen = state.get("passages") or {}

    def get(pid: str) -> dict | None:
        try:
            doc = hybrid.get_passage(pid)
        except FileNotFoundError:
            doc = None
        return doc or seen.get(pid)
    return get


def _lesson_text(state: dict, extra: str = "") -> str:
    cfg = _cfg(state)
    if not cfg["improvement"]["playbook_enabled"]:
        return ""
    return playbook.as_prompt(playbook.relevant(state.get("lessons") or [], f"{state['question']} {extra}", cfg))


# ------------------------------------------------------------------------- (1) guard + memory

async def input_guard_node(state: DeskState) -> dict:
    cfg = _cfg(state)
    g = input_guard(state["question"], cfg)
    emit(state["run_id"], "guard", stage="input", allowed=g.allowed, category=g.category, reason=g.reason,
         config_version=cfg["version"], config_hash=config_fingerprint(cfg), mode=cfg["orchestration"]["mode"])
    if not g.allowed:
        M.GUARD_BLOCKS.labels(f"input_{g.category}").inc()
        M.NOTES.labels("refused").inc()
        return {"guard": g.__dict__, "status": "refused", "message": g.reason}
    return {"guard": g.__dict__, "scope_tickers": g.tickers, "status": "running"}


async def recall_node(state: DeskState, *, store: BaseStore) -> dict:
    cfg = _cfg(state)
    analyst = state.get("analyst", "analyst")
    out: dict[str, Any] = {"watchlist": await mem.get_watchlist(store, analyst), "prefs": {}, "past_notes": {}, "lessons": []}
    if cfg["memory"]["enabled"]:
        out["prefs"] = await mem.get_prefs(store, analyst)
        for t in state.get("scope_tickers", []):
            if (prev := await mem.last_note(store, t)):
                out["past_notes"][t] = prev
    else:
        out["prefs"] = dict(mem.DEFAULT_PREFS)
    if cfg["improvement"]["playbook_enabled"]:
        out["lessons"] = await mem.get_lessons(store, "playbook")
    emit(state["run_id"], "memory", action="recall", memory_enabled=cfg["memory"]["enabled"],
         prefs=out["prefs"], past_notes=sorted(out["past_notes"]), lessons=len(out["lessons"]),
         watchlist_entries=len(out["watchlist"]))
    return out


# ------------------------------------------------------------------------- (2) plan

async def plan_node(state: DeskState) -> dict:
    ctx = _ctx(state, "research_lead")
    try:
        plan = planner.make_plan(state["question"], ctx, state.get("prefs") or {}, _lesson_text(state))
    except BudgetExceeded as exc:
        return {"status": "failed", "message": str(exc), "usage": ctx.usage_delta()}
    emit(state["run_id"], "plan", version=plan.version, plan=plan.model_dump())
    return {"plan": plan.model_dump(), "plan_history": [{"version": plan.version, "reason": "initial", "plan": plan.model_dump()}],
            "replan_count": state.get("replan_count", 0), "usage": ctx.usage_delta()}


async def plan_review_node(state: DeskState) -> dict:
    cfg = _cfg(state)
    plan = Plan(**state["plan"])
    if cfg["orchestration"]["auto_approve_plan"]:
        return {}
    answer = interrupt({"type": "plan_review", "plan": plan.model_dump(), "question": state["question"],
                        "instructions": "approve, edit (send a modified plan) or reject"})
    action = (answer or {}).get("action", "approve")
    emit(state["run_id"], "approval", stage="plan", action=action)
    if action == "reject":
        M.NOTES.labels("plan_rejected").inc()
        return {"status": "plan_rejected", "message": "The analyst rejected the plan."}
    if action == "edit" and answer.get("plan"):
        edited = {**plan.model_dump(), **answer["plan"]}
        edited["companies"] = [t.upper() for t in edited["companies"] if t.upper() in BY_TICKER]
        edited["years"] = [y for y in edited["years"] if y["ticker"].upper() in edited["companies"]]
        for t in edited["companies"]:  # an edited plan may add a company without years
            if not any(y["ticker"] == t for y in edited["years"]):
                edited["years"].append({"ticker": t, "fiscal_years": planner.latest_years(BY_TICKER[t], 3)})
        new = Plan(**{**edited, "version": plan.version + 1})
        emit(state["run_id"], "plan", version=new.version, plan=new.model_dump(), edited_by="analyst")
        return {"plan": new.model_dump(), "plan_history": [{"version": new.version, "reason": "edited by analyst", "plan": new.model_dump()}]}
    return {}


def after_plan_review(state: DeskState) -> list[Send] | str:
    if state.get("status") in ("plan_rejected", "failed"):
        return END
    cfg = _cfg(state)
    if cfg["orchestration"]["mode"] == "single":
        return "single_agent"
    plan = Plan(**state["plan"])
    base = {k: state.get(k) for k in ("run_id", "question", "cfg_overrides", "watchlist", "usage", "lessons", "plan")}
    sends = []
    for t in plan.companies:
        years = plan.years_for(t)
        sends.append(Send("researcher", {**base, "task": ResearchTask(ticker=t, fiscal_years=years, sub_questions=plan.sub_questions, question=state["question"]).model_dump()}))
        if plan.metrics:
            sends.append(Send("quant", {**base, "task": QuantTask(ticker=t, fiscal_years=years, metrics=plan.metrics).model_dump()}))
        if plan.risk_diff and len(years) >= 1:
            y_to = max(years)
            y_from = sorted(years)[-2] if len(years) >= 2 else y_to - 1
            sends.append(Send("risk_diff", {**base, "task": RiskDiffTask(ticker=t, year_from=y_from, year_to=y_to).model_dump()}))
    return sends or "gather"


# ------------------------------------------------------------------------- (3)(4)(5) specialists

async def researcher_node(payload: dict) -> dict:
    ctx = _ctx(payload, "researcher")
    task = ResearchTask(**payload["task"])
    try:
        report = await specialists.researcher(task, ctx, _lesson_text(payload, "filings research"))
        out = report.model_dump()
    except BudgetExceeded as exc:
        out = {"ticker": task.ticker, "findings": [], "gaps": [str(exc)], "dropped": []}
    emit(payload["run_id"], "node", node="researcher", ticker=task.ticker, findings=len(out["findings"]), gaps=out["gaps"])
    return {"findings": [out], "passages": ctx.passages_seen, "usage": ctx.usage_delta()}


async def quant_node(payload: dict) -> dict:
    ctx = _ctx(payload, "quant")
    task = QuantTask(**payload["task"])
    try:
        report = (await specialists.quant(task, ctx, _lesson_text(payload, "numbers fiscal years units"))).model_dump()
    except BudgetExceeded as exc:
        report = {"ticker": task.ticker, "fact_ids": [], "missing": [str(exc)], "available_years": []}
    # the registry keeps every fact the tools returned (inputs of derived facts included)
    emit(payload["run_id"], "node", node="quant", ticker=task.ticker, facts=len(report["fact_ids"]), missing=report["missing"])
    return {"quant_reports": [report], "facts": ctx.facts_seen, "usage": ctx.usage_delta()}


async def risk_diff_node(payload: dict) -> dict:
    ctx = _ctx(payload, "risk_diff")
    task = RiskDiffTask(**payload["task"])
    try:
        report = (await specialists.risk_diff(task, ctx, _lesson_text(payload, "risk factors"))).model_dump()
    except BudgetExceeded as exc:
        report = {"ticker": task.ticker, "year_from": task.year_from, "year_to": task.year_to, "error": str(exc)}
    emit(payload["run_id"], "node", node="risk_diff", ticker=task.ticker, counts=report.get("counts"), error=report.get("error", ""))
    return {"risk_reports": [report], "passages": ctx.passages_seen, "usage": ctx.usage_delta()}


async def gather_node(state: DeskState) -> dict:
    return {}


def after_gather(state: DeskState) -> str:
    cfg = _cfg(state)
    plan = Plan(**state["plan"])
    found = planner.contradictions(state.get("quant_reports", []), state.get("risk_reports", []), plan)
    if found and state.get("replan_count", 0) < cfg["budgets"]["max_replans"]:
        new, _ = planner.replan(plan, found)
        if [y.model_dump() for y in new.years] != [y.model_dump() for y in plan.years]:
            return "replan"  # only when the evidence actually changes the plan
    return "write"


async def replan_node(state: DeskState) -> dict:
    plan = Plan(**state["plan"])
    found = planner.contradictions(state.get("quant_reports", []), state.get("risk_reports", []), plan)
    new, reason = planner.replan(plan, found)
    M.REPLANS.inc()
    emit(state["run_id"], "replan", from_version=plan.version, to_version=new.version, reason=reason, evidence=found)
    return {"plan": new.model_dump(), "plan_history": [{"version": new.version, "reason": reason, "plan": new.model_dump()}],
            "replan_count": state.get("replan_count", 0) + 1,
            "findings": [RESET], "quant_reports": [RESET], "risk_reports": [RESET]}


# ------------------------------------------------------------------------- (6) write + critic

def _memory_lines(state: DeskState) -> list[str]:
    plan = Plan(**state["plan"])
    if not plan.what_changed_since_last_note or not _cfg(state)["memory"]["enabled"]:
        return []
    lines = []
    for t in plan.companies:
        lines += mem.what_changed((state.get("past_notes") or {}).get(t), state.get("facts") or {}, t)
    return lines


async def write_node(state: DeskState) -> dict:
    ctx = _ctx(state, "writer")
    cfg = ctx.cfg
    plan = Plan(**state["plan"])
    last = (state.get("critic_reports") or [None])[-1]
    revision = "\n".join(last["issues"]) if last and not last["passed"] and state.get("critic_round", 0) > 0 else ""
    try:
        draft, refs = await note_writer.write_note(
            state["question"], plan, state.get("facts") or {}, state.get("findings") or [], state.get("risk_reports") or [],
            state.get("prefs") or {}, _memory_lines(state), _lesson_text(state, "writing note"), ctx,
            revision=revision, previous=state.get("draft", ""), passage_lookup=_lookup(state))
    except BudgetExceeded as exc:
        return {"status": "failed", "message": str(exc), "usage": ctx.usage_delta()}
    planted = state.get("planted", "")
    if cfg["demo"]["plant_wrong_number"] and not state.get("critic_round") and not planted:
        draft, planted = note_writer.plant_wrong_number(draft)
        emit(state["run_id"], "node", node="writer", planted_error=planted)
    emit(state["run_id"], "node", node="writer", round=state.get("critic_round", 0), chars=len(draft), revision=bool(revision))
    return {"draft": draft, "refs": refs, "planted": planted, "usage": ctx.usage_delta()}


async def critic_node(state: DeskState) -> dict:
    ctx = _ctx(state, "critic")
    report = await critic_mod.critique(state["draft"], state["refs"], state.get("facts") or {}, _lookup(state), ctx)
    rnd = state.get("critic_round", 0) + 1
    if not report["passed"]:
        M.CRITIC_REJECTIONS.inc()
    emit(state["run_id"], "critic", round=rnd, passed=report["passed"], issues=report["issues"],
         verified_share=report["verified_share"], claims=f"{report['claims_supported']}/{report['claims_checked']}")
    return {"critic_reports": [{k: v for k, v in report.items() if k != "claims"} | {"round": rnd}],
            "critic_round": rnd, "usage": ctx.usage_delta()}


def after_write(state: DeskState) -> str:
    if state.get("status") == "failed":
        return END
    return "critic" if _cfg(state)["critic"]["enabled"] else "output_check"


def after_critic(state: DeskState) -> str:
    # critic_round 1 is the first check; at most critic_max_rounds revisions follow it
    last = state["critic_reports"][-1]
    if not last["passed"] and state.get("critic_round", 0) <= _cfg(state)["budgets"]["critic_max_rounds"]:
        return "write"
    return "output_check"


async def output_check_node(state: DeskState) -> dict:
    cfg = _cfg(state)
    note = state.get("note") if state.get("approval", {}).get("status") == "edited" else state["draft"]
    plan = Plan(**state["plan"])
    check = output_guard(note, state.get("refs") or {}, _lookup(state), cfg, state.get("watchlist") or [], plan.companies)
    M.NUMBERS.labels("true").inc(check["verified"])
    M.NUMBERS.labels("false").inc(check["numbers"] - check["verified"])
    M.VERIFIED_SHARE.set(check["verified_share"])
    if not check["passed"]:
        M.GUARD_BLOCKS.labels("output").inc()
    emit(state["run_id"], "guard", stage="output", passed=check["passed"], issues=check["issues"],
         verified_share=check["verified_share"], numbers=check["numbers"])
    if state.get("started_at"):
        M.RUN_LATENCY.observe(time.time() - state["started_at"])
    return {"output_check": check, "note": note, "note_hash": note_hash(note, plan.companies)}


# ------------------------------------------------------------------------- (7) approval + publish

async def approval_gate_node(state: DeskState) -> dict:
    cfg = _cfg(state)
    plan = Plan(**state["plan"])
    note, h, check = state["note"], state["note_hash"], state["output_check"]
    if cfg["orchestration"]["auto_approve_note"] or not cfg["approval"]["required"]:
        if check["passed"]:
            token = issue_token(h, "auto-approver")
            M.APPROVALS.labels("auto_approve").inc()
            return {"approval": {"status": "approved", "approver": "auto-approver", "token": token},
                    "approval_events": [{"action": "auto_approve", "hash": h}]}
        M.APPROVALS.labels("auto_reject").inc()
        return {"approval": {"status": "rejected", "approver": "auto-approver", "reason": "; ".join(check["issues"])[:500]},
                "approval_events": [{"action": "auto_reject", "hash": h}], "status": "rejected"}
    answer = interrupt({"type": "note_approval", "note": note, "note_hash": h, "tickers": plan.companies,
                        "output_check": check, "critic_rounds": state.get("critic_round", 0),
                        "instructions": "approve (send note_hash you reviewed) | edit (send edited_note) | reject"})
    action = (answer or {}).get("action", "")
    approver = (answer or {}).get("approver", "analyst")
    emit(state["run_id"], "approval", stage="note", action=action, approver=approver, hash=h[:16])
    M.APPROVALS.labels(action or "none").inc()
    if action == "reject":
        return {"approval": {"status": "rejected", "approver": approver, "reason": answer.get("reason", "")},
                "approval_events": [{"action": "reject", "hash": h}], "status": "rejected"}
    if action == "edit":
        edited = answer.get("edited_note", "").strip()
        return {"note": edited, "approval": {"status": "edited", "approver": approver, "previous_hash": h},
                "approval_events": [{"action": "edit", "hash": h, "new_hash": note_hash(edited, plan.companies)}]}
    if action == "approve":
        if answer.get("note_hash") != h:
            return {"approval": {"status": "stale", "reason": "approval was for a different version of the note"},
                    "approval_events": [{"action": "stale_approval", "hash": answer.get("note_hash"), "current": h}]}
        if not check["passed"]:
            return {"approval": {"status": "blocked", "reason": "output checks failed; edit the note first: " + "; ".join(check["issues"])[:400]},
                    "approval_events": [{"action": "approve_blocked", "hash": h}]}
        return {"approval": {"status": "approved", "approver": approver, "token": issue_token(h, approver)},
                "approval_events": [{"action": "approve", "hash": h}]}
    return {"approval": {"status": "invalid", "reason": f"unknown action {action!r}"}}


def after_gate(state: DeskState) -> str:
    status = state.get("approval", {}).get("status")
    if status == "approved":
        return "publish"
    if status == "edited":
        return "recheck"
    if status == "rejected":
        return "remember"
    return "approval_gate"  # stale / blocked / invalid: ask again


async def recheck_node(state: DeskState) -> dict:
    """An analyst edit invalidates the approval; numbers and wording are re-checked by code."""
    ctx = _ctx(state, "critic")
    report = await critic_mod.critique(state["note"], state["refs"], state.get("facts") or {}, _lookup(state), ctx,
                                       check_claims=False)
    emit(state["run_id"], "critic", round="after_edit", passed=report["passed"], issues=report["issues"])
    return {"critic_reports": [{k: v for k, v in report.items() if k != "claims"} | {"round": "after_edit"}], "usage": ctx.usage_delta()}


async def publish_node(state: DeskState) -> dict:
    ctx = _ctx(state, "publisher")
    plan = Plan(**state["plan"])
    res = await guarded_call(ctx, "publisher", "research_portal.publish_note",
                             {"note_markdown": state["note"], "tickers": plan.companies, "approval": state["approval"]["token"]})
    emit(state["run_id"], "publish", ok=res.get("ok"), note_id=res.get("note_id"), error=res.get("error", ""))
    status = "published" if res.get("ok") else "failed"
    M.NOTES.labels(status).inc()
    M.NOTE_COST.observe((state.get("usage") or {}).get("cost_usd", 0.0) + ctx.cost_usd)
    return {"published": res, "status": status, "message": res.get("error", ""), "usage": ctx.usage_delta()}


async def remember_node(state: DeskState, *, store: BaseStore) -> dict:
    cfg = _cfg(state)
    if state.get("status") != "published" or not cfg["memory"]["enabled"]:
        if state.get("status") == "rejected":
            M.NOTES.labels("rejected").inc()
        return {}
    plan = Plan(**state["plan"])
    for t in plan.companies:
        key_facts = {f"{f['metric']}/FY{f['fiscal_year']}": {"value": f["value"], "display": f["display"], "fact_id": f["fact_id"]}
                     for f in (state.get("refs") or {}).values() if f["ticker"] == t}
        risk = next((r.get("counts") for r in state.get("risk_reports", []) if r["ticker"] == t and not r.get("error")), None)
        await mem.add_note_record(store, t, {"note_id": state["published"].get("note_id"), "question": state["question"],
                                             "key_facts": key_facts, "years": plan.years_for(t), "risk_counts": risk,
                                             "note_hash": state["note_hash"], "analyst": state.get("analyst")})
    await mem.add_coverage(store, state.get("analyst", "analyst"), plan.companies)
    emit(state["run_id"], "memory", action="store_note", tickers=plan.companies)
    return {}


# ------------------------------------------------------------------------- C1: single agent

FACT_ID_RE = re.compile(r"\[([A-Z]{1,5}/[A-Za-z_:]+/FY\d{4}/[0-9\-]+|[A-Z]{1,5}/[a-z_]+/FY\d{4}/calc)\]")


async def single_agent_node(state: DeskState) -> dict:
    ctx = _ctx(state, "single_agent")
    cfg = ctx.cfg
    plan = Plan(**state["plan"])
    coverage = "; ".join(f"{c.ticker} ({c.name}, FY ends {c.fy_end})" for c in COMPANIES)
    system = cfg["prompts"]["single_agent"].format(coverage=coverage, max_steps=cfg["budgets"]["react_max_steps"] * 3)
    tools = await build_tools("single_agent", ctx)
    agent = create_react_agent(chat("single_agent", ctx, max_tokens=1800), tools, prompt=system,
                               pre_model_hook=specialists._trimmer(cfg), name="single_agent")
    task = (f"{state['question']}\nPlan years per company: "
            + ", ".join(f"{t} {plan.years_for(t)}" for t in plan.companies))
    body = ""
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=task)]},
                                     {"recursion_limit": 2 * cfg["budgets"]["react_max_steps"] * 3 + 4})
        body = str(result["messages"][-1].content)
    except Exception as exc:
        if is_fatal(exc):
            raise
        body = f"## Summary\nThe single agent did not finish: {type(exc).__name__}."
        emit(state["run_id"], "error", agent="single_agent", error=str(exc)[:300])
    # code converts the agent's [fact_id] citations into [F#] refs over facts the tools returned
    refs: dict[str, dict] = {}
    by_id = {}
    for fid in dict.fromkeys(FACT_ID_RE.findall(body)):
        if fid in ctx.facts_seen:
            ref = f"F{len(refs) + 1}"
            refs[ref] = ctx.facts_seen[fid]
            by_id[fid] = ref
    body = FACT_ID_RE.sub(lambda m: f"[{by_id.get(m.group(1), 'F?')}]", body)
    body += "\n\n## Sources\n" + note_writer.sources(refs, [], lambda pid: None)
    emit(state["run_id"], "node", node="single_agent", facts=len(refs), tool_calls=ctx.tool_calls)
    return {"draft": body, "refs": refs, "facts": ctx.facts_seen, "passages": ctx.passages_seen, "usage": ctx.usage_delta()}


# ------------------------------------------------------------------------- build

def build_graph():
    g = StateGraph(DeskState)
    for name, fn in [("input_guard", input_guard_node), ("recall", recall_node), ("plan", plan_node),
                     ("plan_review", plan_review_node), ("researcher", researcher_node), ("quant", quant_node),
                     ("risk_diff", risk_diff_node), ("gather", gather_node), ("replan", replan_node),
                     ("write", write_node), ("critic", critic_node), ("output_check", output_check_node),
                     ("approval_gate", approval_gate_node), ("recheck", recheck_node), ("publish", publish_node),
                     ("remember", remember_node), ("single_agent", single_agent_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: END if s.get("status") == "refused" else "recall", ["recall", END])
    g.add_edge("recall", "plan")
    g.add_conditional_edges("plan", lambda s: END if s.get("status") == "failed" else "plan_review", ["plan_review", END])
    g.add_conditional_edges("plan_review", after_plan_review, ["researcher", "quant", "risk_diff", "gather", "single_agent", END])
    for n in ("researcher", "quant", "risk_diff"):
        g.add_edge(n, "gather")
    g.add_conditional_edges("gather", after_gather, ["replan", "write"])
    g.add_edge("replan", "plan_review")
    g.add_conditional_edges("write", after_write, ["critic", "output_check", END])
    g.add_conditional_edges("critic", after_critic, ["write", "output_check"])
    g.add_edge("single_agent", "output_check")
    g.add_edge("output_check", "approval_gate")
    g.add_conditional_edges("approval_gate", after_gate, ["publish", "recheck", "remember", "approval_gate"])
    g.add_edge("recheck", "output_check")
    g.add_edge("publish", "remember")
    g.add_edge("remember", END)
    return g

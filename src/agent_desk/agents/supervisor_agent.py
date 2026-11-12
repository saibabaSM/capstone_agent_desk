"""Supervisor decision logic for the L11-style team supervisor.

This file keeps only the reasoning/selection behavior. The workflow graph and
shared state schema live in workflow.py and team_state.py.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm.llm import get_tool_llm
from .team_state import TeamState

WORKERS = {"researcher_worker", "quant_worker", "risk_worker"}
VALID_ROUTES = WORKERS | {"REPLAN", "FINISH"}

# The supervisor's reply is one short JSON object (a route plus a brief or a final report). The cap
# stops a model that loops inside the "message" string from generating for minutes: one such call
# produced 115,571 tokens, cost $0.06 and ended in LengthFinishReasonError.
MAX_ROUTE_TOKENS = 1000
# Calls to make before giving up; the runaway is random, so a second try usually works.
ROUTE_ATTEMPTS = 2

#choice
#{'next': 'quant_worker', 
# 'message': "What was Microsoft's operating income for fiscal year 2023 and revenue for fiscal year 2023? 
# Also, what was Microsoft's operating income for fiscal year 2024 and revenue for fiscal year 2024? 
# What was Microsoft's operating income for fiscal year 2025 and revenue for fiscal year 2025?"}
#
#{'next': 'researcher_worker', 
# 'message': 'Find the risk factors disclosed by Microsoft in their 10-K filings for fiscal years 2023, 2024, and 2025, 
# specifically in Item 1A.'}
class Route(TypedDict):
    next: Literal["researcher_worker", "quant_worker", "risk_worker", "REPLAN", "FINISH"]
    message: str

# example:
#    1. What was Apple's operating income for FY2025?
# -> 2. What was Apple's revenue for FY2025?
#    3. What were Apple's risk factors for FY2025?
def _plan_text(plan: dict, step: int) -> str:
    questions = plan.get("sub_questions", [])
    if not questions:
        return "No plan is available."
    if step >= len(questions):
        return (
            f"Goal: {plan.get('goal', '')}\n"
            f"All {len(questions)} sub-questions have been answered. Choose FINISH and write "
            "the final report."
        )
    lines = [f"{'->' if i == step else '  '} {i + 1}. {q}" for i, q in enumerate(questions)]
    return f"Goal: {plan.get('goal', '')}\nSub-questions (current one marked ->):\n" + "\n".join(lines)

#choice
#{'next': 'quant_worker', 'message': "What was Microsoft's operating margin for fiscal years 2023, 2024 and 2025?"}
def supervisor_decision(state: TeamState) -> dict:
    """Choose the next worker (or REPLAN / FINISH) and write the brief for it."""
    replan_count = state.get("replan_count", 0)
    replans_left = state.get("max_replans", 2) - replan_count

    messages = state["messages"]
    last = messages[-1] if messages else None
    step = state.get("plan_step", 0)
    # A successful worker report means the sub-question it was working on is answered.
    if getattr(last, "name", None) in WORKERS and not str(last.content).startswith("WORKER FAILED"):
        step += 1

    system = (
        "You are the supervisor. You coordinate these workers and never do their work yourself:\n"
        "- researcher_worker: searches 10-K filing text (Item 1A risk factors, Item 7 MD&A, etc.) "
        "for a company and fiscal year and returns narrative passages with citations. It reports "
        "what a filing says in one year; it does not compare years.\n"
        "- quant_worker: looks up reported numbers (revenue, operating income, ...) for a company "
        "and one or more fiscal years from SEC Company Facts and computes ratios. It cannot "
        "return dates or text.\n"
        "- risk_worker: compares one company's Item 1A risk factors between two fiscal years and "
        "returns the risks that were added, removed or materially changed, with passage ids. Use "
        "it for every question about what changed, was added or was removed in risk factors; "
        "the brief must name the company and both fiscal years. It only has the filings stored "
        "locally and answers with the years that are available when a year is missing.\n"
        "- REPLAN: ask the research lead for a new plan. Choose it when a worker reports an error "
        "(WORKER FAILED), an empty result, 'no passages', 'could not find', or a filing that is "
        "not available for a year, or when the plan cannot be answered with the workers' tools. "
        "Put the reason in `message`, including any available years a worker listed. "
        f"Replans left: {replans_left}.\n"
        "- FINISH: stop when the answer is complete or cannot be completed; put your final "
        "report in `message`.\n\n"
        f"{_plan_text(state.get('plan') or {}, step)}\n\n"
        "Write the brief for the current sub-question (marked ->) only, as one concrete question "
        "the chosen worker's tools can answer, with the company, fiscal year and metric spelled "
        "out. Keep the company, metric and every fiscal year from the current sub-question in "
        "the brief; do not narrow a multi-year sub-question to a single year. Do not work on "
        "later sub-questions yet. If the current sub-question cannot be "
        "answered by any worker as written, rephrase it into a question one can answer; if that "
        "is impossible, choose REPLAN. A worker sees only the brief you write for it, not the "
        "conversation. Never ask for something a worker has already reported. One worker at a time."
    )

    router = get_tool_llm(max_tokens=MAX_ROUTE_TOKENS).with_structured_output(
        Route,
        method="json_schema",
        strict=True,
        include_raw=True,
    )

    # Ask up to ROUTE_ATTEMPTS times. A call that raises (for example LengthFinishReasonError when
    # the reply hits the cap) or returns nothing parseable is retried; the last problem is kept so
    # the final STOPPED message can say what really went wrong.
    choice: dict = {}
    problem = ""
    for _ in range(ROUTE_ATTEMPTS):
        try:
            out = router.invoke([SystemMessage(content=system), *messages])
        except Exception as exc:
            problem = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        choice = out["parsed"] or {}
        if choice.get("next") in VALID_ROUTES:
            break
        problem = f"unparseable reply: {out.get('parsing_error') or choice or 'empty'}"
        choice = {}

    next_step = choice.get("next")
    message = choice.get("message", "")
    if next_step not in VALID_ROUTES:
        next_step = "FINISH"
        message = f"STOPPED: supervisor returned no usable route after {ROUTE_ATTEMPTS} attempts ({problem})"
    if next_step == "REPLAN" and replans_left <= 0:
        next_step = "FINISH"
        message = f"Could not complete the request after {replan_count} replans. Last problem: {message}"

    update: dict = {
        "next": next_step,
        "messages": [AIMessage(content=f"[to {next_step}] {message}", name="supervisor")],
    }
    if next_step == "REPLAN":
        update["replan_reason"] = message
    else:
        update["task"] = message
        update["plan_step"] = step
    return update


def run_supervisor(question: str) -> str:
    """Convenience wrapper when the supervisor is called directly."""
    initial: TeamState = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "next": "",
        "task": "",
        "plan": {},
        "plan_step": 0,
        "replan_count": 0,
        "max_replans": 2,
        "replan_reason": "",
        "narrative": [],
        "facts": [],
        "risk": [],
        "final_answer": "",
    }
    return supervisor_decision(initial)["task"]

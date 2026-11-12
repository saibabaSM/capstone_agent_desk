"""Coordinator: routes a user question to the right specialist agents."""

from __future__ import annotations

import json

from .quant_agent import run_quant_agent
from .research_lead_agent import run_research_lead
from .researcher_agent import run_researcher


def analyst_edit_interrupt(plan: dict) -> dict:
    """Human-in-the-loop approval gate for the research plan."""
    print(json.dumps(plan, indent=2))
    edit = input("Press Enter to accept, or paste a revised JSON plan: ").strip()
    if not edit:
        return plan
    return json.loads(edit)


def run_coordinator(question: str) -> str:
    """Run the research lead, approve the plan, and route to the correct worker(s)."""
    brief = run_research_lead(question)
    plan = json.loads(brief)
    plan = analyst_edit_interrupt(plan)

    route = plan.get("route", "both")
    if route == "filing_retrieval":
        return run_researcher(question)

    if route == "company_facts":
        return run_quant_agent(question)

    filing = run_researcher(question)
    facts = run_quant_agent(question)
    return (
        "Narrative findings:\n" + filing
        + "\n\nQuant findings:\n" + facts
    )

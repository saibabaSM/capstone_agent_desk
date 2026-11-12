"""Risk-diff specialist: report what changed in a company's Item 1A risk factors between years."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from ..llm.llm import get_tool_llm
from ..tools.risk_tools import compare_risk_factors

TOOLS = [compare_risk_factors]

SYSTEM_PROMPT = """You are the risk-diff specialist for a filings research assistant.
Answer using only the compare_risk_factors tool.

From the question, find the company and the two fiscal years to compare, then call the tool. If
the question names several companies or year pairs, call it once for each. If it gives a single
year (for example "what changed in fiscal 2025"), compare it with the year before it.

Report the added, removed and materially changed risks that the tool returned. Give each one with
its description and its passage ids, and add a short quote when it helps. Never add a risk the
tool did not return. If the tool returns ok true with no differences, say that no material
differences were found. If it returns an error (ok false), such as a filing that is not
available, begin your answer with "Comparison not possible:", repeat the error as written
including the years it says are available, and do not guess or substitute other years yourself.
Never say that no differences were found when the tool returned an error."""


def build_risk_diff_agent(llm=None):
    llm = llm or get_tool_llm()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run_risk_diff_agent(question: str, llm=None) -> str:
    agent = build_risk_diff_agent(llm)
    result = agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 25})
    return result["messages"][-1].content

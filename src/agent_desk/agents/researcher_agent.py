"""Narrative researcher: a LangGraph ReAct agent over the filing-search tool only."""

from langgraph.prebuilt import create_react_agent

from ..llm.llm import get_tool_llm
from ..tools.rag_tools import search_filing

TOOLS = [search_filing]

SYSTEM_PROMPT = """You are the narrative filings researcher.
Answer using only the filing-search tool.
Use search_filing for narrative questions about 10-K sections, risk factors, business model,
strategy, legal proceedings, liquidity and other textual content.
Do not use the EDGAR Company Facts tools. Do not answer numeric-company-fact questions with
this agent.
Call tools one at a time. Cite the Item and fiscal year for every filing passage you use.
If the question mentions a company or year, pass them to search_filing. Only claim a filing was
searched if the tool returned passages for it."""


def build_researcher(llm=None):
    llm = llm or get_tool_llm()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run_researcher(question: str, llm=None) -> str:
    agent = build_researcher(llm)
    result = agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 25})
    return result["messages"][-1].content

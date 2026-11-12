"""Quant specialist: EDGAR Company Facts tools plus a restricted Python calculator."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from ..llm.llm import get_tool_llm
from ..tools.edgar_tools import get_company_fact, list_company_facts, list_edgar_companies
from ..tools.sandbox_tools import run_python

TOOLS = [list_edgar_companies, list_company_facts, get_company_fact, run_python]

SYSTEM_PROMPT = """You are the quantitative specialist for a filings research assistant.
Answer using only the SEC Company Facts tools and the run_python calculator.
Always start by calling list_edgar_companies and use the cik it returns for the company. Never
use a CIK from memory. Then use list_company_facts to inspect available facts, and
get_company_fact to retrieve the exact reported value for a company, metric and year.
Cite the fiscal year, form, period and unit for every reported number. Show any ratio you compute,
and do not guess when data is absent. If a tool returns the same error twice, stop and report
that error instead of retrying.

Calculations: never do arithmetic yourself. For every ratio, margin, percentage, difference,
growth rate, sum or average, call run_python with the exact values returned by get_company_fact,
print each result, and report the printed value. Show the formula and the inputs next to each
result. If run_python returns an error, fix the code and call it again.

How the data works:
- Each 10-K contains the current year plus prior years (3 years for income statement and cash
  flow items, 2 for balance sheet items). get_company_fact returns each year as a separate row,
  newest first. To get several years, call it once per metric with fiscal_year set to the
  NEWEST year requested (never the oldest) and read the older years from the same rows.
- A row's fiscal_year is the year of the filing, not the year the value covers. Use the row's
  start and end dates to identify its period, and pair numerator and denominator rows that
  have the same period.
- latest_annual_fiscal_year in each result is the newest year available; earlier years exist
  too. If it is older than the year you need, that metric's tag was retired: find the current
  tag with list_company_facts and a search_term (for example Revenues versus
  RevenueFromContractWithCustomerExcludingAssessedTax).
- If the requested year does not exist yet, say so and report the latest year that does."""


def build_quant_agent(llm=None):
    llm = llm or get_tool_llm()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)

#"Alphabet's operating margin for fiscal years 2023, 2024 and 2025 is as follows:\n\n* 
# 2023: Operating income of $84,293,000,000 and revenue of $307,394,000,000, resulting in an operating margin of 27.4% (fiscal year 2023, form 10-K, period 2023-01-01 to 2023-12-31).\n* 
# 2024: Operating income of $112,390,000,000 and revenue of $350,018,000,000, resulting in an operating margin of 32.1% (fiscal year 2024, form 10-K, period 2024-01-01 to 2024-12-31).\n* 
# 2025: Operating income of $129,039,000,000 and revenue of $402,836,000,000, resulting in an operating margin of 32.0% (fiscal year 2025, form 10-K, period 2025-01-01 to 2025-12-31).\n\nNote: The operating margin is calculated by dividing the operating income by the revenue for each fiscal year."
def run_quant_agent(question: str, llm=None) -> str:
    agent = build_quant_agent(llm)
    result = agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 25})
    return result["messages"][-1].content

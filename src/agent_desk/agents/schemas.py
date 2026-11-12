"""Typed hand-off messages between the research lead and the specialists (L11).

Lead -> specialist:   ResearchTask, QuantTask, RiskDiffTask   (sent with LangGraph Send)
Specialist -> lead:   ResearchFindings, QuantReport, RiskDiffReport
The `*Out` models are what a ReAct specialist must finish with (response_format); code then
validates every id against what the tools actually returned before the report is accepted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompanyYears(BaseModel):
    ticker: str
    fiscal_years: list[int]


class PlanDraft(BaseModel):
    """What the research lead (LLM) proposes; code fills in and validates the years."""
    goal: str = Field(description="One sentence: what the note must answer")
    companies: list[str] = Field(description="Covered tickers, e.g. ['MSFT', 'GOOGL']")
    n_years: int = Field(3, description="How many latest fiscal years per company (1-5)")
    metrics: list[str] = Field(description="Metric names from the allowed list")
    risk_diff: bool = Field(description="True if the question asks about risk factors / what changed in them")
    what_changed_since_last_note: bool = Field(False, description="True if the analyst asks what changed since the last note")
    sub_questions: list[str] = Field(description="3-6 concrete sub-questions")
    outline: list[str] = Field(description="Section headings for the note")


class Plan(BaseModel):
    goal: str
    companies: list[str]
    years: list[CompanyYears]
    metrics: list[str]
    risk_diff: bool
    what_changed_since_last_note: bool = False
    sub_questions: list[str]
    outline: list[str]
    token_budget: int
    version: int = 1
    notes: list[str] = []

    def years_for(self, ticker: str) -> list[int]:
        return next((cy.fiscal_years for cy in self.years if cy.ticker == ticker), [])


class ResearchTask(BaseModel):
    ticker: str
    fiscal_years: list[int]
    sub_questions: list[str]
    question: str = ""  # the analyst's own words (untrusted; used as a search query)


class Finding(BaseModel):
    claim: str = Field(description="One sentence stating what the filing says")
    passage_ids: list[str] = Field(description="passage_id values exactly as returned by search")


class ResearchFindingsOut(BaseModel):
    findings: list[Finding] = Field(description="2-6 findings, each with passage_ids")
    gaps: list[str] = Field(default=[], description="Sub-questions the filings did not answer")


class ResearchFindings(BaseModel):
    ticker: str
    findings: list[Finding]
    gaps: list[str] = []
    dropped: list[str] = []  # findings removed by validation (bad or unseen passage ids)


class QuantTask(BaseModel):
    ticker: str
    fiscal_years: list[int]
    metrics: list[str]


class QuantOut(BaseModel):
    fact_ids: list[str] = Field(description="fact_id of every fact the note should use")
    notes: str = Field(default="", description="Anything missing or unusual (e.g. a year not available)")


class QuantReport(BaseModel):
    ticker: str
    fact_ids: list[str]
    missing: list[str] = []  # "metric/FYyyyy: reason"
    available_years: list[int] = []
    filled_by_code: list[str] = []  # metric-years the agent missed and code fetched


class RiskDiffTask(BaseModel):
    ticker: str
    year_from: int
    year_to: int


class RiskPoint(BaseModel):
    summary: str
    passage_ids: list[str]


class RiskDiffOut(BaseModel):
    added: list[RiskPoint] = []
    removed: list[RiskPoint] = []
    changed: list[RiskPoint] = []


class RiskDiffReport(BaseModel):
    ticker: str
    year_from: int
    year_to: int
    added: list[RiskPoint] = []
    removed: list[RiskPoint] = []
    changed: list[RiskPoint] = []
    counts: dict[str, int] = {}
    error: str = ""
    available_years: list[int] = []

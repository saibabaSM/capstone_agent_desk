"""Metric catalog: which XBRL concepts make each metric, and how derived ratios are computed.

The model only chooses *which* metrics a note needs. This module decides the concept, the fiscal
year, and the arithmetic. Derived metrics are computed by code generated here and executed in the
python sandbox (no network, time limit); the result becomes a derived fact whose `inputs` list the
reported fact IDs, so the critic can recompute it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..companies import Company
from ..sandbox.python_sandbox import run_code
from .xbrl import Fact, annual_observations, load_companyfacts, make_fact


@dataclass(frozen=True)
class BaseMetric:
    name: str
    concepts: tuple[str, ...]
    pick: str = "first"  # first | max  (max: the total among several revenue tags)
    instant: bool = False
    unit: str = "USD"


@dataclass(frozen=True)
class DerivedMetric:
    name: str
    inputs: tuple[str, ...]  # base metric names; "prev:" prefix = prior fiscal year
    expression: str  # Python over the input names
    unit: str  # "%" or "USD"
    description: str


BASE_METRICS: dict[str, BaseMetric] = {m.name: m for m in (
    BaseMetric("revenue", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                           "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"), pick="max"),
    BaseMetric("operating_income", ("OperatingIncomeLoss",)),
    BaseMetric("net_income", ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss")),
    BaseMetric("gross_profit", ("GrossProfit",)),
    BaseMetric("cost_of_revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
    BaseMetric("rd_expense", ("ResearchAndDevelopmentExpense",
                              "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost")),
    BaseMetric("operating_cash_flow", ("NetCashProvidedByUsedInOperatingActivities",)),
    BaseMetric("capex", ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")),
    BaseMetric("diluted_eps", ("EarningsPerShareDiluted",), unit="USD/shares"),
)}

DERIVED_METRICS: dict[str, DerivedMetric] = {m.name: m for m in (
    DerivedMetric("operating_margin", ("operating_income", "revenue"),
                  "operating_income / revenue * 100", "%", "operating income / revenue"),
    DerivedMetric("net_margin", ("net_income", "revenue"), "net_income / revenue * 100", "%",
                  "net income / revenue"),
    DerivedMetric("gross_margin", ("gross_profit", "revenue"), "gross_profit / revenue * 100", "%",
                  "gross profit / revenue"),
    DerivedMetric("rd_intensity", ("rd_expense", "revenue"), "rd_expense / revenue * 100", "%",
                  "R&D expense / revenue"),
    DerivedMetric("free_cash_flow", ("operating_cash_flow", "capex"), "operating_cash_flow - capex",
                  "USD", "operating cash flow - capital expenditure"),
    DerivedMetric("revenue_growth", ("revenue", "prev:revenue"), "(revenue / prev_revenue - 1) * 100",
                  "%", "revenue / prior-year revenue - 1"),
)}

ALL_METRICS = sorted([*BASE_METRICS, *DERIVED_METRICS])


def base_fact(company: Company, metric: str, fiscal_year: int) -> tuple[Fact | None, str]:
    """(fact, "") or (None, reason). The reason names the years that ARE available."""
    spec = BASE_METRICS.get(metric)
    if spec is None:
        return None, f"unknown base metric '{metric}'"
    data = load_companyfacts(company)
    if data is None:
        return None, f"no cached Company Facts for {company.ticker}; run sync-data"
    found: list[tuple[str, dict[str, Any]]] = []
    years: set[int] = set()
    for concept in spec.concepts:
        obs = annual_observations(data, concept, instant=spec.instant)
        years |= set(obs)
        if fiscal_year in obs:
            found.append((concept, obs[fiscal_year]))
            if spec.pick == "first":
                break
    if not found:
        # gross profit is not tagged by every company: fall back to revenue - cost of revenue
        if metric == "gross_profit":
            return _gross_profit_fallback(company, fiscal_year)
        return None, (f"{company.ticker} has no {metric} for FY{fiscal_year}; "
                      f"available fiscal years: {sorted(years)[-6:]}")
    concept, obs = max(found, key=lambda item: float(item[1]["val"])) if spec.pick == "max" else found[0]
    return make_fact(company, metric, concept, obs), ""


def _gross_profit_fallback(company: Company, fiscal_year: int) -> tuple[Fact | None, str]:
    revenue, why = base_fact(company, "revenue", fiscal_year)
    cost, why2 = base_fact(company, "cost_of_revenue", fiscal_year)
    if not revenue or not cost:
        return None, why or why2
    return _derive(company, "gross_profit", fiscal_year, "revenue - cost_of_revenue", "USD",
                   {"revenue": revenue, "cost_of_revenue": cost})


def _derive(company: Company, metric: str, fiscal_year: int, expression: str, unit: str,
            inputs: dict[str, Fact]) -> tuple[Fact | None, str]:
    """Run the arithmetic in the sandbox and wrap the result as a derived fact."""
    lines = [f"{name} = {fact.value!r}" for name, fact in inputs.items()]
    lines.append(f"print(repr({expression}))")
    result = run_code("\n".join(lines))
    if not result["ok"]:
        return None, f"sandbox error computing {metric}: {result['error']}"
    try:
        value = float(result["output"].strip())
    except ValueError:
        return None, f"sandbox returned a non-number for {metric}: {result['output'][:80]}"
    current = [f for f in inputs.values() if f.fiscal_year == fiscal_year] or list(inputs.values())
    anchor = current[0]
    return Fact(
        fact_id=f"{company.ticker}/{metric}/FY{fiscal_year}/calc",
        ticker=company.ticker,
        metric=metric,
        concept=f"derived:{metric}",
        label=metric.replace("_", " "),
        value=value,
        unit=unit,
        fiscal_year=fiscal_year,
        period_start=anchor.period_start,
        period_end=anchor.period_end,
        accession=anchor.accession,
        form=anchor.form,
        filed=anchor.filed,
        kind="derived",
        formula=expression,
        inputs=[f.fact_id for f in inputs.values()],
    ), ""


def get_metric(company: Company, metric: str, fiscal_year: int) -> tuple[list[Fact], str]:
    """All facts needed for one metric-year: inputs first, then the derived fact itself.

    Returns (facts, error). For a base metric the list has one fact.
    """
    if metric in BASE_METRICS:
        fact, why = base_fact(company, metric, fiscal_year)
        return ([fact] if fact else []), why
    spec = DERIVED_METRICS.get(metric)
    if spec is None:
        return [], f"unknown metric '{metric}'. Known: {', '.join(ALL_METRICS)}"
    inputs: dict[str, Fact] = {}
    for name in spec.inputs:
        year = fiscal_year - 1 if name.startswith("prev:") else fiscal_year
        base = name.removeprefix("prev:")
        fact, why = base_fact(company, base, year)
        if fact is None:
            return list(inputs.values()), why
        inputs[name.replace("prev:", "prev_")] = fact
    derived, why = _derive(company, metric, fiscal_year, spec.expression, spec.unit, inputs)
    if derived is None:
        return list(inputs.values()), why
    return [*inputs.values(), derived], ""


def recompute(derived: dict[str, Any], registry: dict[str, dict[str, Any]]) -> float | None:
    """Recompute a derived fact from the registry (used by the critic). None if inputs missing."""
    metric = derived["metric"]
    spec = DERIVED_METRICS.get(metric)
    if spec is None and metric == "gross_profit":
        names, expression = ["revenue", "cost_of_revenue"], "revenue - cost_of_revenue"
    elif spec is None:
        return None
    else:
        names, expression = [n.replace("prev:", "prev_") for n in spec.inputs], spec.expression
    if len(derived.get("inputs", [])) != len(names):
        return None
    values = {}
    for name, fact_id in zip(names, derived["inputs"]):
        if fact_id not in registry:
            return None
        values[name] = registry[fact_id]["value"]
    code = "\n".join(f"{k} = {v!r}" for k, v in values.items()) + f"\nprint(repr({expression}))"
    result = run_code(code)
    return float(result["output"]) if result["ok"] else None


def latest_years(company: Company, n: int, metric: str = "revenue") -> list[int]:
    """The company's latest n fiscal years that have a 10-K value for `metric`."""
    spec = BASE_METRICS.get(metric) or BASE_METRICS["revenue"]
    data = load_companyfacts(company)
    if not data:
        return []
    years: set[int] = set()
    for concept in spec.concepts:
        years |= set(annual_observations(data, concept))
    return sorted(years)[-n:]

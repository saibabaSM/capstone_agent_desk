"""Resolve reported XBRL facts for a company and fiscal year, with a traceable fact ID.

Signature challenge A ("numbers you can trust") rests on this module. Rules, all in code:

* Fiscal year comes from the PERIOD END date (companies.fiscal_year_label), never from the XBRL
  `fy` field: `fy` is the fiscal year of the *filing*, so a FY2025 10-K also carries FY2024 and
  FY2023 comparatives tagged fy=2025. Using `fy` silently shifts numbers by a year.
* Flow concepts (revenue, income, cash flow) must cover a full year: 350-380 days.
* When the same period is reported in several filings, the value from the 10-K whose own fiscal
  year is that year (as originally reported) wins; later restatements are recorded as `restated`.
* Units are kept raw (USD, USD/shares). Display scaling happens only in format_value().

Fact ID format:  <TICKER>/<concept>/FY<year>/<accession>   e.g. MSFT/OperatingIncomeLoss/FY2025/0000950170-25-100235
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..companies import Company, fiscal_year_label
from ..config import data_dir


@dataclass
class Fact:
    fact_id: str
    ticker: str
    metric: str
    concept: str
    label: str
    value: float
    unit: str
    fiscal_year: int
    period_start: str | None
    period_end: str
    accession: str
    form: str
    filed: str
    restated: bool = False
    kind: str = "reported"  # reported | derived
    formula: str = ""
    inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def facts_path(company: Company) -> Path:
    return data_dir() / "edgar_cache" / "companyfacts" / f"CIK{company.cik}.json"


@lru_cache(maxsize=32)
def _load(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_companyfacts(company: Company) -> dict | None:
    path = facts_path(company)
    if not path.exists():
        return None
    return _load(str(path), path.stat().st_mtime)


def _days(start: str | None, end: str) -> int | None:
    if not start:
        return None
    return (date.fromisoformat(end[:10]) - date.fromisoformat(start[:10])).days


def annual_observations(company_facts: dict, concept: str, taxonomy: str = "us-gaap",
                        instant: bool = False) -> dict[int, dict[str, Any]]:
    """{fiscal_year: best observation} for one concept, using the rules in the module docstring."""
    node = company_facts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return {}
    by_year: dict[int, list[dict[str, Any]]] = {}
    for unit, rows in node.get("units", {}).items():
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            end = row.get("end")
            if not end:
                continue
            if not instant:
                length = _days(row.get("start"), end)
                if length is None or not 350 <= length <= 380:
                    continue
            fy = fiscal_year_label(end)
            by_year.setdefault(fy, []).append({**row, "unit": unit, "label": node.get("label", concept)})
    best: dict[int, dict[str, Any]] = {}
    for fy, rows in by_year.items():
        rows.sort(key=lambda r: r.get("filed", ""))
        original = [r for r in rows if r.get("fy") == fy and r.get("form") == "10-K"]
        chosen = original[0] if original else rows[0]
        values = {r["val"] for r in rows}
        best[fy] = {**chosen, "restated": len(values) > 1}
    return best


def make_fact(company: Company, metric: str, concept: str, obs: dict[str, Any]) -> Fact:
    fy = fiscal_year_label(obs["end"])
    return Fact(
        fact_id=f"{company.ticker}/{concept}/FY{fy}/{obs['accn']}",
        ticker=company.ticker,
        metric=metric,
        concept=concept,
        label=obs.get("label", concept),
        value=float(obs["val"]),
        unit=obs["unit"],
        fiscal_year=fy,
        period_start=obs.get("start"),
        period_end=obs["end"],
        accession=obs["accn"],
        form=obs.get("form", ""),
        filed=obs.get("filed", ""),
        restated=bool(obs.get("restated")),
    )


def available_fiscal_years(company: Company, concept_candidates: list[str]) -> list[int]:
    data = load_companyfacts(company)
    if not data:
        return []
    years: set[int] = set()
    for concept in concept_candidates:
        years |= set(annual_observations(data, concept))
    return sorted(years)


def format_value(value: float, unit: str) -> str:
    """The ONE way numbers are displayed. The writer copies these strings; the critic parses them."""
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "USD/shares":
        return f"${value:,.2f}"
    if unit == "USD":
        magnitude = abs(value)
        sign = "-" if value < 0 else ""
        if magnitude >= 1e9:
            return f"{sign}${magnitude / 1e9:,.1f}B"
        if magnitude >= 1e6:
            return f"{sign}${magnitude / 1e6:,.1f}M"
        return f"{sign}${magnitude:,.0f}"
    return f"{value:,.2f} {unit}"

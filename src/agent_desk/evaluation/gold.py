"""Gold values from SEC's companyconcept endpoint, independent of the agent's code path.

    gold_value("MSFT", "operating_margin", 2025) -> 45.62...

Independence: the agent reads companyfacts and picks concepts by rule (finance/metrics.py); the
gold uses a hand-chosen concept per company (eval/gold/concepts.yaml), a different endpoint, and
plain-Python arithmetic here. The fiscal-year convention (year of the period end; 350-380 day
duration; value as first reported in a 10-K) is the DEFINITION both sides share.
"""

from __future__ import annotations

import json
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

import requests

from ..companies import BY_TICKER, fiscal_year_label
from ..config import PROJECT_ROOT, data_dir, load_yaml
from ..data_sync import user_agent

CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
CONCEPTS = PROJECT_ROOT / "eval" / "gold" / "concepts.yaml"


@lru_cache(maxsize=1)
def concept_map() -> dict:
    return load_yaml(CONCEPTS)


def gold_concept(ticker: str, metric: str) -> str:
    entry = concept_map()[metric]
    return entry.get(ticker) or entry.get("default")


@lru_cache(maxsize=256)
def _concept_json(cik: str, concept: str) -> dict:
    path = data_dir() / "gold_cache" / f"{cik}-{concept}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    time.sleep(0.25)
    resp = requests.get(CONCEPT_URL.format(cik=cik, concept=concept), headers={"User-Agent": user_agent()}, timeout=60)
    data = resp.json() if resp.status_code == 200 else {"units": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def base_value(ticker: str, metric: str, fy: int) -> float | None:
    company = BY_TICKER[ticker]
    data = _concept_json(company.cik, gold_concept(ticker, metric))
    best = None
    for unit, rows in data.get("units", {}).items():
        for r in rows:
            if r.get("form") != "10-K" or not r.get("start") or not r.get("end"):
                continue
            days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
            if not 350 <= days <= 380 or fiscal_year_label(r["end"]) != fy:
                continue
            if best is None or r["filed"] < best["filed"]:
                best = r
    return float(best["val"]) if best else None


def gold_value(ticker: str, metric: str, fy: int) -> float | None:
    b = lambda m, y=fy: base_value(ticker, m, y)  # noqa: E731
    if metric in concept_map():
        return b(metric)
    ratio = {"operating_margin": ("operating_income", "revenue"), "net_margin": ("net_income", "revenue"),
             "gross_margin": ("gross_profit", "revenue"), "rd_intensity": ("rd_expense", "revenue")}
    if metric in ratio:
        num, den = b(ratio[metric][0]), b(ratio[metric][1])
        return None if num is None or not den else num / den * 100
    if metric == "free_cash_flow":
        ocf, capex = b("operating_cash_flow"), b("capex")
        return None if ocf is None or capex is None else ocf - capex
    if metric == "revenue_growth":
        cur, prev = b("revenue"), b("revenue", fy - 1)
        return None if cur is None or not prev else (cur / prev - 1) * 100
    raise KeyError(metric)


def gold_latest_years(ticker: str, n: int) -> list[int]:
    company = BY_TICKER[ticker]
    data = _concept_json(company.cik, gold_concept(ticker, "revenue"))
    years = set()
    for rows in data.get("units", {}).values():
        for r in rows:
            if r.get("form") == "10-K" and r.get("start") and r.get("end"):
                days = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
                if 350 <= days <= 380:
                    years.add(fiscal_year_label(r["end"]))
    return sorted(years)[-n:]


def gold_unit(metric: str) -> str:
    return "%" if metric in ("operating_margin", "net_margin", "gross_margin", "rd_intensity", "revenue_growth") else "USD"


def save_path(split: str) -> Path:
    return PROJECT_ROOT / "eval" / split / "cases.jsonl"

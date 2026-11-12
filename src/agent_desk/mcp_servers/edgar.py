"""edgar MCP server: filings index, 10-K sections and XBRL facts. Cached; read-only.

Tools
  list_filings(company)                              [read]  10-Ks available locally
  get_section(company, fiscal_year, item)            [read]  text of one Item of a 10-K
  list_concepts(company, search)                     [read]  XBRL concepts with their latest year
  get_xbrl_fact(company, concept, fiscal_year)       [read]  one reported fact with fact_id
  get_metric(company, metric, fiscal_years)          [read]  catalog metric (+ ratios computed in the sandbox)

Everything reads the local EDGAR cache. On a Company Facts cache miss the server fetches from
data.sec.gov once (rate-limited, descriptive User-Agent) and caches it; each result says
cache=hit|miss so the dashboard can show the hit rate.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

from fastmcp import FastMCP
from pydantic import BaseModel

from ..companies import COMPANIES, resolve
from ..config import data_dir
from ..data_sync import COMPANYFACTS_URL, EdgarHTTP
from ..finance.metrics import ALL_METRICS, get_metric as _get_metric
from ..finance.xbrl import annual_observations, facts_path, format_value, load_companyfacts, make_fact
from ..rag.structured_parser import parse_filing
from .common import READ, READ_EXTERNAL, run_server

mcp = FastMCP("edgar", instructions="SEC EDGAR filings and XBRL facts (cached). Values are raw USD.")
_http_lock = threading.Lock()
_http: EdgarHTTP | None = None


class FactOut(BaseModel):
    fact_id: str
    ticker: str
    metric: str
    concept: str
    value: float
    display: str
    unit: str
    fiscal_year: int
    period_start: Optional[str]
    period_end: str
    accession: str
    filed: str
    kind: str
    formula: str = ""
    inputs: list[str] = []
    restated: bool = False


class MetricResult(BaseModel):
    ok: bool
    facts: list[FactOut] = []
    errors: list[str] = []
    cache: str = "hit"
    available_metrics: list[str] = []


class FilingInfo(BaseModel):
    fiscal_year: int
    accession: str
    report_date: str
    filing_date: str
    source_url: str


class FilingsResult(BaseModel):
    ok: bool
    ticker: str = ""
    fiscal_year_end: str = ""
    filings: list[FilingInfo] = []
    error: str = ""


class SectionResult(BaseModel):
    ok: bool
    ticker: str = ""
    fiscal_year: int = 0
    item: str = ""
    text: str = ""
    truncated: bool = False
    error: str = ""


class ConceptInfo(BaseModel):
    concept: str
    label: str
    latest_fiscal_year: Optional[int]


class ConceptsResult(BaseModel):
    ok: bool
    concepts: list[ConceptInfo] = []
    error: str = ""


def _ensure_facts(company) -> str:
    """Make sure Company Facts is cached. Returns 'hit' or 'miss' (fetched now)."""
    global _http
    path = facts_path(company)
    if path.exists():
        return "hit"
    with _http_lock:
        if path.exists():
            return "hit"
        _http = _http or EdgarHTTP()
        data = _http.get(COMPANYFACTS_URL.format(cik=company.cik)).json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(0.2)
    return "miss"


def _fact_out(f) -> FactOut:
    return FactOut(display=format_value(f.value, f.unit), **{k: v for k, v in f.to_dict().items()
                                                             if k in FactOut.model_fields and k != "display"})


@mcp.tool(annotations=READ)
def list_filings(company: str) -> FilingsResult:
    """List the company's 10-K filings available locally (fiscal year, accession, dates)."""
    comp = resolve(company)
    if comp is None:
        return FilingsResult(ok=False, error=f"'{company}' is not covered. Covered: {', '.join(c.ticker for c in COMPANIES)}")
    rows = []
    for meta_path in sorted((data_dir() / "filings" / comp.key).glob("*/filing.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append(FilingInfo(fiscal_year=m["fiscal_year"], accession=m["accession"], report_date=m["report_date"],
                               filing_date=m["filing_date"], source_url=m["source_url"]))
    if not rows:
        return FilingsResult(ok=False, ticker=comp.ticker, error="no filings downloaded; run sync-data")
    return FilingsResult(ok=True, ticker=comp.ticker, fiscal_year_end=comp.fy_end, filings=rows)


@mcp.tool(annotations=READ)
def get_section(company: str, fiscal_year: int, item: str, max_chars: int = 6000) -> SectionResult:
    """Return the text of one Item (e.g. "1A", "7") of the company's 10-K for a fiscal year."""
    comp = resolve(company)
    if comp is None:
        return SectionResult(ok=False, error=f"'{company}' is not covered")
    folder = data_dir() / "filings" / comp.key / str(fiscal_year)
    files = sorted(folder.glob("*.htm*")) if folder.is_dir() else []
    if not files:
        have = sorted(p.name for p in (data_dir() / "filings" / comp.key).glob("[0-9]*"))
        return SectionResult(ok=False, error=f"no 10-K for {comp.ticker} FY{fiscal_year}; available: {have}")
    want = item.upper().replace("ITEM", "").strip()
    text = "\n\n".join(b.text for b in parse_filing(files[0]) if b.item == want)
    if not text:
        return SectionResult(ok=False, error=f"Item {want} not found in {files[0].name}")
    limit = max(500, min(max_chars, 20000))
    return SectionResult(ok=True, ticker=comp.ticker, fiscal_year=fiscal_year, item=want, text=text[:limit],
                         truncated=len(text) > limit)


@mcp.tool(annotations=READ_EXTERNAL)
def list_concepts(company: str, search: str = "", limit: int = 30) -> ConceptsResult:
    """List us-gaap XBRL concepts for a company whose name or label contains `search`."""
    comp = resolve(company)
    if comp is None:
        return ConceptsResult(ok=False, error=f"'{company}' is not covered")
    _ensure_facts(comp)
    data = load_companyfacts(comp) or {}
    term = search.lower().strip()
    out = []
    for concept, node in data.get("facts", {}).get("us-gaap", {}).items():
        label = node.get("label") or concept
        if term and term not in f"{concept} {label}".lower():
            continue
        years = annual_observations(data, concept) or annual_observations(data, concept, instant=True)
        out.append(ConceptInfo(concept=concept, label=label, latest_fiscal_year=max(years) if years else None))
    out.sort(key=lambda c: -(c.latest_fiscal_year or 0))
    return ConceptsResult(ok=True, concepts=out[:max(1, min(limit, 100))])


@mcp.tool(annotations=READ_EXTERNAL)
def get_xbrl_fact(company: str, concept: str, fiscal_year: int) -> MetricResult:
    """One reported us-gaap fact for a fiscal year (aligned by period end date), with its fact_id."""
    comp = resolve(company)
    if comp is None:
        return MetricResult(ok=False, errors=[f"'{company}' is not covered"])
    cache = _ensure_facts(comp)
    data = load_companyfacts(comp) or {}
    obs = annual_observations(data, concept) or annual_observations(data, concept, instant=True)
    if fiscal_year not in obs:
        return MetricResult(ok=False, cache=cache, errors=[f"{concept} has no FY{fiscal_year} value; years with data: {sorted(obs)[-6:]}"])
    return MetricResult(ok=True, cache=cache, facts=[_fact_out(make_fact(comp, concept, concept, obs[fiscal_year]))])


@mcp.tool(annotations=READ_EXTERNAL)
def get_metric(company: str, metric: str, fiscal_years: list[int]) -> MetricResult:
    """Catalog metric for each fiscal year, e.g. metric="operating_margin", fiscal_years=[2024, 2025].

    Ratios are computed in the python sandbox from reported facts; derived facts list their inputs.
    Use the returned fact_id and display string verbatim.
    """
    comp = resolve(company)
    if comp is None:
        return MetricResult(ok=False, errors=[f"'{company}' is not covered"])
    if metric not in ALL_METRICS:
        return MetricResult(ok=False, errors=[f"unknown metric '{metric}'"], available_metrics=ALL_METRICS)
    if not fiscal_years or len(fiscal_years) > 6:
        return MetricResult(ok=False, errors=["give 1 to 6 fiscal years"])
    cache = _ensure_facts(comp)
    facts, errors, seen = [], [], set()
    for year in fiscal_years:
        found, why = _get_metric(comp, metric, int(year))
        if why:
            errors.append(why)
        for f in found:
            if f.fact_id not in seen:
                seen.add(f.fact_id)
                facts.append(_fact_out(f))
    return MetricResult(ok=bool(facts), facts=facts, errors=errors, cache=cache)


if __name__ == "__main__":
    run_server(mcp, 8102)

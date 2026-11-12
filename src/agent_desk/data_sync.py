"""Download the coverage universe from SEC EDGAR into the local cache (run once, then offline).

    python -m agent_desk sync-data            # latest 3 10-Ks + companyfacts for all 12 companies
    python -m agent_desk sync-data --tickers MSFT GOOGL --filings 3

EDGAR asks automated clients to send a descriptive User-Agent and stay under 10 requests per
second. SEC_USER_AGENT from .env is used when set; otherwise a generic client name. Requests are
spaced by REQUEST_DELAY_SECONDS and every response is cached, so a second run makes no requests.

Layout written:
    data/filings/<key>/<fiscal_year>/<primary document>.htm   10-K HTML
    data/filings/<key>/<fiscal_year>/filing.json                 accession, dates, URL
    data/edgar_cache/companyfacts/CIK##########.json             XBRL Company Facts
    data/edgar_cache/submissions/CIK##########.json              filing index
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from .companies import BY_TICKER, COMPANIES, Company, fiscal_year_label
from .config import data_dir

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"
REQUEST_DELAY_SECONDS = 0.2  # 5 requests per second, half of SEC's limit
GENERIC_AGENT = "AnalystDesk capstone research client (educational use)"


def user_agent() -> str:
    value = os.getenv("SEC_USER_AGENT", "").strip()
    if not value or "example.com" in value.lower():
        return GENERIC_AGENT
    return value


class EdgarHTTP:
    """Rate-limited GET with a shared session."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0
        self.requests_made = 0

    def get(self, url: str) -> requests.Response:
        wait = REQUEST_DELAY_SECONDS - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.requests_made += 1
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response


def _cached_json(path: Path, url: str, http: EdgarHTTP, refresh: bool = False) -> dict[str, Any]:
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = http.get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)
    return data


def ten_k_filings(company: Company, http: EdgarHTTP, refresh: bool = False) -> list[dict[str, str]]:
    """All original 10-K filings (not 10-K/A), newest first, from the recent submissions block."""
    path = data_dir() / "edgar_cache" / "submissions" / f"CIK{company.cik}.json"
    data = _cached_json(path, SUBMISSIONS_URL.format(cik=company.cik), http, refresh)
    recent = data["filings"]["recent"]
    keys = ("form", "accessionNumber", "primaryDocument", "reportDate", "filingDate")
    rows = [dict(zip(keys, row)) for row in zip(*(recent[k] for k in keys))]
    rows = [r for r in rows if r["form"] == "10-K" and r["reportDate"]]
    rows.sort(key=lambda r: r["reportDate"], reverse=True)
    return rows


def sync_company(company: Company, n_filings: int, http: EdgarHTTP, refresh: bool = False) -> dict:
    report: dict[str, Any] = {"ticker": company.ticker, "filings": []}
    facts_path = data_dir() / "edgar_cache" / "companyfacts" / f"CIK{company.cik}.json"
    _cached_json(facts_path, COMPANYFACTS_URL.format(cik=company.cik), http, refresh)
    report["companyfacts"] = str(facts_path)
    for filing in ten_k_filings(company, http, refresh)[:n_filings]:
        fy = fiscal_year_label(filing["reportDate"])
        folder = data_dir() / "filings" / company.key / str(fy)
        target = folder / Path(filing["primaryDocument"]).name
        if not target.exists():
            url = ARCHIVE_URL.format(
                cik_int=int(company.cik),
                acc=filing["accessionNumber"].replace("-", ""),
                doc=filing["primaryDocument"],
            )
            folder.mkdir(parents=True, exist_ok=True)
            target.write_bytes(http.get(url).content)
        meta = {
            "ticker": company.ticker,
            "company": company.key,
            "fiscal_year": fy,
            "accession": filing["accessionNumber"],
            "report_date": filing["reportDate"],
            "filing_date": filing["filingDate"],
            "document": target.name,
            "source_url": ARCHIVE_URL.format(
                cik_int=int(company.cik),
                acc=filing["accessionNumber"].replace("-", ""),
                doc=filing["primaryDocument"],
            ),
        }
        (folder / "filing.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report["filings"].append({"fy": fy, "file": target.name})
    return report


def sync(tickers: list[str] | None = None, n_filings: int = 3, refresh: bool = False) -> list[dict]:
    http = EdgarHTTP()
    companies = [BY_TICKER[t.upper()] for t in tickers] if tickers else list(COMPANIES)
    reports = []
    for company in companies:
        try:
            reports.append(sync_company(company, n_filings, http, refresh))
        except requests.RequestException as exc:
            reports.append({"ticker": company.ticker, "error": str(exc)})
        print(json.dumps(reports[-1]))
    print(f"EDGAR requests made: {http.requests_made}")
    return reports

"""Download 10-K filings (HTML) from SEC EDGAR into data/filings/<company>/<year>/."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILINGS_DIR = PROJECT_ROOT / "data" / "filings"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
REQUEST_DELAY_SECONDS = 0.2
TIMEOUT_SECONDS = 30

load_dotenv(PROJECT_ROOT / ".env")


def _user_agent() -> str:
    value = os.getenv("SEC_USER_AGENT", "").strip()
    if not value or "example" in value.lower():
        raise ValueError("SEC_USER_AGENT is missing or still uses the example value.")
    return value


def _get(url: str, session: requests.Session) -> requests.Response:
    time.sleep(REQUEST_DELAY_SECONDS)
    response = session.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response


def _folder_name(company_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", company_name.lower()).strip("_")


def _filings_from_block(block: dict[str, Any]) -> list[dict[str, str]]:
    """Turn the column-oriented SEC filings block into one dict per filing."""
    keys = ("form", "accessionNumber", "primaryDocument", "reportDate", "filingDate")
    columns = [block.get(key, []) for key in keys]
    return [dict(zip(keys, row)) for row in zip(*columns)]


def list_10k_filings(cik: str, session: requests.Session) -> list[dict[str, str]]:
    """Return every 10-K filing for a CIK, including older ones in archive files."""
    padded = str(cik).strip().zfill(10)
    data = _get(SUBMISSIONS_URL.format(name=f"CIK{padded}.json"), session).json()
    filings = _filings_from_block(data["filings"]["recent"])
    for extra in data["filings"].get("files", []):
        older = _get(SUBMISSIONS_URL.format(name=extra["name"]), session).json()
        filings.extend(_filings_from_block(older))
    return [f for f in filings if f["form"] == "10-K" and f["reportDate"]]


def download_filings(
    company_name: str,
    cik: str,
    years: list[int],
    base_dir: Path = FILINGS_DIR,
    session: requests.Session | None = None,
) -> dict[int, dict[str, str]]:
    """Download the 10-K for each fiscal year (year of the period end date).

    Returns {year: {"status": "downloaded" | "skipped" | "not_found" | "error", ...}}.
    """
    own_session = session is None
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": _user_agent()})

    results: dict[int, dict[str, str]] = {}
    try:
        by_year = {int(f["reportDate"][:4]): f for f in list_10k_filings(cik, session)}
        cik_number = str(int(cik))
        for year in years:
            filing = by_year.get(int(year))
            if filing is None:
                results[year] = {"status": "not_found"}
                continue
            document = filing["primaryDocument"]
            target = base_dir / _folder_name(company_name) / str(year) / Path(document).name
            if target.exists():
                results[year] = {"status": "skipped", "path": str(target)}
                continue
            url = ARCHIVE_URL.format(
                cik=cik_number,
                accession=filing["accessionNumber"].replace("-", ""),
                document=document,
            )
            try:
                content = _get(url, session).content
            except requests.RequestException as exc:
                results[year] = {"status": "error", "message": str(exc)}
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            results[year] = {"status": "downloaded", "path": str(target)}
    finally:
        if own_session:
            session.close()
    return results


def download_from_spec(spec: dict[str, Any]) -> dict[int, dict[str, str]]:
    """Accept {"company_name": ..., "cik": ..., "years": [...]}."""
    return download_filings(spec["company_name"], spec["cik"], spec["years"])


# SPECS = [
#     {"company_name": "microsoft", "cik": "0000789019", "years": [2023, 2024, 2025]},
# ]
# SPECS = [
#     {"company_name": "alphabet", "cik": "0001652044", "years": [2023, 2024, 2025]},
# ]
SPECS = [
    {"company_name": "apple", "cik": "0000320193", "years": [2023, 2024]},
]


def main() -> None:
    for spec in SPECS:
        print(spec["company_name"])
        for year, result in download_from_spec(spec).items():
            print(f"  {year}: {result}")


if __name__ == "__main__":
    main()


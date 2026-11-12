"""LangChain tools backed only by the SEC XBRL Company Facts API."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "edgar_cache" / "companyfacts"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_RESULT_LIMIT = 50
MAX_RESULT_LIMIT = 200

#revisit
COMPANIES = {
    "AAPL": {"name": "Apple Inc.", "cik": "0000320193"},
    "MSFT": {"name": "Microsoft Corporation", "cik": "0000789019"},
    "GOOGL": {"name": "Alphabet Inc.", "cik": "0001652044"},
}

load_dotenv(PROJECT_ROOT / ".env")


def _json_result(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_cik(cik: str) -> str | None:
    normalized = cik.strip()
    if not normalized.isdigit() or len(normalized) > 10:
        return None
    return normalized.zfill(10)


def _read_cached_facts(cik: str) -> tuple[dict[str, Any] | None, str | None]:
    cache_path = CACHE_DIR / f"CIK{cik}.json"
    if not cache_path.exists():
        return None, None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Could not read cached Company Facts for CIK {cik}: {exc}"
    if not isinstance(data, dict):
        return None, f"Cached Company Facts for CIK {cik} is not a JSON object."
    return data, None


def _write_cached_facts(cik: str, data: dict[str, Any]) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"CIK{cik}.json"
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=CACHE_DIR,
            delete=False,
            suffix=".tmp",
        ) as temp_file:
            temp_path = temp_file.name
            json.dump(data, temp_file)
        os.replace(temp_path, cache_path)
    except OSError as exc:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"Could not cache Company Facts for CIK {cik}: {exc}"
    return None


def _fetch_company_facts(
    cik: str, force_refresh: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    if not force_refresh:
        cached, cache_error = _read_cached_facts(cik)
        if cache_error:
            return None, cache_error
        if cached is not None:
            return cached, None

    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent or "example.com" in user_agent.lower():
        return None, (
            "SEC_USER_AGENT is missing or still uses the example value. "
            "Set a descriptive value in the project .env file."
        )

    url = COMPANY_FACTS_URL.format(cik=cik)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, f"SEC Company Facts request failed for CIK {cik}: {exc}"

    try:
        data = response.json()
    except ValueError as exc:
        return None, f"SEC returned invalid JSON for CIK {cik}: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("facts"), dict):
        return None, f"SEC response for CIK {cik} does not contain Company Facts."
    response_cik = str(data.get("cik", "")).zfill(10)
    if response_cik != cik:
        return None, f"SEC response CIK did not match requested CIK {cik}."

    cache_error = _write_cached_facts(cik, data)
    if cache_error:
        return None, cache_error
    return data, None


def _validated_limit(limit: int) -> int | None:
    if isinstance(limit, bool) or not 1 <= limit <= MAX_RESULT_LIMIT:
        return None
    return limit


def _latest_annual_fiscal_year(units: dict[str, Any]) -> int | None:
    """Newest fiscal year that has a full-year 10-K row in the given unit entries."""
    years = [
        entry["fy"]
        for entries in units.values()
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("form") == "10-K"
        and entry.get("fp") == "FY"
        and isinstance(entry.get("fy"), int)
    ]
    return max(years, default=None)


@tool
def list_edgar_companies() -> str:
    """List companies in the local directory that can be queried by this project."""
    return _json_result(
        ok=True,
        source="local directory",
        companies=[
            {"ticker": ticker, **company}
            for ticker, company in sorted(COMPANIES.items())
        ],
        note="Company Facts itself does not provide company discovery.",
    )


@tool
def list_company_facts(
    cik: str,
    search_term: str = "",
    taxonomy: str = "",
    limit: int = DEFAULT_RESULT_LIMIT,
    force_refresh: bool = False,
) -> str:
    """List XBRL fact concepts for a company, optionally filtering by name or taxonomy.

    Each concept shows latest_annual_fiscal_year, the newest fiscal year with 10-K data under that
    concept. Companies switch concepts over time, so the same metric (for example revenue) can sit
    under an older concept that stops at an earlier year and a newer one that continues. Prefer the
    concept whose latest_annual_fiscal_year is highest.
    """
    normalized_cik = _normalize_cik(cik)
    result_limit = _validated_limit(limit)
    if normalized_cik is None:
        return _json_result(ok=False, error="cik must contain 1 to 10 digits.")
    if result_limit is None:
        return _json_result(
            ok=False,
            error=f"limit must be between 1 and {MAX_RESULT_LIMIT}.",
        )

    company_facts, error = _fetch_company_facts(normalized_cik, force_refresh)
    if error:
        return _json_result(ok=False, error=error)
    assert company_facts is not None

    requested_taxonomy = taxonomy.strip().lower()
    requested_term = search_term.strip().lower()
    facts: list[dict[str, Any]] = []
    for fact_taxonomy, concepts in company_facts["facts"].items():
        if requested_taxonomy and fact_taxonomy.lower() != requested_taxonomy:
            continue
        if not isinstance(concepts, dict):
            continue
        for concept_name, details in concepts.items():
            if not isinstance(details, dict):
                continue
            label = str(details.get("label", concept_name))
            if requested_term and requested_term not in (
                f"{concept_name} {label}".lower()
            ):
                continue
            units = details.get("units", {})
            facts.append(
                {
                    "taxonomy": fact_taxonomy,
                    "concept": concept_name,
                    "label": label,
                    "units": sorted(units) if isinstance(units, dict) else [],
                    # Newest fiscal year with a full-year 10-K row, or None if the concept has no
                    # annual 10-K data. A low year means the company stopped using this tag.
                    "latest_annual_fiscal_year": (
                        _latest_annual_fiscal_year(units) if isinstance(units, dict) else None
                    ),
                }
            )

    facts.sort(key=lambda fact: (fact["taxonomy"], fact["concept"]))
    return _json_result(
        ok=True,
        cik=normalized_cik,
        entity=company_facts.get("entityName"),
        total_matches=len(facts),
        facts=facts[:result_limit],
    )


@tool
def get_company_fact(
    cik: str,
    concept: str,
    taxonomy: str = "us-gaap",
    fiscal_year: int | None = None,
    unit: str = "",
    limit: int = 20,
    force_refresh: bool = False,
    form: str = "10-K",
) -> str:
    """Get reported values and filing metadata for an XBRL concept and company.

    By default only annual 10-K rows are returned, newest period first. Pass form="" to include
    every form (10-Q quarters too). The result always includes latest_annual_fiscal_year, the
    newest fiscal year with 10-K data for this concept. If fiscal_year has no data the result
    is an error that names that latest year.
    """
    normalized_cik = _normalize_cik(cik)
    result_limit = _validated_limit(limit)
    if normalized_cik is None:
        return _json_result(ok=False, error="cik must contain 1 to 10 digits.")
    if not concept.strip():
        return _json_result(ok=False, error="concept must not be empty.")
    if result_limit is None:
        return _json_result(
            ok=False,
            error=f"limit must be between 1 and {MAX_RESULT_LIMIT}.",
        )

    company_facts, error = _fetch_company_facts(normalized_cik, force_refresh)
    if error:
        return _json_result(ok=False, error=error)
    assert company_facts is not None

    concepts = company_facts["facts"].get(taxonomy, {})
    fact = concepts.get(concept) if isinstance(concepts, dict) else None
    if not isinstance(fact, dict):
        return _json_result(
            ok=False,
            error=f"Concept {taxonomy}:{concept} was not found for CIK {normalized_cik}.",
        )

    available_units = fact.get("units", {})
    if not isinstance(available_units, dict):
        return _json_result(ok=False, error="SEC fact has an invalid units structure.")
    requested_unit = unit.strip()
    selected_units = (
        {requested_unit: available_units[requested_unit]}
        if requested_unit in available_units
        else available_units
    )
    if requested_unit and requested_unit not in available_units:
        return _json_result(
            ok=False,
            error=f"Unit {requested_unit} is unavailable for {taxonomy}:{concept}.",
            available_units=sorted(available_units),
        )

    requested_form = form.strip()
    latest_annual_fiscal_year = _latest_annual_fiscal_year(selected_units)
    observations: list[dict[str, Any]] = []
    for fact_unit, entries in selected_units.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if fiscal_year is not None and entry.get("fy") != fiscal_year:
                continue
            if requested_form and entry.get("form") != requested_form:
                continue
            observations.append(
                {
                    "unit": fact_unit,
                    "value": entry.get("val"),
                    "fiscal_year": entry.get("fy"),
                    "fiscal_period": entry.get("fp"),
                    "form": entry.get("form"),
                    "filed": entry.get("filed"),
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                    "accession_number": entry.get("accn"),
                    "frame": entry.get("frame"),
                }
            )
    if fiscal_year is not None and not observations:
        return _json_result(
            ok=False,
            error=(
                f"No {requested_form or 'SEC'} data for {taxonomy}:{concept} in fiscal year "
                f"{fiscal_year}. This concept's newest 10-K data is for fiscal year "
                f"{latest_annual_fiscal_year}. That does not mean the company has no fiscal "
                f"{fiscal_year} data: companies switch to a different concept for the same metric. "
                f"Search list_company_facts for the metric and use the concept whose "
                f"latest_annual_fiscal_year is highest before reporting it as unavailable."
            ),
            latest_annual_fiscal_year=latest_annual_fiscal_year,
        )
    observations.sort(
        key=lambda item: (item.get("end") or "", item.get("filed") or ""),
        reverse=True,
    )
    return _json_result(
        ok=True,
        cik=normalized_cik,
        entity=company_facts.get("entityName"),
        taxonomy=taxonomy,
        concept=concept,
        label=fact.get("label", concept),
        form=requested_form or "any",
        latest_annual_fiscal_year=latest_annual_fiscal_year,
        observations=observations[:result_limit],
        total_observations=len(observations),
    )

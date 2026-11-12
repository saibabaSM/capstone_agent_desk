"""Load the full Item 1A (Risk Factors) section of a saved 10-K as ID-labelled passages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .filing_parser import load_passages

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILINGS_DIR = PROJECT_ROOT / "data" / "filings"
FILING_SUFFIXES = {".htm", ".html"}


@dataclass
class RiskPassage:
    id: str  # for example "microsoft-2025-1A-37"
    text: str


def _filing_files(year_dir: Path) -> list[Path]:
    if not year_dir.is_dir():
        return []
    return sorted(p for p in year_dir.iterdir() if p.suffix.lower() in FILING_SUFFIXES)


def _available(company: str) -> list[str]:
    """Years of this company that actually contain a filing (empty folders don't count)."""
    company_dir = FILINGS_DIR / company
    if not company_dir.is_dir():
        return []
    return sorted(p.name for p in company_dir.iterdir() if _filing_files(p))


# The data the workers can actually read: company -> fiscal years with a locally stored filing.
# This defines "available" for planning: a company's latest year is the newest year listed here.
# Example: {"alphabet": [2023, 2024, 2025], "apple": [2025], "microsoft": [2023, 2024, 2025]}
def available_filings() -> dict[str, list[int]]:
    if not FILINGS_DIR.is_dir():
        return {}
    result: dict[str, list[int]] = {}
    for company_dir in sorted(p for p in FILINGS_DIR.iterdir() if p.is_dir()):
        years = sorted(int(y) for y in _available(company_dir.name) if y.isdigit())
        if years:
            result[company_dir.name] = years
    return result


def find_filing(company: str, year: int | str) -> Path:
    """Return the 10-K file for data/filings/<company>/<year>/, or raise with what exists."""
    name = company.strip().lower()
    files = _filing_files(FILINGS_DIR / name / str(year))
    if files:
        return files[0]
    years = _available(name)
    if years:
        raise FileNotFoundError(
            f"No {name} 10-K for fiscal year {year}. Available years: {', '.join(years)}."
        )
    companies = (
        sorted(p.name for p in FILINGS_DIR.iterdir() if _available(p.name))
        if FILINGS_DIR.is_dir()
        else []
    )
    raise FileNotFoundError(
        f"No filings for company '{name}'. Available companies: {', '.join(companies) or 'none'}."
    )


def load_risk_section(company: str, year: int | str) -> list[RiskPassage]:
    """Return every Item 1A passage of the filing, numbered in document order."""
    path = find_filing(company, year)
    name = company.strip().lower()
    texts = [p.text for p in load_passages(path) if p.item.startswith("Item 1A")]
    if not texts:
        raise ValueError(f"No Item 1A passages found in {path.name}.")
    return [RiskPassage(f"{name}-{year}-1A-{i}", text) for i, text in enumerate(texts, start=1)]

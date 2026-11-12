"""The coverage universe: 12 companies in two sectors.

`key` is the lowercase folder name used under data/filings/<key>/<fiscal_year>/ (kept compatible
with the starter code, which used "microsoft", "alphabet", "apple").
`fy_end` is the month the fiscal year ends; it matters because Microsoft's FY2025 ends in June
2025 while Alphabet's ends in December 2025, so "FY2025" is not the same twelve months.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    ticker: str
    key: str
    name: str
    cik: str
    sector: str
    fy_end: str  # human description of the fiscal year end
    aliases: tuple[str, ...] = ()


COMPANIES: tuple[Company, ...] = (
    # Software and internet
    Company("MSFT", "microsoft", "Microsoft Corporation", "0000789019", "software_internet", "June 30", ("microsoft",)),
    Company("GOOGL", "alphabet", "Alphabet Inc.", "0001652044", "software_internet", "December 31", ("alphabet", "google")),
    Company("AAPL", "apple", "Apple Inc.", "0000320193", "software_internet", "late September", ("apple",)),
    Company("META", "meta", "Meta Platforms, Inc.", "0001326801", "software_internet", "December 31", ("meta platforms", "meta", "facebook")),
    Company("ORCL", "oracle", "Oracle Corporation", "0001341439", "software_internet", "May 31", ("oracle",)),
    Company("ADBE", "adobe", "Adobe Inc.", "0000796343", "software_internet", "late November / early December", ("adobe",)),
    # Semiconductors
    Company("NVDA", "nvidia", "NVIDIA Corporation", "0001045810", "semiconductors", "late January", ("nvidia",)),
    Company("AMD", "amd", "Advanced Micro Devices, Inc.", "0000002488", "semiconductors", "late December", ("advanced micro devices", "amd")),
    Company("INTC", "intel", "Intel Corporation", "0000050863", "semiconductors", "late December", ("intel",)),
    Company("QCOM", "qualcomm", "QUALCOMM Incorporated", "0000804328", "semiconductors", "late September", ("qualcomm",)),
    Company("TXN", "texas_instruments", "Texas Instruments Incorporated", "0000097476", "semiconductors", "December 31", ("texas instruments",)),
    Company("AVGO", "broadcom", "Broadcom Inc.", "0001730168", "semiconductors", "around November 1", ("broadcom",)),
)

BY_TICKER = {c.ticker: c for c in COMPANIES}
BY_KEY = {c.key: c for c in COMPANIES}


def resolve(name_or_ticker: str) -> Company | None:
    """Map a ticker, folder key, or alias (case-insensitive) to a Company."""
    text = name_or_ticker.strip()
    if text.upper() in BY_TICKER:
        return BY_TICKER[text.upper()]
    low = text.lower()
    if low in BY_KEY:
        return BY_KEY[low]
    for company in COMPANIES:
        if low in company.aliases or low == company.name.lower():
            return company
    return None


def fiscal_year_label(period_end: str) -> int:
    """Fiscal-year label for a period end date (YYYY-MM-DD).

    Companies label a fiscal year by the calendar year in which it ends, except that 52/53-week
    years ending in the first days of January belong to the previous year (AMD's fiscal 2022 ended
    on 31 Dec 2022, but some years end on e.g. 1 Jan). NVIDIA's year ending late January 2025 is
    its fiscal 2025, which the plain rule already gives.
    """
    year, month, day = (int(part) for part in period_end[:10].split("-"))
    if month == 1 and day <= 7:
        return year - 1
    return year

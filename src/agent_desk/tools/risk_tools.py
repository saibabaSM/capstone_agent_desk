"""Agent-facing tool for comparing a company's risk factors between two fiscal years."""

import json

from langchain_core.tools import tool

from ..rag.risk_compare import compare_company_years


@tool
def compare_risk_factors(company: str, year_from: int, year_to: int) -> str:
    """Compare a company's Item 1A risk factors between two fiscal years.

    Returns JSON with the risks that were added, removed or materially changed between the two
    years. Each finding has a description, the passage ids it comes from (ids look like
    "microsoft-2025-1A-37"), a short verbatim quote and, for changed risks, the old and new
    passages compared. Only the filings stored locally can be compared. If a year has no
    filing, ok is false and the error lists the years that are available.

    company: lowercase company name such as "microsoft" or "alphabet".
    year_from, year_to: the two fiscal years, for example 2024 and 2025, in either order.
    The comparison reads both full sections, so a call takes a minute or two.
    """
    try:
        diff = compare_company_years(company, year_from, year_to)
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2)

    result = {
        "ok": True,
        "company": diff["company"],
        "year_from": diff["year_from"],
        "year_to": diff["year_to"],
        "counts": {
            "added": len(diff["added"]),
            "removed": len(diff["removed"]),
            "changed": len(diff["changed"]),
        },
        "added": diff["added"],
        "removed": diff["removed"],
        "changed": diff["changed"],
        # How many model proposals verification dropped; the reasons are not sent to the agent.
        "rejected_count": len(diff["rejected"]),
    }
    if not (diff["added"] or diff["removed"] or diff["changed"]):
        result["note"] = "No material differences were found between the two years."
    return json.dumps(result, ensure_ascii=False, indent=2)

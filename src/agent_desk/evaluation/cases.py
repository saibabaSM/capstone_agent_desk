"""Build eval/{dev,heldout}/cases.jsonl from eval/gold/scenarios.yaml with gold values filled in."""

from __future__ import annotations

import json
from typing import Any

from ..companies import BY_TICKER
from ..config import PROJECT_ROOT, load_yaml
from .gold import gold_latest_years, gold_unit, gold_value, save_path

SCENARIOS = PROJECT_ROOT / "eval" / "gold" / "scenarios.yaml"


def _years(ticker: str, spec: Any) -> list[int]:
    if isinstance(spec, str) and spec.startswith("latest:"):
        return gold_latest_years(ticker, int(spec.split(":")[1]))
    return [int(y) for y in spec]


def build() -> dict[str, int]:
    scenarios = load_yaml(SCENARIOS)["scenarios"]
    out: dict[str, list[dict]] = {"dev": [], "heldout": []}
    for s in scenarios:
        expected: dict[str, Any] = {"status": "refused" if s["type"] == "refusal" else "published"}
        facts = []
        for ticker, metric, spec in s.get("facts", []):
            for fy in _years(ticker, spec):
                value = gold_value(ticker, metric, fy)
                facts.append({"ticker": ticker, "metric": metric, "fiscal_year": fy, "value": value, "unit": gold_unit(metric)})
        if facts:
            expected["facts"] = facts
        if s.get("risk"):
            expected["risk"] = [{"ticker": t, "year_from": a, "year_to": b} for t, a, b in s["risk"]]
        if s.get("cite"):
            t, y, item = s["cite"]
            expected["cite"] = {"company": BY_TICKER[t].key, "year": y, "item": item}
        if s.get("refusal"):
            expected["refusal"] = s["refusal"]
        if s.get("final_years"):
            expected["final_years"] = s["final_years"]
            expected["replan"] = True
        if s.get("new_year"):
            t, spec = s["new_year"]
            expected["new_year"] = {"ticker": t, "fiscal_year": _years(t, spec)[0]}
        if s.get("fy_caveat"):
            expected["fy_caveat"] = True
        tickers = sorted({f["ticker"] for f in facts} | {r["ticker"] for r in expected.get("risk", [])})
        required = []
        if facts:
            required.append("edgar.get_metric")
        if expected.get("risk"):
            required.append("filings_kb.diff_risk_factors")
        if expected.get("cite"):
            required.append("filings_kb.search_filings")
        case = {"id": s["id"], "split": s["split"], "type": s["type"], "question": s["question"],
                "setup_question": s.get("setup_question"), "tickers": tickers, "expected": expected,
                "required_tools": required, "forbidden": ["publish_without_approval", "advice_language", "unverified_numbers"]}
        missing = [f for f in facts if f["value"] is None]
        if missing:
            case["gold_warning"] = f"gold value unavailable for {missing}"
        out[s["split"]].append(case)
    for split, cases in out.items():
        path = save_path(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for c in cases:
                fh.write(json.dumps(c) + "\n")
    return {k: len(v) for k, v in out.items()}


def load(split: str) -> list[dict]:
    return [json.loads(l) for l in save_path(split).open(encoding="utf-8") if l.strip()]

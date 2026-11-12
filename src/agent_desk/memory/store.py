"""Long-term memory on a LangGraph store (SQLite). The written policy is docs/memory_policy.md.

Namespaces (who can read -> what):
  ("prefs", analyst)       the analyst's own runs      {"peer_table": true, "units": "billions", ...}
  ("coverage", analyst)    the analyst's own runs      {"tickers": [...]}
  ("watchlist", analyst)   guardrails ONLY             private tickers; never put in a prompt or tool arg
  ("notes", ticker)        every analyst on the desk   one record per PUBLISHED note: key facts + ids, no free text from the web
  ("lessons", "playbook")  every run (if enabled)      reviewed, frozen Reflexion lessons
  ("lessons", "candidates") reviewers only             unreviewed lessons; never retrieved by agents

Saved only at defined moments: prefs/watchlist when the analyst sets them (API), note records
after a note is PUBLISHED (an unapproved draft is never remembered), lessons when a human approves.
Deletion: forget_analyst() removes prefs, coverage and watchlist; delete_note_record() removes one
note record; every record carries `saved_at` and notes expire after NOTE_RETENTION_DAYS.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.store.base import BaseStore

NOTE_RETENTION_DAYS = 730
DEFAULT_PREFS = {"peer_table": True, "units": "billions", "max_words": 450}


async def get_prefs(store: BaseStore, analyst: str) -> dict[str, Any]:
    item = await store.aget(("prefs", analyst), "prefs")
    return {**DEFAULT_PREFS, **(item.value if item else {})}


async def set_prefs(store: BaseStore, analyst: str, prefs: dict[str, Any]) -> dict[str, Any]:
    allowed = {k: v for k, v in prefs.items() if k in ("peer_table", "units", "max_words", "focus")}
    merged = {**(await get_prefs(store, analyst)), **allowed, "saved_at": time.time()}
    await store.aput(("prefs", analyst), "prefs", merged)
    return merged


async def get_watchlist(store: BaseStore, analyst: str) -> list[str]:
    item = await store.aget(("watchlist", analyst), "watchlist")
    return list(item.value.get("tickers", [])) if item else []


async def set_watchlist(store: BaseStore, analyst: str, tickers: list[str]) -> None:
    await store.aput(("watchlist", analyst), "watchlist", {"tickers": sorted({t.upper() for t in tickers}), "saved_at": time.time()})


async def add_coverage(store: BaseStore, analyst: str, tickers: list[str]) -> None:
    item = await store.aget(("coverage", analyst), "coverage")
    current = set(item.value.get("tickers", [])) if item else set()
    await store.aput(("coverage", analyst), "coverage", {"tickers": sorted(current | set(tickers)), "saved_at": time.time()})


async def get_coverage(store: BaseStore, analyst: str) -> list[str]:
    item = await store.aget(("coverage", analyst), "coverage")
    return list(item.value.get("tickers", [])) if item else []


async def add_note_record(store: BaseStore, ticker: str, record: dict[str, Any]) -> str:
    key = f"note-{record.get('note_id') or int(time.time() * 1000)}"
    await store.aput(("notes", ticker.upper()), key, {**record, "saved_at": time.time()})
    return key


async def last_note(store: BaseStore, ticker: str) -> dict[str, Any] | None:
    items = await store.asearch(("notes", ticker.upper()), limit=100)
    fresh = [i for i in items if time.time() - i.value.get("saved_at", 0) < NOTE_RETENTION_DAYS * 86400]
    if not fresh:
        return None
    return max(fresh, key=lambda i: i.value.get("saved_at", 0)).value


async def delete_note_record(store: BaseStore, ticker: str, key: str) -> None:
    await store.adelete(("notes", ticker.upper()), key)


async def forget_analyst(store: BaseStore, analyst: str) -> dict[str, int]:
    removed = {}
    for ns, key in ((("prefs", analyst), "prefs"), (("coverage", analyst), "coverage"), (("watchlist", analyst), "watchlist")):
        exists = await store.aget(ns, key)
        if exists:
            await store.adelete(ns, key)
        removed[ns[0]] = int(bool(exists))
    return removed


async def export_analyst(store: BaseStore, analyst: str) -> dict[str, Any]:
    """Everything stored about one analyst (for review/correction requests)."""
    return {"prefs": await get_prefs(store, analyst), "coverage": await get_coverage(store, analyst),
            "watchlist": await get_watchlist(store, analyst)}


def what_changed(previous: dict[str, Any] | None, facts: dict[str, dict[str, Any]], ticker: str) -> list[str]:
    """Code diff between the key facts of the last note and the facts gathered now."""
    if not previous:
        return [f"No earlier published note for {ticker} in memory."]
    lines = [f"Last note on {ticker}: {time.strftime('%Y-%m-%d', time.gmtime(previous.get('saved_at', 0)))} "
             f"(question: {previous.get('question', '')[:120]})."]
    old = previous.get("key_facts", {})
    new = {f"{f['metric']}/FY{f['fiscal_year']}": f for f in facts.values() if f["ticker"] == ticker}
    old_years = {int(k.split("/FY")[1]) for k in old}
    new_years = {f["fiscal_year"] for f in new.values()}
    if new_years - old_years:
        lines.append(f"New fiscal year(s) since then: {sorted(new_years - old_years)}.")
    for key, f in sorted(new.items()):
        if key in old and abs(float(old[key]["value"]) - float(f["value"])) > 1e-6 * max(1.0, abs(float(f["value"]))):
            # no old number in prose: it would be an unreferenced figure; the current one is cited in the table
            lines.append(f"{key}: the reported value differs from our last note (restated or re-tagged); "
                         f"current fact `{f['fact_id']}`.")
    if previous.get("risk_counts"):
        lines.append(f"Last note's risk-factor comparison: {previous['risk_counts']}.")
    return lines


async def get_lessons(store: BaseStore, kind: str = "playbook") -> list[dict[str, Any]]:
    return [i.value | {"key": i.key} for i in await store.asearch(("lessons", kind), limit=500)]


async def put_lesson(store: BaseStore, kind: str, key: str, lesson: dict[str, Any]) -> None:
    await store.aput(("lessons", kind), key, lesson)


async def delete_lesson(store: BaseStore, kind: str, key: str) -> None:
    await store.adelete(("lessons", kind), key)

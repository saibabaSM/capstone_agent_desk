"""Command line for Analyst Desk.

    python -m agent_desk sync-data                       download 10-Ks + XBRL into the local cache
    python -m agent_desk ingest [--plain]                build the hybrid index (table-aware or plain)
    python -m agent_desk ask "Compare ..." [--auto] [--experiment configs/experiments/e1_c1.yaml]
    python -m agent_desk resume <thread_id>              continue a killed or paused run
    python -m agent_desk serve [--port 8000]             FastAPI + UI + /metrics
    python -m agent_desk mcp <filings_kb|edgar|python_sandbox|research_portal> [--http PORT]
    python -m agent_desk graph                           print the graph as Mermaid
    python -m agent_desk memory show|forget|watchlist|prefs --analyst alice [...]
    python -m agent_desk eval ...                        see agent_desk/evaluation/cli.py
    python -m agent_desk reflexion run|review|freeze     see agent_desk/improvement/reflexion.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap

from .config import load_yaml


def _print_plan(plan: dict) -> None:
    print("\n=== PLAN v%s ===" % plan.get("version"))
    print("Goal:", plan["goal"])
    for cy in plan["years"]:
        print(f"  {cy['ticker']}: fiscal years {cy['fiscal_years']}")
    print("Metrics:", ", ".join(plan["metrics"]) or "-", "| risk diff:", plan["risk_diff"],
          "| what changed since last note:", plan.get("what_changed_since_last_note"))
    for i, q in enumerate(plan["sub_questions"], 1):
        print(f"  {i}. {q}")
    for n in plan.get("notes", []):
        print("  note:", n)


def _ask_plan(plan: dict) -> dict:
    _print_plan(plan)
    choice = input("\n[a]pprove / [e]dit years / [r]eject? ").strip().lower() or "a"
    if choice.startswith("r"):
        return {"action": "reject"}
    if choice.startswith("e"):
        years = []
        for cy in plan["years"]:
            raw = input(f"  {cy['ticker']} fiscal years {cy['fiscal_years']} -> ").strip()
            years.append({"ticker": cy["ticker"], "fiscal_years": [int(y) for y in raw.replace(",", " ").split()] or cy["fiscal_years"]})
        return {"action": "edit", "plan": {"years": years}}
    return {"action": "approve"}


def _ask_note(payload: dict) -> dict:
    print("\n=== NOTE FOR APPROVAL ===  hash", payload["note_hash"][:16])
    print(payload["note"])
    check = payload["output_check"]
    print(f"\nOutput checks: {'PASS' if check['passed'] else 'FAIL'}; numbers verified {check['verified']}/{check['numbers']}")
    for issue in check["issues"]:
        print("  -", issue)
    choice = input("\n[a]pprove / [e]dit (opens $EDITOR) / [r]eject? ").strip().lower() or "a"
    if choice.startswith("r"):
        return {"action": "reject", "reason": input("reason: ")}
    if choice.startswith("e"):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as fh:
            fh.write(payload["note"])
        subprocess.call([os.getenv("EDITOR", "vi"), fh.name])
        return {"action": "edit", "edited_note": open(fh.name, encoding="utf-8").read()}
    return {"action": "approve", "note_hash": payload["note_hash"], "approver": os.getenv("USER", "analyst")}


async def _drive(rt, events, interactive: bool) -> str:
    thread = None
    while True:
        pending = None
        async for ev in events:
            thread = ev["thread_id"]
            if "interrupt" in ev:
                pending = ev["interrupt"]
            else:
                upd = ev["update"]
                brief = {k: upd[k] for k in ("status", "message") if k in upd}
                print(f"[{ev['node']}] {json.dumps(brief) if brief else ''}", flush=True)
                if ev["node"] == "critic" and upd.get("critic_reports"):
                    r = upd["critic_reports"][-1]
                    print(f"   critic round {r['round']}: {'PASS' if r['passed'] else 'FAIL'} "
                          f"(numbers {r['numbers_verified']}/{r['numbers_counted']})")
                    for issue in r["issues"][:5]:
                        print("    -", textwrap.shorten(issue, 160))
                if ev["node"] == "replan":
                    print("   replanned:", upd["plan_history"][-1]["reason"])
        if pending is None:
            break
        if not interactive:
            print(f"\nPaused for a human decision ({pending['type']}). Resume with: python -m agent_desk resume {thread}")
            return thread
        answer = _ask_plan(pending["plan"]) if pending["type"] == "plan_review" else _ask_note(pending)
        events = rt.resume(thread, answer)
    st = await rt.state(thread)
    vals = st["values"]
    print("\n=== RESULT ===", vals.get("status"), vals.get("message", ""))
    if vals.get("status") in ("published", "rejected") and vals.get("note"):
        print(vals["note"])
    u = vals.get("usage") or {}
    print(f"\nthread {thread} | tokens {u.get('tokens', 0)} | cost ${u.get('cost_usd', 0):.4f} | "
          f"tool calls {sum((u.get('tool_calls') or {}).values())} | trace data/traces/{thread}.jsonl")
    return thread


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent_desk")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync-data"); s.add_argument("--tickers", nargs="*"); s.add_argument("--filings", type=int, default=3)
    s = sub.add_parser("ingest"); s.add_argument("--plain", action="store_true"); s.add_argument("--both", action="store_true")
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("--analyst", default="analyst")
    s.add_argument("--experiment"); s.add_argument("--auto", action="store_true", help="auto-approve plan and note")
    s.add_argument("--no-input", action="store_true", help="pause at interrupts instead of prompting")
    s.add_argument("--thread"); s.add_argument("--plant-wrong-number", action="store_true")
    s = sub.add_parser("resume"); s.add_argument("thread"); s.add_argument("--no-input", action="store_true")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.add_argument("--host", default="127.0.0.1")
    s = sub.add_parser("mcp"); s.add_argument("server"); s.add_argument("--http", type=int)
    sub.add_parser("graph")
    s = sub.add_parser("memory"); s.add_argument("action", choices=["show", "forget", "watchlist", "prefs"])
    s.add_argument("--analyst", default="analyst"); s.add_argument("values", nargs="*")
    s = sub.add_parser("eval"); s.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("reflexion"); s.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    if args.cmd == "sync-data":
        from .data_sync import sync
        sync(args.tickers, args.filings)
    elif args.cmd == "ingest":
        from .rag.hybrid import ingest
        for ta in ([True, False] if args.both else [not args.plain]):
            print(ingest(table_aware=ta))
    elif args.cmd in ("ask", "resume"):
        from .agents.runtime import desk_runtime

        async def go() -> None:
            async with desk_runtime() as rt:
                if args.cmd == "ask":
                    overrides = load_yaml(args.experiment) if args.experiment else {}
                    overrides.pop("name", None); overrides.pop("description", None)
                    if args.auto:
                        overrides.setdefault("orchestration", {}).update(auto_approve_plan=True, auto_approve_note=True)
                    if args.plant_wrong_number:
                        overrides.setdefault("demo", {})["plant_wrong_number"] = True
                    events = rt.start(args.question, args.analyst, overrides, args.thread)
                else:
                    events = rt.resume(args.thread)
                await _drive(rt, events, interactive=not args.no_input and sys.stdin.isatty())
        asyncio.run(go())
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("agent_desk.api.server:app", host=args.host, port=args.port)
    elif args.cmd == "mcp":
        import importlib
        from .tools.mcp_client import DEFAULT_PORTS, SERVERS
        server = importlib.import_module(SERVERS[args.server]).mcp
        if args.http:
            server.run(transport="http", host="0.0.0.0", port=args.http or DEFAULT_PORTS[args.server], show_banner=False)
        else:
            server.run(show_banner=False)
    elif args.cmd == "graph":
        from .agents.team_graph import build_graph
        print(build_graph().compile().get_graph().draw_mermaid())
    elif args.cmd == "memory":
        from .agents.runtime import desk_runtime
        from .memory import store as mem

        async def go_mem() -> None:
            async with desk_runtime() as rt:
                if args.action == "show":
                    print(json.dumps(await mem.export_analyst(rt.store, args.analyst), indent=2))
                elif args.action == "forget":
                    print(await mem.forget_analyst(rt.store, args.analyst))
                elif args.action == "watchlist":
                    await mem.set_watchlist(rt.store, args.analyst, args.values)
                    print("watchlist saved (private)")
                elif args.action == "prefs":
                    kv = dict(v.split("=", 1) for v in args.values)
                    kv = {k: (v.lower() == "true" if v.lower() in ("true", "false") else v) for k, v in kv.items()}
                    print(await mem.set_prefs(rt.store, args.analyst, kv))
        asyncio.run(go_mem())
    elif args.cmd == "eval":
        from .evaluation.cli import main as eval_main
        return eval_main(args.rest)
    elif args.cmd == "reflexion":
        from .improvement.reflexion import main as rx_main
        return rx_main(args.rest)
    return 0

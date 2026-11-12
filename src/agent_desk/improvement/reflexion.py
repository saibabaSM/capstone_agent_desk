"""Reflexion loop on DEVELOPMENT cases only (L12).

    python -m agent_desk reflexion run [--experiment configs/experiments/e1_c2_team.yaml]
        agent runs every dev case -> checkers score it -> each failure (and each critic rejection)
        goes to a reflection step that writes ONE short lesson with the condition it applies to
        -> eval/playbook/candidates.jsonl (status: pending). Nothing is used yet.
    python -m agent_desk reflexion review                      interactive: approve / reject (+ reason) each pending lesson
    python -m agent_desk reflexion review --approve L01 L03 --reject L02 --reason "too case-specific"
    python -m agent_desk reflexion freeze                      approved lessons -> eval/playbook/frozen.json (+ sha256)
    python -m agent_desk reflexion withdraw L03 --reason "made held-out results worse"

The playbook is frozen BEFORE the held-out run; `eval run --split heldout --playbook` loads exactly
the frozen file. Lessons are retrieved per case by relevance (improvement/playbook.py), never all.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

from ..config import PROJECT_ROOT, get_config, is_offline
from ..control.context import RunContext
from ..llm.models import chat
from ..observability.tracing import read_trace

PLAYBOOK_DIR = PROJECT_ROOT / "eval" / "playbook"
CANDIDATES = PLAYBOOK_DIR / "candidates.jsonl"
FROZEN = PLAYBOOK_DIR / "frozen.json"

# Offline reflection: failure pattern -> generic lesson (used when no LLM is available).
RULES = [
    (r"missing from the note's facts|never cited", "Fetch every planned metric for every planned fiscal year and cite each value with its [F#] reference.",
     "the question asks for a metric over several fiscal years", ["numbers", "coverage"]),
    (r"agent .* vs gold", "Use the catalog metric (edgar.get_metric) rather than a raw concept; companies change revenue tags between years.",
     "a company reports revenue under more than one XBRL concept", ["xbrl", "revenue"]),
    (r"Item 1A citation", "For risk-factor questions, quote at least one added or changed Item 1A passage from the later fiscal year.",
     "the question asks what changed in risk factors", ["risk", "citation"]),
    (r"fiscal_year_caveat|fiscal year", "State each company's fiscal-year end when comparing companies; FY labels cover different calendar periods.",
     "comparing companies with different fiscal year ends", ["comparison", "fiscal_year"]),
    (r"replan|final plan years", "If a requested fiscal year has no filing yet, use the latest available year and say so in the caveats.",
     "the requested fiscal year may not be filed yet", ["planning", "fiscal_year"]),
    (r"NUMBER .*unattributed|unverified", "Never write a number that is not in the fact list; quote filing numbers verbatim inside quotation marks with the passage id.",
     "the draft mentions numbers from MD&A text", ["numbers", "writer"]),
]


def _load() -> list[dict]:
    if not CANDIDATES.exists():
        return []
    return [json.loads(l) for l in CANDIDATES.open(encoding="utf-8") if l.strip()]


def _save(rows: list[dict]) -> None:
    PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _failures(result: dict) -> list[str]:
    out = list(result["check"]["notes"]) + list(result["check"]["violations"])
    out += [k for k, v in result["check"]["checks"].items() if not v]
    for e in read_trace(result["thread"]):
        if e["type"] == "critic" and not e.get("passed"):
            out += [f"critic: {i}" for i in e.get("issues", [])[:4]]
    return list(dict.fromkeys(out))[:12]


def reflect(result: dict, failures: list[str], ctx: RunContext) -> dict | None:
    trajectory = ", ".join(f"{t['agent']}:{t['tool']}:{t['status']}" for t in result.get("tool_log", [])[:25])
    if is_offline(ctx.cfg):
        text = " ".join(failures)
        for pattern, lesson, when, tags in RULES:
            if re.search(pattern, text, re.I):
                return {"lesson": lesson, "applies_when": when, "tags": tags}
        return None
    prompt = ctx.cfg["prompts"]["reflection"].format(question=result.get("question", result["case_id"]),
                                                     failures="\n".join(failures), trajectory=trajectory)
    raw = chat("reflection", ctx, max_tokens=250).invoke([("human", prompt)]).content
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not data or not data.get("lesson") or not data.get("applies_when"):
        return None
    return {"lesson": data["lesson"][:300], "applies_when": data["applies_when"][:200], "tags": data.get("tags", [])[:5]}


def run(experiment: str | None, limit: int | None) -> None:
    from ..evaluation import cases as case_mod
    from ..evaluation.runner import load_results
    from ..evaluation.runner import run as run_cases
    cases = case_mod.load("dev")[: limit or None]
    path = asyncio.run(run_cases(cases, experiment, rep=1, label="reflexion"))
    results = load_results(path)
    questions = {c["id"]: c["question"] for c in cases}
    cfg = get_config()
    ctx = RunContext(run_id=f"reflexion-{time.strftime('%Y%m%d-%H%M%S')}", cfg=cfg, agent="reflection")
    rows = _load()
    start = len(rows)
    for r in results:
        failures = _failures(r)
        if r["check"]["success"] and not any(f.startswith("critic:") for f in failures):
            continue
        lesson = reflect({**r, "question": questions.get(r["case_id"])}, failures, ctx)
        if lesson is None:
            continue
        rows.append({"id": f"L{len(rows) + 1:02d}", "status": "pending", **lesson, "source_case": r["case_id"],
                     "failures": failures[:6], "created_at": time.strftime("%Y-%m-%d %H:%M"), "split": "dev"})
    _save(rows)
    run_cost = sum(r["cost_usd"] for r in results)
    print(f"{len(rows) - start} new candidate lessons -> {CANDIDATES}")
    print(f"cost of generating lessons: agent runs ${run_cost:.4f} + reflection ${ctx.cost_usd:.4f}")


def review(approve: list[str], reject: list[str], reason: str, reviewer: str) -> None:
    rows = _load()
    interactive = not approve and not reject
    for r in rows:
        if r["status"] != "pending":
            continue
        decision = None
        if r["id"] in approve:
            decision = "approved"
        elif r["id"] in reject:
            decision = "rejected"
        elif interactive:
            print(f"\n{r['id']} (from {r['source_case']})\n  WHEN: {r['applies_when']}\n  LESSON: {r['lesson']}\n  failures: {r['failures'][:3]}")
            ans = input("  [a]pprove / [r]eject / [s]kip? ").strip().lower()
            decision = {"a": "approved", "r": "rejected"}.get(ans[:1])
            if decision == "rejected":
                reason = input("  reason: ").strip()
        if decision:
            r.update(status=decision, reviewer=reviewer, reviewed_at=time.strftime("%Y-%m-%d %H:%M"),
                     review_reason=reason if decision == "rejected" else "")
    _save(rows)
    print({s: sum(r["status"] == s for r in rows) for s in ("pending", "approved", "rejected", "withdrawn")})


def freeze() -> None:
    lessons = [{"key": r["id"], **{k: r[k] for k in ("lesson", "applies_when", "tags", "source_case")}, "status": "approved"}
               for r in _load() if r["status"] == "approved"]
    blob = json.dumps(lessons, sort_keys=True).encode()
    FROZEN.write_text(json.dumps({"frozen_at": time.strftime("%Y-%m-%d %H:%M"), "sha256": hashlib.sha256(blob).hexdigest(),
                                  "lessons": lessons}, indent=2), encoding="utf-8")
    print(f"froze {len(lessons)} lessons -> {FROZEN}")


def withdraw(lesson_id: str, reason: str) -> None:
    rows = _load()
    for r in rows:
        if r["id"] == lesson_id:
            r.update(status="withdrawn", review_reason=reason, reviewed_at=time.strftime("%Y-%m-%d %H:%M"))
    _save(rows)
    print(f"{lesson_id} withdrawn; run `reflexion freeze` again before the next held-out run")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="agent_desk reflexion")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run"); s.add_argument("--experiment"); s.add_argument("--limit", type=int)
    s = sub.add_parser("review"); s.add_argument("--approve", nargs="*", default=[]); s.add_argument("--reject", nargs="*", default=[])
    s.add_argument("--reason", default=""); s.add_argument("--reviewer", default="reviewer")
    sub.add_parser("freeze")
    s = sub.add_parser("withdraw"); s.add_argument("id"); s.add_argument("--reason", required=True)
    args = p.parse_args(argv)
    if args.cmd == "run":
        run(args.experiment, args.limit)
    elif args.cmd == "review":
        review(args.approve, args.reject, args.reason, args.reviewer)
    elif args.cmd == "freeze":
        freeze()
    elif args.cmd == "withdraw":
        withdraw(args.id, args.reason)
    return 0

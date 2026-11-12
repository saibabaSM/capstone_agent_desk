"""python -m agent_desk eval <command>

  build-gold                                       scenarios.yaml -> eval/{dev,heldout}/cases.jsonl (gold from SEC)
  run --split dev --experiment configs/experiments/e1_c2.yaml [--reps 3] [--limit N] [--ids a b] [--playbook]
  report eval/results/a.jsonl [b.jsonl ...]       comparison table (markdown) -> stdout
  gate eval/results/x.jsonl --min-success 0.8 --min-verified 0.98     exit 1 below threshold (CI)
  retrieval [--out eval/reports/retrieval.md]     Recall@5 ablations over eval/retrieval/queries.jsonl
  redteam --experiment configs/experiments/e2_defences_off.yaml      attack success rate
  judge-sample --results x.jsonl                  claims for humans to label -> eval/judge/to_label.jsonl
  judge-agreement                                 Cohen's kappa of the judge vs human labels
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..config import PROJECT_ROOT
from . import cases as case_mod
from .runner import load_results, run, summarize

PLAYBOOK = PROJECT_ROOT / "eval" / "playbook" / "frozen.json"


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="agent_desk eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-gold")
    s = sub.add_parser("run")
    s.add_argument("--split", default="dev", choices=["dev", "heldout"])
    s.add_argument("--experiment"); s.add_argument("--reps", type=int, default=1); s.add_argument("--limit", type=int)
    s.add_argument("--ids", nargs="*"); s.add_argument("--types", nargs="*"); s.add_argument("--concurrency", type=int, default=3)
    s.add_argument("--playbook", action="store_true", help="load the frozen playbook and enable lessons")
    s.add_argument("--label", default="")
    s = sub.add_parser("report"); s.add_argument("files", nargs="+")
    s = sub.add_parser("gate"); s.add_argument("file"); s.add_argument("--min-success", type=float, default=0.8)
    s.add_argument("--min-verified", type=float, default=0.98)
    s = sub.add_parser("retrieval"); s.add_argument("--out")
    s = sub.add_parser("redteam"); s.add_argument("--experiment"); s.add_argument("--ids", nargs="*")
    s.add_argument("--label", default="")
    s = sub.add_parser("judge-sample"); s.add_argument("--results", nargs="+", required=True); s.add_argument("--n", type=int, default=25)
    s = sub.add_parser("judge-agreement"); s.add_argument("--file", default=str(PROJECT_ROOT / "eval" / "judge" / "labels.jsonl"))
    args = p.parse_args(argv)

    if args.cmd == "build-gold":
        print(case_mod.build())
    elif args.cmd == "run":
        cases = case_mod.load(args.split)
        if args.ids:
            cases = [c for c in cases if c["id"] in args.ids]
        if args.types:
            cases = [c for c in cases if c["type"] in args.types]
        if args.limit:
            cases = cases[: args.limit]
        playbook = json.loads(PLAYBOOK.read_text())["lessons"] if args.playbook and PLAYBOOK.exists() else None
        extra = {"improvement": {"playbook_enabled": True}} if playbook else None
        paths = []
        for rep in range(1, args.reps + 1):
            print(f"== {args.experiment or 'default'} | {args.split} | rep {rep} | {len(cases)} cases")
            paths.append(asyncio.run(run(cases, args.experiment, rep, args.concurrency, args.label, playbook, extra)))
        for path in paths:
            print(json.dumps(summarize(load_results(path))))
    elif args.cmd == "report":
        rows = []
        for f in args.files:
            res = load_results(f)
            rows.append({"run": Path(f).stem, **summarize(res)})
        cols = ["run", "cases", "success", "success_rate", "policy_violations", "false_blocks", "errors", "cost_per_case_usd",
                "latency_p50_s", "latency_p95_s", "tool_calls_per_case", "critic_rejections", "verified_share_min"]
        print(table(rows, cols))
    elif args.cmd == "gate":
        res = load_results(args.file)
        s = summarize(res)
        published = [r for r in res if r["status"] == "published"]
        verified = min((r["verified_share"] for r in published if r.get("verified_share") is not None), default=1.0)
        print(json.dumps(s))
        failures = []
        if s["success_rate"] < args.min_success:
            failures.append(f"success rate {s['success_rate']} < {args.min_success}")
        if verified < args.min_verified:
            failures.append(f"verified-number share {verified} < {args.min_verified}")
        if s["policy_violations"]:
            failures.append(f"{s['policy_violations']} policy violation(s)")
        if failures:
            print("EVAL GATE FAILED: " + "; ".join(failures))
            for r in res:
                if not r["check"]["success"]:
                    print(f"  - {r['case_id']}: {r['check']['notes'][:2]} {r['check']['violations']}")
            return 1
        print("EVAL GATE PASSED")
    elif args.cmd == "retrieval":
        from .retrieval_eval import main as r_main
        return r_main(args.out)
    elif args.cmd == "redteam":
        from .redteam import main as rt_main
        return rt_main(args.experiment, args.ids, args.label)
    elif args.cmd == "judge-sample":
        from .judge_calibration import sample
        print(sample(args.results, args.n))
    elif args.cmd == "judge-agreement":
        from .judge_calibration import agreement
        print(json.dumps(agreement(args.file), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

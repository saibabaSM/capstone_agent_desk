"""Retrieve only the playbook lessons relevant to the current case (L12).

A lesson: {"lesson": "...", "applies_when": "...", "tags": [...], "status": "approved", ...}
Relevance = BM25-style token overlap between the case text and applies_when + tags + lesson;
lessons under cfg.improvement.lesson_min_score are not shown, at most lessons_top_k are.
"""

from __future__ import annotations

from typing import Any

from ..rag.hybrid import tokenize


def score(lesson: dict[str, Any], text: str) -> float:
    case = set(tokenize(text))
    when = set(tokenize(lesson.get("applies_when", "") + " " + " ".join(lesson.get("tags", []))))
    body = set(tokenize(lesson.get("lesson", "")))
    if not case:
        return 0.0
    return (2 * len(case & when) + len(case & body)) / (2 * max(len(when), 1) + max(len(body), 1))


def relevant(lessons: list[dict[str, Any]], text: str, cfg: dict) -> list[dict[str, Any]]:
    ranked = sorted(((score(l, text), l) for l in lessons if l.get("status", "approved") == "approved"), key=lambda x: -x[0])
    return [l | {"score": round(s, 3)} for s, l in ranked if s >= cfg["improvement"]["lesson_min_score"]][: cfg["improvement"]["lessons_top_k"]]


def as_prompt(lessons: list[dict[str, Any]]) -> str:
    if not lessons:
        return ""
    return "LESSONS FROM REVIEWED PAST MISTAKES (apply only when the condition holds):\n" + "\n".join(
        f"- When {l['applies_when']}: {l['lesson']}" for l in lessons)

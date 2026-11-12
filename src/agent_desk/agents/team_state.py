"""Shared state schema for the supervisor + specialist team."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

# Returned as the only item of a results update to empty the list (used when the plan is replaced).
RESET = "__reset_results__"


def accumulate_or_reset(current: list[str] | None, update: list[str]) -> list[str]:
    """Reducer for worker results: append new entries, or start over when the update is [RESET].

    Example: accumulate_or_reset(["a"], ["b"]) -> ["a", "b"]
             accumulate_or_reset(["a", "b"], [RESET]) -> []
    """
    if update == [RESET]:
        return []
    return (current or []) + update


class TeamState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    next: str
    task: str
    plan: dict
    plan_step: int
    replan_count: int
    max_replans: int
    replan_reason: str
    # Worker results accumulate (one entry per worker call) so repeat calls don't overwrite;
    # a replan empties them with [RESET] because the new plan starts from scratch.
    narrative: Annotated[list[str], accumulate_or_reset]
    facts: Annotated[list[str], accumulate_or_reset]
    risk: Annotated[list[str], accumulate_or_reset]
    final_answer: str

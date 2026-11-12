"""The whole graph, offline: plan interrupt, critic catching a planted number, the approval gate
refusing a stale approval, an edit invalidating the approval, publish, memory, and resume."""

import asyncio

from agent_desk.agents.runtime import desk_runtime
from agent_desk.agents.team_graph import build_graph

from .conftest import needs_index

Q = "Compare Microsoft's and Alphabet's operating margins over the last two fiscal years."


def test_graph_compiles_with_all_nodes():
    nodes = set(build_graph().compile().get_graph().nodes)
    assert {"plan_review", "researcher", "quant", "risk_diff", "critic", "approval_gate", "publish", "replan"} <= nodes


def test_refused_request_stops_at_input_guard(tmp_path):
    async def go():
        async with desk_runtime(str(tmp_path / "cp.db"), str(tmp_path / "st.db")) as rt:
            events = [e async for e in rt.start("Should I buy Intel stock now?", "t", {"llm": {"mode": "offline"}}, thread_id="r1")]
            return events, await rt.state("r1")
    events, state = asyncio.run(go())
    assert state["values"]["status"] == "refused" and [e["node"] for e in events] == ["input_guard"]


@needs_index
def test_full_human_in_the_loop_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTAL_DB", str(tmp_path / "portal.db"))
    ov = {"llm": {"mode": "offline"}, "demo": {"plant_wrong_number": True}}

    async def go():
        async with desk_runtime(str(tmp_path / "cp.db"), str(tmp_path / "st.db")) as rt:
            evs = [e async for e in rt.start(Q, "alice", ov, thread_id="t1")]
            assert evs[-1]["interrupt"]["type"] == "plan_review"
            evs = [e async for e in rt.resume("t1", {"action": "approve"})]
            critic = [e for e in evs if e.get("node") == "critic"]
            assert not critic[0]["update"]["critic_reports"][-1]["passed"]      # planted number caught
            assert critic[-1]["update"]["critic_reports"][-1]["passed"]         # and fixed
            gate = evs[-1]["interrupt"]
            assert gate["type"] == "note_approval" and gate["output_check"]["passed"]
            h1 = gate["note_hash"]
            # the analyst edits the note: the old approval must no longer work
            edited = gate["note"].replace("## Summary", "## Summary\nEdited by the analyst.", 1)
            evs = [e async for e in rt.resume("t1", {"action": "edit", "edited_note": edited})]
            h2 = evs[-1]["interrupt"]["note_hash"]
            assert h2 != h1
            evs = [e async for e in rt.resume("t1", {"action": "approve", "note_hash": h1, "approver": "alice"})]
            assert evs[-1]["interrupt"]["type"] == "note_approval"                # stale approval refused, asked again
            st = await rt.state("t1")
            assert st["values"]["approval"]["status"] == "stale"
            evs = [e async for e in rt.resume("t1", {"action": "approve", "note_hash": h2, "approver": "alice"})]
            st = (await rt.state("t1"))["values"]
            assert st["status"] == "published" and st["published"]["note_hash"] == h2
            notes = await rt.store.asearch(("notes", "MSFT"))
            assert notes and notes[0].value["note_hash"] == h2
    asyncio.run(go())


@needs_index
def test_resume_after_crash(tmp_path, monkeypatch):
    """Kill the run after the plan (simulated: stop consuming events), then resume from the checkpoint."""
    monkeypatch.setenv("PORTAL_DB", str(tmp_path / "portal.db"))
    ov = {"llm": {"mode": "offline"}, "orchestration": {"auto_approve_plan": True, "auto_approve_note": True}}

    async def first():
        async with desk_runtime(str(tmp_path / "cp.db"), str(tmp_path / "st.db")) as rt:
            async for e in rt.start(Q, "bob", ov, thread_id="crash"):
                if e.get("node") == "plan_review":
                    break   # the process "dies" here

    async def second():
        async with desk_runtime(str(tmp_path / "cp.db"), str(tmp_path / "st.db")) as rt:
            before = await rt.state("crash")
            assert before["next"]                      # work is pending
            _ = [e async for e in rt.resume("crash")]
            return (await rt.state("crash"))["values"]["status"]

    asyncio.run(first())
    assert asyncio.run(second()) == "published"

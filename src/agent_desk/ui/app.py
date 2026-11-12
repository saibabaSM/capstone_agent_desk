"""Streamlit page for Agent Desk: ask a question, review and edit the plan, approve it, read the answer.

Run it from the project folder:

    streamlit run src/agent_desk/ui/app.py

The page walks through four phases, kept in st.session_state["phase"]:

    ask      the question box and Run; Run starts the graph, which stops at the plan
    plan     Step 1: review the plan; Edit unlocks it; Approve and run resumes the graph
    working  Step 2: a spinner while the graph runs the approved plan
    answer   the final answer with Approve (marks it approved on screen) and a new question

An unchanged plan is sent back as {"action": "approve"}, an edited one as {"action": "edit", ...},
which is what approve_plan_node in agents/workflow.py expects. If the supervisor replans, the graph
stops at the plan again, so the page returns to "plan" with the new plan. The graph runs with an
in-memory checkpointer, so a paused run is lost when the page is reloaded.
"""

from __future__ import annotations

import copy
import sys
import uuid
from pathlib import Path

# `streamlit run` executes this file as a script, so the folder that contains the agent_desk
# package (src) has to be put on the import path first.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_desk.agents.workflow import build_team_workflow, initial_state  # noqa: E402
from agent_desk.rag.risk_section import available_filings  # noqa: E402

# The writer's answer and the worker findings are joined with this text in writer_node.
EVIDENCE_MARKER = "\n\nSupporting findings:\n\n"

DEFAULTS = {
    "phase": "ask",
    "run_id": 0,  # changes on every reset, so the question box never reuses an old widget key
    "question": "",
    "graph": None,  # the compiled graph, built once per run
    "config": None,  # holds the thread id that ties the pause and the resume together
    "plan": None,  # the plan the graph paused on
    "plan_version": 0,  # changes with every new plan, so the plan's widgets start fresh
    "questions": [],  # [{"id": 0, "text": "..."}]; ids stay fixed when a question is removed
    "next_id": 0,
    "editing": False,
    "pending": None,  # the answer to resume the graph with, set by Approve and run
    "answer": "",
    "approved": False,
    "replanned": False,
    "error": "",
}


# ---- Small helpers ---------------------------------------------------------------------------


# "2023-2025" for consecutive years, otherwise "2023, 2025".
def _years(years: list[int]) -> str:
    if len(years) > 1 and years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}-{years[-1]}"
    return ", ".join(map(str, years))


# "Available filings: alphabet 2023-2025 · apple 2025 · microsoft 2023-2025"
def filings_line() -> str:
    parts = [f"{company} {_years(years)}" for company, years in available_filings().items()]
    return "Available filings: " + " · ".join(parts) if parts else "No filings are stored."


# The plan with whitespace trimmed and empty entries dropped. Comparing the plan on screen with
# the plan the graph paused on, in this form, tells an edit from an unchanged plan.
def normalized(plan: dict) -> dict:
    return {
        "goal": str(plan.get("goal", "")).strip(),
        "sub_questions": [str(q).strip() for q in plan.get("sub_questions", []) if str(q).strip()],
        "outline": [str(h).strip() for h in plan.get("outline", []) if str(h).strip()],
    }


# The plan as it currently stands in the page's fields.
def current_plan() -> dict:
    s, v = st.session_state, st.session_state.plan_version
    outline = s[f"outline_{v}"] if f"outline_{v}" in s else ", ".join(s.plan.get("outline", []))
    return normalized(
        {
            "goal": s[f"goal_{v}"] if f"goal_{v}" in s else s.plan.get("goal", ""),
            "sub_questions": [
                s[f"q_{v}_{q['id']}"] if f"q_{v}_{q['id']}" in s else q["text"] for q in s.questions
            ],
            "outline": outline.split(","),
        }
    )


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, copy.deepcopy(value))


# Clears the page for a new question. run_id and plan_version are kept and never go back down:
# the browser remembers each widget's text by its key and sends it back on the next rerun, so a key
# that was used before would bring the old text back (an old question, or an old plan's fields).
def reset() -> None:
    s = st.session_state
    run_id, plan_version = s.get("run_id", 0) + 1, s.get("plan_version", 0)
    for key in list(s.keys()):
        del s[key]
    s.run_id, s.plan_version = run_id, plan_version


def add_question(text: str = "") -> None:
    s = st.session_state
    s.questions.append({"id": s.next_id, "text": text})
    s.next_id += 1


# Makes a plan the plan on screen: new widget keys (so no old text lingers) and locked fields.
def load_plan(plan: dict) -> None:
    s = st.session_state
    s.plan = plan
    s.plan_version += 1
    s.editing = False
    s.questions, s.next_id = [], 0
    for question in plan.get("sub_questions", []):
        add_question(str(question))


# Runs the graph (first run or resume). A failure is kept as a message instead of a traceback, and
# the page moves to the "error" phase, because a failed run cannot be continued.
def run_graph(payload) -> dict | None:
    s = st.session_state
    try:
        return s.graph.invoke(payload, s.config)
    except Exception as exc:
        s.error = f"{type(exc).__name__}: {exc}"
        s.phase = "error"
        return None


# Moves the page on after the graph stops: to the plan if it paused there, to the answer if it ended.
def handle_result(result: dict) -> None:
    s = st.session_state
    paused = result.get("__interrupt__")
    if paused:
        load_plan(paused[0].value["plan"])
        s.phase = "plan"
    else:
        s.answer = result.get("final_answer", "")
        s.phase = "answer"


# ---- Sections of the page --------------------------------------------------------------------


def render_question() -> None:
    s = st.session_state
    asking = s.phase == "ask"
    box = f"question_box_{s.run_id}"
    st.text_area(
        "Question",
        key=box,
        height=110,
        disabled=not asking,
        placeholder="For example: Compare Microsoft's and Alphabet's operating margins over the last three fiscal years.",
    )
    if not asking:
        return
    if st.button("Run", type="primary"):
        question = st.session_state.get(box, "").strip()
        if not question:
            st.warning("Type a question first.")
        else:
            s.question = question
            s.graph = build_team_workflow(InMemorySaver())
            s.config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 40}
            with st.spinner("Creating the plan..."):
                result = run_graph(initial_state(question))
            if result is not None:
                handle_result(result)
            st.rerun()
    st.caption(filings_line())


# Approve and run: sends the plan back as approve (unchanged) or edit (changed) and moves on to Step 2.
def approve_plan() -> None:
    s = st.session_state
    plan = current_plan()
    if not plan["sub_questions"]:
        st.error("The plan needs at least one question.")
        return
    s.pending = {"action": "approve"} if plan == normalized(s.plan) else {"action": "edit", "plan": plan}
    s.editing = False
    s.replanned = False
    s.phase = "working"
    st.rerun()


def render_plan() -> None:
    s = st.session_state
    v = s.plan_version
    reviewing = s.phase == "plan"
    editable = reviewing and s.editing

    st.divider()
    status = ":orange[waiting for your approval]" if reviewing else ":green[approved]"
    st.subheader(f"Step 1: Review the plan  {status}")
    if reviewing and s.replanned:
        st.info("The previous plan could not be completed, so the supervisor asked for a new one. Review it below.")

    st.text_input("Goal", value=s.plan.get("goal", ""), key=f"goal_{v}", disabled=not editable)

    st.markdown("**Sub-questions** (the work steps, run in this order)")
    for number, question in enumerate(list(s.questions), start=1):
        box, remove = st.columns([12, 2], vertical_alignment="bottom")
        box.text_input(f"{number}.", value=question["text"], key=f"q_{v}_{question['id']}", disabled=not editable)
        if editable and remove.button("✕", key=f"remove_{v}_{question['id']}", help="Remove this question"):
            s.questions = [q for q in s.questions if q["id"] != question["id"]]
            st.rerun()
    if editable and st.button("+ Add question", key=f"add_{v}"):
        add_question("")
        st.rerun()

    st.text_input(
        "Answer outline (comma separated)",
        value=", ".join(s.plan.get("outline", [])),
        key=f"outline_{v}",
        disabled=not editable,
    )

    if reviewing:
        approve, edit, _ = st.columns([3, 2, 5])
        if approve.button("Approve and run", type="primary", use_container_width=True):
            approve_plan()
        if edit.button("Done editing" if s.editing else "Edit", use_container_width=True):
            s.editing = not s.editing
            st.rerun()
        st.caption(
            "Change the goal, reword or remove questions, add new ones, then approve. Your edited plan is used as written."
            if s.editing
            else "The fields are locked. Click Edit to change the plan."
        )


# Step 2 while the graph runs; the page is busy until it stops.
def render_working() -> None:
    s = st.session_state
    st.divider()
    st.subheader("Step 2: Working...")
    with st.spinner(
        "Running the approved plan. This can take a few minutes: each risk-factor comparison reads two full filings."
    ):
        result = run_graph(Command(resume=s.pending))
    s.pending = None
    if result is not None:
        s.replanned = bool(result.get("__interrupt__"))  # a replan sends the page back to the plan
        handle_result(result)
    st.rerun()


# The writer's text first, the worker findings behind a collapsible section.
def render_answer() -> None:
    s = st.session_state
    st.divider()
    st.subheader("Step 2: Done")

    badge = ":green[approved]" if s.approved else ":orange[waiting for your approval]"
    st.subheader(f"Final answer  {badge}")
    if s.answer.startswith("Supporting findings:\n\n"):
        written, evidence = "", s.answer.split("\n\n", 1)[1]
    else:
        written, _, evidence = s.answer.partition(EVIDENCE_MARKER)
    st.markdown(written or "(The run produced no answer text.)")
    if evidence:
        with st.expander("Supporting findings"):
            st.text(evidence)

    approve, new, _ = st.columns([2, 3, 4])
    if approve.button("Approve", type="primary", disabled=s.approved, use_container_width=True):
        s.approved = True
        st.rerun()
    if new.button("Start a new question", use_container_width=True):
        reset()
        st.rerun()
    if s.approved:
        st.success("✓ Answer approved")


def render_error() -> None:
    st.divider()
    st.error(f"The run stopped: {st.session_state.error}")
    if st.button("Start over"):
        reset()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Agent Desk", layout="centered")
    init_state()
    st.title("Agent Desk")
    st.caption("Ask a question about company 10-K filings")

    phase = st.session_state.phase
    render_question()
    if phase in ("plan", "working", "answer"):
        render_plan()
    if phase == "working":
        render_working()
    elif phase == "answer":
        render_answer()
    elif phase == "error":
        render_error()


main()

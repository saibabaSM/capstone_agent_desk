import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from streamlit.testing.v1 import AppTest

from agent_desk.agents import workflow as w

APP_FILE = Path(__file__).resolve().parents[2] / "src" / "agent_desk" / "ui" / "app.py"

PLAN = {"goal": "g1", "sub_questions": ["Q-one", "Q-two"], "outline": ["o1"]}
PLAN_JSON = json.dumps(PLAN)


class Run:
    """The real graph with every model call replaced, recording what the supervisor and lead saw."""

    def __init__(self, monkeypatch, supervisor=None):
        self.lead_feedback = []
        self.supervisor_views = []
        self._supervisor = supervisor or self._finish_after_one_worker
        monkeypatch.setattr(w, "run_research_lead", self._lead)
        monkeypatch.setattr(w, "run_quant_agent", lambda task: "answer to " + task)
        monkeypatch.setattr(w, "supervisor_decision", self._supervise)

        class Reply:
            content = "WRITTEN ANSWER"

        class Llm:
            def invoke(self, messages):
                return Reply()

        monkeypatch.setattr(w, "get_tool_llm", lambda: Llm())

    def _lead(self, question, feedback=None, llm=None):
        self.lead_feedback.append(feedback)
        return PLAN_JSON

    def _supervise(self, state):
        self.supervisor_views.append(
            {"plan": state["plan"], "messages": [(m.name, str(m.content)) for m in state["messages"]]}
        )
        return self._supervisor(state, len(self.supervisor_views))

    @staticmethod
    def _finish_after_one_worker(state, call_number):
        if call_number % 2:
            task = (state["plan"].get("sub_questions") or ["?"])[0]
            return {"next": "quant_worker", "task": task, "messages": [AIMessage(content="brief", name="supervisor")]}
        return {"next": "FINISH", "task": "done", "messages": [AIMessage(content="fin", name="supervisor")]}


def start(thread="t"):
    """A graph with a checkpointer (so it has the approval step) and the config to run it."""
    app = w.build_team_workflow(InMemorySaver())
    return app, {"configurable": {"thread_id": thread}, "recursion_limit": 40}


def pause(result):
    items = result.get("__interrupt__")
    return items[0].value if items else None


# ---- approve_plan_node through the real graph ------------------------------------------------


def test_the_run_pauses_with_the_plan_and_approve_resumes_it(monkeypatch):
    run = Run(monkeypatch)
    app, config = start()

    result = app.invoke(w.initial_state("Q?"), config)

    assert pause(result) == {"type": "plan_approval", "question": "Q?", "plan": PLAN}
    assert result["final_answer"] == ""

    result = app.invoke(Command(resume={"action": "approve"}), config)

    assert pause(result) is None and result["final_answer"].startswith("WRITTEN ANSWER")
    assert len(run.lead_feedback) == 1  # resuming does not run the research lead again
    assert run.supervisor_views[0]["plan"] == PLAN


def test_the_plain_string_approve_works(monkeypatch):
    Run(monkeypatch)
    app, config = start()
    app.invoke(w.initial_state("Q?"), config)

    result = app.invoke(Command(resume="approve"), config)

    assert pause(result) is None and result["final_answer"].startswith("WRITTEN ANSWER")


def test_an_edited_plan_is_used_exactly_as_given(monkeypatch):
    run = Run(monkeypatch)
    app, config = start()
    app.invoke(w.initial_state("Q?"), config)
    edited = {"goal": "my goal", "sub_questions": [" My own question ", ""], "outline": ["mine"]}

    result = app.invoke(Command(resume={"action": "edit", "plan": edited}), config)

    assert pause(result) is None
    assert app.get_state(config).values["plan"] == edited  # not cleaned, not checked
    seen = run.supervisor_views[0]
    assert seen["plan"] == edited
    # The lead's summary was replaced, not kept next to the new one.
    assert [name for name, _ in seen["messages"]] == [None, "research_lead"]
    assert "my goal" in seen["messages"][1][1] and "edited by the user" in seen["messages"][1][1]
    assert "Q-one" not in seen["messages"][1][1]


@pytest.mark.parametrize(
    "answer",
    [
        {"action": "reject", "feedback": "no"},
        {"action": "dance"},
        {"action": "edit"},
        {"action": "edit", "plan": "not a dict"},
        "something else",
    ],
)
def test_any_other_answer_counts_as_approve(monkeypatch, answer):
    run = Run(monkeypatch)
    app, config = start()
    app.invoke(w.initial_state("Q?"), config)

    result = app.invoke(Command(resume=answer), config)

    assert pause(result) is None and result["final_answer"].startswith("WRITTEN ANSWER")
    assert run.supervisor_views[0]["plan"] == PLAN and len(run.lead_feedback) == 1
    assert app.get_state(config).values["replan_count"] == 0


def test_auto_approve_skips_the_pause(monkeypatch):
    Run(monkeypatch)
    app = w.build_team_workflow(InMemorySaver())
    config = {"configurable": {"thread_id": "t", "auto_approve": True}, "recursion_limit": 40}

    result = app.invoke(w.initial_state("Q?"), config)

    assert pause(result) is None and result["final_answer"].startswith("WRITTEN ANSWER")


def test_a_supervisor_replan_asks_for_approval_again(monkeypatch):
    script = iter(["REPLAN", "quant_worker", "FINISH"])

    def supervisor(state, call_number):
        step = next(script)
        if step == "REPLAN":
            return {"next": "REPLAN", "replan_reason": "no data", "messages": [AIMessage(content="r", name="supervisor")]}
        if step == "quant_worker":
            return {"next": "quant_worker", "task": "Q", "messages": [AIMessage(content="b", name="supervisor")]}
        return {"next": "FINISH", "task": "done", "messages": [AIMessage(content="f", name="supervisor")]}

    run = Run(monkeypatch, supervisor)
    app, config = start()

    first = app.invoke(w.initial_state("Q?"), config)
    second = app.invoke(Command(resume={"action": "approve"}), config)  # supervisor replans -> new plan
    final = app.invoke(Command(resume={"action": "approve"}), config)

    assert pause(first) is not None and pause(second) is not None  # asked twice
    assert pause(final) is None and final["final_answer"].startswith("WRITTEN ANSWER")
    assert len(run.lead_feedback) == 2 and "no data" in run.lead_feedback[1]


def test_without_a_checkpointer_there_is_no_approval_step_and_the_run_finishes(monkeypatch):
    Run(monkeypatch)

    graph = w.build_team_workflow()
    assert "approve_plan" not in graph.get_graph().nodes
    assert "approve_plan" in w.build_team_workflow(InMemorySaver()).get_graph().nodes

    result = graph.invoke(w.initial_state("Q?"), {"recursion_limit": 40})
    assert pause(result) is None and result["final_answer"].startswith("WRITTEN ANSWER")
    assert w.run_team_workflow("Q?").startswith("WRITTEN ANSWER")


# ---- The Streamlit page (ui/app.py), driven headlessly with Streamlit's AppTest ---------------
# The real graph and checkpointer run; only the model calls are replaced (see Run above).


def new_page():
    return AppTest.from_file(str(APP_FILE), default_timeout=60).run()


def button(page, label):
    """Buttons are found by label: in edit mode the remove buttons come first in the list."""
    return next(b for b in page.button if b.label == label)


def labels(page):
    return [b.label for b in page.button]


def to_plan(page, question="Q?"):
    page.text_area[0].set_value(question).run()
    button(page, "Run").click().run()
    assert not page.exception, [e.value for e in page.exception]
    return page


def question_boxes(page):
    return [t.value for t in page.text_input if t.key.startswith("q_")]


def test_the_first_screen_has_only_the_question_box_and_run(monkeypatch):
    Run(monkeypatch)

    page = new_page()

    assert page.session_state.phase == "ask" and labels(page) == ["Run"]
    assert page.caption[-1].value.startswith("Available filings:")


def test_an_empty_question_shows_a_warning_and_stays_on_the_first_screen(monkeypatch):
    Run(monkeypatch)
    page = new_page()

    button(page, "Run").click().run()

    assert page.session_state.phase == "ask" and "Type a question first" in page.warning[0].value


def test_run_shows_the_plan_locked_with_approve_and_edit(monkeypatch):
    Run(monkeypatch)

    page = to_plan(new_page())

    assert page.session_state.phase == "plan"
    assert page.text_input[0].value == "g1" and page.text_input[0].disabled
    assert question_boxes(page) == ["Q-one", "Q-two"]
    assert [t.value for t in page.text_input if t.key.startswith("outline_")] == ["o1"]
    assert page.text_area[0].disabled  # the question can no longer be changed
    assert labels(page) == ["Approve and run", "Edit"]


def test_edit_unlocks_the_plan_and_the_edited_plan_reaches_the_supervisor(monkeypatch):
    run = Run(monkeypatch)
    page = to_plan(new_page())

    button(page, "Edit").click().run()
    assert not page.text_input[0].disabled
    assert {"Done editing", "+ Add question"} <= set(labels(page)) and labels(page).count("✕") == 2

    page.text_input[1].set_value("Q-one reworded").run()
    version = page.session_state.plan_version
    page.button(key=f"add_{version}").click().run()
    new_id = page.session_state.questions[-1]["id"]
    page.text_input(key=f"q_{version}_{new_id}").set_value("Q-three added").run()
    page.button(key=f"remove_{version}_{page.session_state.questions[1]['id']}").click().run()  # drop Q-two
    page.text_input(key=f"goal_{version}").set_value("My own goal").run()
    assert question_boxes(page) == ["Q-one reworded", "Q-three added"]

    button(page, "Approve and run").click().run()

    assert page.session_state.phase == "answer" and not page.exception
    assert run.supervisor_views[0]["plan"] == {
        "goal": "My own goal",
        "sub_questions": ["Q-one reworded", "Q-three added"],
        "outline": ["o1"],
    }
    assert len(run.lead_feedback) == 1  # approving does not plan again


def test_an_unchanged_plan_is_approved_as_it_is(monkeypatch):
    run = Run(monkeypatch)
    page = to_plan(new_page())

    button(page, "Approve and run").click().run()

    assert page.session_state.phase == "answer"
    assert run.supervisor_views[0]["plan"] == PLAN


def test_a_whitespace_only_edit_counts_as_unchanged(monkeypatch):
    run = Run(monkeypatch)
    page = to_plan(new_page())
    button(page, "Edit").click().run()
    page.text_input[1].set_value("  Q-one  ").run()

    button(page, "Approve and run").click().run()

    assert run.supervisor_views[0]["plan"] == PLAN


def test_a_plan_with_no_questions_is_not_sent(monkeypatch):
    run = Run(monkeypatch)
    page = to_plan(new_page())
    button(page, "Edit").click().run()
    version = page.session_state.plan_version
    for question in list(page.session_state.questions):
        page.button(key=f"remove_{version}_{question['id']}").click().run()

    button(page, "Approve and run").click().run()

    assert page.session_state.phase == "plan" and "at least one question" in page.error[0].value
    assert run.supervisor_views == []


def test_the_answer_text_is_shown_and_the_findings_sit_in_an_expander(monkeypatch):
    Run(monkeypatch)

    class Reply:
        content = "Written part.\n\nSupporting findings:\n\nQuant findings:\nQ: a\nA: b"

    class Llm:
        def invoke(self, messages):
            return Reply()

    monkeypatch.setattr(w, "get_tool_llm", lambda: Llm())
    page = to_plan(new_page())

    button(page, "Approve and run").click().run()

    text = " ".join(m.value for m in page.markdown)
    assert "Written part." in text and "Supporting findings" not in text
    assert [e.label for e in page.expander] == ["Supporting findings"]
    assert "Q: a" in page.expander[0].text[0].value
    assert any("Final answer" in s.value and "waiting for your approval" in s.value for s in page.subheader)


def test_approving_the_answer_marks_it_and_a_new_question_starts_clean(monkeypatch):
    Run(monkeypatch)
    page = to_plan(new_page(), "First question")
    button(page, "Approve and run").click().run()
    first_run_id, first_version = page.session_state.run_id, page.session_state.plan_version

    button(page, "Approve").click().run()

    assert page.session_state.approved and button(page, "Approve").disabled
    assert any("Answer approved" in s.value for s in page.success)

    button(page, "Start a new question").click().run()

    assert page.session_state.phase == "ask" and page.text_area[0].value == ""
    # Widget keys must never be reused: the browser would bring the old text back.
    assert page.session_state.run_id > first_run_id and page.session_state.plan_version == first_version
    page = to_plan(page, "Second question")
    assert page.session_state.plan_version > first_version


def test_a_supervisor_replan_sends_the_page_back_to_the_plan(monkeypatch):
    script = iter(["REPLAN", "quant_worker", "FINISH"])

    def supervisor(state, call_number):
        step = next(script)
        if step == "REPLAN":
            return {"next": "REPLAN", "replan_reason": "no data", "messages": [AIMessage(content="r", name="supervisor")]}
        if step == "quant_worker":
            return {"next": "quant_worker", "task": "Q", "messages": [AIMessage(content="b", name="supervisor")]}
        return {"next": "FINISH", "task": "done", "messages": [AIMessage(content="f", name="supervisor")]}

    run = Run(monkeypatch, supervisor)
    page = to_plan(new_page())

    button(page, "Approve and run").click().run()

    assert page.session_state.phase == "plan" and "could not be completed" in page.info[0].value
    assert len(run.lead_feedback) == 2 and "no data" in run.lead_feedback[1]

    button(page, "Approve and run").click().run()

    assert page.session_state.phase == "answer"


def test_a_failed_run_shows_the_error_and_start_over_works(monkeypatch):
    Run(monkeypatch)
    monkeypatch.setattr(w, "run_research_lead", lambda q, feedback=None, llm=None: (_ for _ in ()).throw(RuntimeError("model down")))
    page = new_page()
    page.text_area[0].set_value("Q?").run()

    button(page, "Run").click().run()

    assert page.session_state.phase == "error" and labels(page) == ["Start over"]
    assert "RuntimeError: model down" in page.error[0].value
    button(page, "Start over").click().run()
    assert page.session_state.phase == "ask"

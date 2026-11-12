from langchain_core.messages import HumanMessage

from agent_desk.agents import supervisor_agent as sup


class FakeRouter:
    """Plays back one scripted result per invoke: an Exception is raised, anything else returned."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def patch_router(monkeypatch, router):
    seen = {}

    class FakeLLM:
        def with_structured_output(self, *args, **kwargs):
            return router

    def fake_get_tool_llm(max_tokens=None):
        seen["max_tokens"] = max_tokens
        return FakeLLM()

    monkeypatch.setattr(sup, "get_tool_llm", fake_get_tool_llm)
    return seen


def state():
    return {
        "question": "Q?",
        "messages": [HumanMessage(content="Q?"), HumanMessage(content="Plan ...", name="research_lead")],
        "plan": {"sub_questions": ["What was Apple's operating income for fiscal 2025?"]},
        "plan_step": 0,
        "replan_count": 0,
        "max_replans": 2,
    }


def ok(next_step="quant_worker", message="brief"):
    return {"raw": None, "parsed": {"next": next_step, "message": message}, "parsing_error": None}


def test_router_call_is_capped(monkeypatch):
    seen = patch_router(monkeypatch, FakeRouter([ok()]))

    sup.supervisor_decision(state())

    assert seen["max_tokens"] == sup.MAX_ROUTE_TOKENS == 1000


def test_a_runaway_reply_is_retried_once_and_the_second_answer_is_used(monkeypatch):
    router = FakeRouter([RuntimeError("LengthFinishReasonError: length limit was reached"), ok("quant_worker", "ask quant")])
    patch_router(monkeypatch, router)

    update = sup.supervisor_decision(state())

    assert router.calls == 2
    assert update["next"] == "quant_worker" and update["task"] == "ask quant"


def test_an_unparseable_reply_is_retried_too(monkeypatch):
    bad = {"raw": None, "parsed": None, "parsing_error": "bad json"}
    router = FakeRouter([bad, ok("FINISH", "done")])
    patch_router(monkeypatch, router)

    update = sup.supervisor_decision(state())

    assert router.calls == 2 and update["next"] == "FINISH" and update["task"] == "done"


def test_two_failures_stop_with_the_real_error_in_the_message(monkeypatch):
    router = FakeRouter([ValueError("first problem"), RuntimeError("Could not parse response: length limit")])
    patch_router(monkeypatch, router)

    update = sup.supervisor_decision(state())

    assert router.calls == 2 and update["next"] == "FINISH"
    assert "after 2 attempts" in update["task"]
    assert "RuntimeError: Could not parse response: length limit" in update["task"]  # the last error is kept


def test_a_good_first_reply_makes_a_single_call(monkeypatch):
    router = FakeRouter([ok()])
    patch_router(monkeypatch, router)

    sup.supervisor_decision(state())

    assert router.calls == 1


def test_an_invalid_route_name_counts_as_a_failed_attempt(monkeypatch):
    router = FakeRouter([ok("nonsense_worker", "x"), ok("risk_worker", "compare")])
    patch_router(monkeypatch, router)

    update = sup.supervisor_decision(state())

    assert router.calls == 2 and update["next"] == "risk_worker"

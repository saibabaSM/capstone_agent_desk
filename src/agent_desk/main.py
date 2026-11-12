import os
import sys
from pathlib import Path

from langsmith import tracing_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_desk.agents.langgraph_workflow import run_langgraph_workflow
from agent_desk.agents.workflow import run_team_workflow

QUESTION1 = "What was Apple's operating income in the latest fiscal year?"
QUESTION2 = (
    "According to Apple's 10-K, what supply chain risks does the company "
    "describe in its Risk Factors?"
)

QUESTION3 = (
    "Compare Microsoft’s and Alphabet’s operating margins over the last three fiscal years, "
    "and tell me what changed in their risk factors."
)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        if os.getenv("LANGSMITH_API_KEY"):
            with tracing_context(
                enabled=True,
                project_name=os.getenv("LANGCHAIN_PROJECT", "agent-desk"),
                metadata={"question": QUESTION1},
            ):
                # print(run_langgraph_workflow(QUESTION1))
                print(run_team_workflow(QUESTION1))
        else:
            # print(run_langgraph_workflow(QUESTION))
            print(run_team_workflow(QUESTION3))
        return 0
    except Exception as exc:  # pragma: no cover - debugger path
        print(f"Workflow failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

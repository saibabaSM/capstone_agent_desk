import os

import pytest

from agent_desk.config import data_dir

os.environ.setdefault("AGENT_DESK_LLM_MODE", "offline")

HAS_FACTS = (data_dir() / "edgar_cache" / "companyfacts" / "CIK0000789019.json").exists()
HAS_INDEX = (data_dir() / "index" / "table" / "chunks.jsonl").exists()

needs_facts = pytest.mark.skipif(not HAS_FACTS, reason="run `python -m agent_desk sync-data` first")
needs_index = pytest.mark.skipif(not (HAS_FACTS and HAS_INDEX), reason="run `python -m agent_desk ingest` first")


@pytest.fixture
def cfg():
    from agent_desk.config import get_config
    c = get_config()
    c["llm"]["mode"] = "offline"
    return c

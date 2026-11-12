# Analyst Desk - common commands. `make help` lists them.
PY ?= .venv/bin/python
export PYTHONPATH := src

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

setup:           ## create .venv (Python 3.11) and install pinned requirements
	python3.11 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install torch && $(PY) -m pip install -r requirements.txt

data:            ## download 10-Ks + XBRL for the 12 companies and build both indexes (~10 min, once)
	$(PY) -m agent_desk sync-data && $(PY) -m agent_desk ingest --both

up:              ## everything in Docker: app, 4 MCP servers, Qdrant, Prometheus, Grafana
	docker compose up --build

serve:           ## local app + UI at http://127.0.0.1:8000 (MCP servers in-process)
	$(PY) -m agent_desk serve

test:            ## unit + graph tests (offline)
	$(PY) -m pytest -q

eval-ci:         ## the CI eval gate, offline
	AGENT_DESK_LLM_MODE=offline $(PY) -m agent_desk eval run --split dev --experiment configs/experiments/ci_offline.yaml --label ci
	$(PY) -m agent_desk eval gate $$(ls -t eval/results/ci_offline-ci-*.jsonl | head -1)

retrieval:       ## Recall@5 ablations -> eval/reports/retrieval.md
	$(PY) -m agent_desk eval retrieval --out eval/reports/retrieval.md

experiments:     ## E1, E2, E4 on the held-out split, 3 repetitions each (live LLM; costs money)
	for e in e1_c0_fixed e1_c1_single e1_c2_team e4_critic_on_planted e4_critic_off_planted e4_plain_chunks; do \
	  $(PY) -m agent_desk eval run --split heldout --experiment configs/experiments/$$e.yaml --reps 3; done
	$(PY) -m agent_desk eval redteam --experiment configs/experiments/e2_defences_on.yaml
	$(PY) -m agent_desk eval redteam --experiment configs/experiments/e2_defences_off.yaml

.PHONY: help setup data up serve test eval-ci retrieval experiments

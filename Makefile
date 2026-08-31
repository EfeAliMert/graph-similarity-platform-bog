PYTHON ?= python3
PYG_PYTHON := .venvs/gnn-pyg/bin/python
GRAPHSIM_PYTHON := .venvs/graphsim/bin/python

.PHONY: run test test-artifacts check preflight manifest matrix-plan matrix-full matrix-existing study hpo checkpoint-audit model-audit graphsim-calibrate setup models models-verify datasets datasets-verify checkpoints checkpoints-verify

run:
	$(PYG_PYTHON) app.py

test:
	$(PYG_PYTHON) -m unittest discover -s tests -v

test-artifacts:
	GSP_REQUIRE_ARTIFACTS=1 $(PYG_PYTHON) -m unittest discover -s tests -v

check:
	PYTHONPYCACHEPREFIX=tmp/pycache $(PYG_PYTHON) -m compileall -q app.py graph_similarity_platform tests scripts
	node --check static/app.js

preflight:
	$(PYG_PYTHON) scripts/preflight_check.py --base-url http://127.0.0.1:5002

manifest:
	$(PYG_PYTHON) scripts/generate_artifact_manifest.py

matrix-plan:
	$(PYG_PYTHON) scripts/run_research_matrix.py --benchmark-only

matrix-full:
	$(PYG_PYTHON) scripts/run_research_matrix.py --benchmark-only --models all --seeds 379,2026,3407 --budget 25 --batch-size 32 --evaluate-pairs 50 --execute

checkpoint-audit:
	$(PYG_PYTHON) scripts/audit_checkpoints.py

model-audit:
	$(PYG_PYTHON) scripts/audit_model_outputs.py

matrix-existing:
	$(PYG_PYTHON) scripts/run_research_matrix.py --benchmark-only --evaluate-existing --evaluate-pairs 50 --execute --continue-on-error

study:
	$(PYG_PYTHON) scripts/complete_research_study.py --continue-on-error

hpo:
	$(PYG_PYTHON) scripts/optimize_all.py --budget quick --skip-existing --continue-on-error

graphsim-calibrate:
	@for dataset in aids700nef linux imdbmulti ptc mutag proteins enzymes; do \
		$(GRAPHSIM_PYTHON) scripts/calibrate_graphsim_checkpoint.py --dataset $$dataset; \
	done

setup:
	$(PYTHON) -m venv .venvs/gnn-pyg
	$(PYG_PYTHON) -m pip install pip==25.3
	$(PYG_PYTHON) -m pip install -r requirements-gnn-pyg.txt
	$(PYG_PYTHON) scripts/repair_macos_wheel_tags.py
	$(PYTHON) -m venv .venvs/graphsim
	$(GRAPHSIM_PYTHON) -m pip install pip==25.3
	$(GRAPHSIM_PYTHON) -m pip install -r requirements-graphsim.txt
	$(GRAPHSIM_PYTHON) scripts/repair_macos_wheel_tags.py

models:
	$(PYTHON) scripts/fetch_models.py

models-verify:
	$(PYTHON) scripts/fetch_models.py --verify-only

datasets:
	$(PYG_PYTHON) scripts/fetch_datasets.py
	$(PYG_PYTHON) scripts/prepare_model_datasets.py

datasets-verify:
	$(PYG_PYTHON) scripts/fetch_datasets.py --verify-only
	$(PYG_PYTHON) scripts/prepare_model_datasets.py --verify-only

checkpoints:
	$(PYTHON) scripts/fetch_checkpoints.py

checkpoints-verify:
	$(PYTHON) scripts/fetch_checkpoints.py --verify-only

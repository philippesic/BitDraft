.PHONY: setup check doctor build-cpu build-metal build-cuda

setup:
	git submodule update --init --recursive
	uv sync --locked

check:
	uv run --locked ruff check src scripts tests
	uv run --locked ruff format --check src scripts tests
	uv run --locked python -c 'import pathlib, tomllib; [tomllib.loads(p.read_text()) for d in ("configs", ".codex") for p in pathlib.Path(d).rglob("*.toml")]'
	uv run --group w1a1 --locked python -m unittest discover -s tests
	git diff --check

doctor:
	uv run --locked python scripts/environment.py

build-cpu:
	uv run --locked python scripts/build_llama.py cpu

build-metal:
	uv run --locked python scripts/build_llama.py metal

build-cuda:
	uv run --locked python scripts/build_llama.py cuda

# Current workflow contracts; CPU fixtures only, no model or GPU allocation.
PY ?= .venv/bin/python
.PHONY: workflow-check
workflow-check:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$(PY)" -m unittest discover -s scripts/contender_pipeline -p 'test_*.py'

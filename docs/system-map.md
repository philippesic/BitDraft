# System map

| Flow | State | Path | Symbol | Trigger |
| --- | --- | --- | --- | --- |
| setup | active | pyproject.toml | check | Explicit wheel-only uv sync |
| check | active | tools/check.py | main | Local and pre-commit; optional hosted runner |
| doctor | active | tools/doctor.py | diagnose | Read-only environment and hook inspection |
| selftest | active | tools/self_test.py | run_corpus | Stage 9 |
| policy | active | tools/policy_review.py | verification_report | Full passing check |
| W1 | active | w1a1_eagle/index.py | IndexReader | Retained metadata and role admission |
| W2 | pending | w1a1_eagle/native/teacher.py | capture | Separate hardware gate |
| W3 | active | w1a1_eagle/numerical/corpus.py | Corpus | Bounded CPU capture sampling |
| W4 | active | w1a1_eagle/supervisor.py | execute | Scoped CPU run |
| W5 | active | w1a1_eagle/numerical/worker.py | restore | Own schema-3 tensor resume/branch |
| W6 | active | w1a1_eagle/numerical/export.py | export_he4 | Bounded CPU GGUF construction |
| W7 | pending | w1a1_eagle/training/lifecycle.py | evaluate | Separate source gate |
| W8 | pending | benchmarks/__main__.py | qualify | Separate hardware gate |
| W9 | pending | benchmarks/__main__.py | run | Separate benchmark gate |
| W10 | active | w1a1_eagle/lifecycle.py | Journal | Exposure and attempt accounting |
| W11 | pending | benchmarks/runner.py | execute | Separate benchmark gate |
| W12 | pending | w1a1_eagle/native/build.py | build | Separate native gate |

Pending paths reserve responsibilities; they do not exist or provide commands.
Active check flow: environment → repository → comments → Ruff → mypy → architecture
→ dependencies → documentation → CPU tests → specimens → qualification status.

# System map

| Flow | State | Path | Symbol | Trigger |
| --- | --- | --- | --- | --- |
| setup | active | pyproject.toml | check | Explicit wheel-only uv sync |
| check | active | tools/check.py | main | Local, pre-commit and CI |
| selftest | active | tools/self_test.py | run_corpus | Stage 9 |
| policy | active | tools/policy_review.py | verification_report | Full passing check |
| W1 | pending | w1a1_eagle/data/ledger.py | import_sources | Separate source gate |
| W2 | pending | w1a1_eagle/native/teacher.py | capture | Separate hardware gate |
| W3 | pending | w1a1_eagle/data/windows.py | windows | Separate source gate |
| W4 | pending | w1a1_eagle/training/lifecycle.py | fresh | Separate source gate |
| W5 | pending | w1a1_eagle/checkpoints/store.py | resume | Separate source gate |
| W6 | pending | w1a1_eagle/native/export.py | export | Separate source gate |
| W7 | pending | w1a1_eagle/training/lifecycle.py | evaluate | Separate source gate |
| W8 | pending | benchmarks/__main__.py | qualify | Separate hardware gate |
| W9 | pending | benchmarks/__main__.py | run | Separate benchmark gate |
| W10 | pending | w1a1_eagle/data/ledger.py | consume | Separate source gate |
| W11 | pending | benchmarks/runner.py | execute | Separate benchmark gate |
| W12 | pending | w1a1_eagle/native/build.py | build | Separate native gate |

Pending paths reserve responsibilities; they do not exist or provide commands.
Active check flow: environment → repository → comments → Ruff → mypy → architecture
→ dependencies → documentation → CPU tests → specimens → qualification status.

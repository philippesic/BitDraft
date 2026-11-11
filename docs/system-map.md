# System map

## Layout

| Path | Holds |
| --- | --- |
| `w1a1_eagle/` | Product: data admission, intent, supervision, lifecycle, CPU training |
| `w1a1_eagle/numerical/` | Torch and GGUF code: captures, corpus, worker, weights, export |
| `w1a1_eagle/checkpoints/` | The state store |
| `tools/` | Checks, doctor and the metadata snapshot tool |
| `policy/` | Rule registry, boundaries, commands, pins, exceptions, evidence registry |
| `tests/` | CPU tests; `tests/numerical/` runs in the numerical interpreter |
| `tests/cases/` | One bad and one control specimen per rule |
| `evidence/` | Dated records backing findings |
| `tasks/` | Open multi-session work; context, resumed only when asked |

## Boundaries

From [boundaries](../policy/boundaries.toml), enforced by the architecture stage:

- `w1a1_eagle` never imports `tools`, `tests` or `benchmarks`; `tools` never imports `tests`.
- `w1a1_eagle.contracts` and `tools.contracts` use only the standard library.
- `w1a1_eagle.model`, `w1a1_eagle.training` and the contracts do no file or process IO.
- Only the owners below write durable state.

| Owner | State |
| --- | --- |
| `w1a1_eagle.data` | Prompt roles and consumption |
| `w1a1_eagle.checkpoints.store` | Tensor states, manifests, latest pointer |
| `w1a1_eagle.lifecycle` | Event journal and controller identity |

Environment variables are read only in `w1a1_eagle/config.py` and `tools/config.py`.
Operational time comes from `w1a1_eagle/clock.py`.

## Entries and flows

| Flow | State | Path | Symbol | Trigger |
| --- | --- | --- | --- | --- |
| setup | active | pyproject.toml | check | `uv sync --locked --only-group check --no-build` |
| check | active | tools/check.py | main | Developer; `--full` from `.githooks/pre-commit` |
| doctor | active | tools/doctor.py | diagnose | Developer |
| selftest | active | tools/self_test.py | run_corpus | Proof stage and `--self-test` |
| policy | active | tools/policy_review.py | verification_report | Passing full check |
| cli | active | w1a1_eagle/__main__.py | main | `python -m w1a1_eagle` |
| snapshot | active | tools/snapshot_metadata.py | pack | `python -m tools.snapshot_metadata` |
| admission | active | w1a1_eagle/index.py | IndexReader | Preflight and run |
| sampling | active | w1a1_eagle/numerical/corpus.py | Corpus | Worker |
| supervision | active | w1a1_eagle/supervisor.py | execute | Run |
| restore | active | w1a1_eagle/numerical/worker.py | restore | Resume and branch |
| export | active | w1a1_eagle/numerical/export.py | export_he4 | End of run |
| journal | active | w1a1_eagle/lifecycle.py | Journal | Every run event |

Public commands are declared in [commands](../policy/commands.toml).

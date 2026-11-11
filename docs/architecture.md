# Architecture

## Layout

| Path | Holds |
| --- | --- |
| `w1a1_eagle/` | Product: data admission, intent, supervision, lifecycle, CPU/CUDA training |
| `w1a1_eagle/numerical/` | Torch and GGUF code: captures, corpus, worker, weights, export |
| `w1a1_eagle/checkpoints/` | The state store |
| `tools/` | Checks, doctor and the metadata snapshot tool |
| `policy/` | Rule registry, boundaries, commands, pins, exceptions, evidence registry |
| `tests/` | CPU tests; `tests/numerical/` runs in the numerical interpreter |
| `tests/cases/` | One bad and one control specimen per rule |
| `results/` | Durable conclusions and small supporting data |
| `evidence/` | Existing immutable execution and provenance receipts |
| `skills/close-results/` | Reusable investigation closeout procedure |
| `native/` | Authored tokenizer and capture helpers for the pinned runtime |
| `benchmarks/` | Frozen-source native comparisons, live tools, harness panels and campaign coordination |
| `typings/` | Minimal typed interfaces for untyped external APIs |
| `tasks/` | Open multi-session work; context, resumed only when asked |

The [model](model.md) describes the current numerical graph and trainable state.
[Research context](research.md) explains the development direction and historical
sources. The [support boundary](support.md) distinguishes implemented behavior from unqualified
execution paths.

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
Operational time comes from `w1a1_eagle/clock.py`; benchmark request timing is owned
by `benchmarks/measurement.py`.

## Entries and flows

| Flow | State | Path | Symbol | Trigger |
| --- | --- | --- | --- | --- |
| setup | active | pyproject.toml | check | `uv sync --locked --only-group check --no-build` |
| check | active | tools/check.py | main | Developer; `--full` from `.githooks/pre-commit` |
| doctor | active | tools/doctor.py | diagnose | Developer |
| selftest | active | tools/self_test.py | run_corpus | Proof stage and `--self-test` |
| policy | active | tools/policy_review.py | verification_report | Passing full check |
| cli | active | w1a1_eagle/__main__.py | main | `python -m w1a1_eagle` |
| benchmark | active | benchmarks/__main__.py | main | `python -m benchmarks` |
| conversion | active | w1a1_eagle/numerical/conversion.py | main | Explicit bounded feature conversion |
| precision | active | w1a1_eagle/numerical/precision.py | main | Explicit CPU/CUDA screen |
| snapshot | active | tools/snapshot_metadata.py | pack | `python -m tools.snapshot_metadata` |
| admission | active | w1a1_eagle/index.py | IndexReader | Preflight and run |
| sampling | active | w1a1_eagle/numerical/corpus.py | Corpus | Worker |
| supervision | active | w1a1_eagle/supervisor.py | execute | Run |
| restore | active | w1a1_eagle/numerical/worker.py | restore | Resume and branch |
| export | active | w1a1_eagle/numerical/export.py | export_he4 | End of run |
| journal | active | w1a1_eagle/lifecycle.py | Journal | Every run event |

Public commands are declared in [commands](../policy/commands.toml).

## Benchmark metadata interfaces

The `benchmarks` package implements read-only planning, artifact authentication and
structural trace reduction; see [benchmark tools](benchmarks.md). It depends on product
contracts; product code never imports it. The request runner launches a pinned native server under an exclusive GPU lease, authenticates
its inputs and preserves every request/response. Live workflow and sampled verification
qualification remain tracked in the implementation task.

## Reserved benchmark interfaces

The request runner is implemented; a model plug-in registry is not. Some validators
reserve future software interfaces: extension filenames ending in `_adapter.py`,
`_workload.py` or `_dataset.py`, unique extension IDs and declared protocol/schema/digest
metadata. The source checker also rejects conditional or lookup-table dispatch on
adapter, dataset or workload IDs outside a discovery module.

These restrictions concern software organization. They do not prescribe a new neural
network. For example, they could prevent a future benchmark from choosing a runner
through a simple branch, forcing a discovery mechanism before that mechanism has been
designed. Revisit these interface guards when implementing the benchmark design. Current
data-role and comparison protections apply independently.

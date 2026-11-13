# Development

This guide owns environment installation and the development workflow. The
[architecture map](architecture.md) locates source owners and public entries;
[checks](checks.md#choosing-verification) selects validation lanes. Read the relevant
data, execution or benchmark guide for the change. [FINDINGS](../FINDINGS.md) leads to
prior conclusions and their evidence; historical source recovery is in [evidence](evidence.md).

## Setup

Requires CPython 3.11.15 and uv 0.11.32 on Linux x86-64. Exact archive identities are
in [toolchain](../policy/toolchain.toml); `pyproject.toml` and `uv.lock` pin dependencies.
Install the check environment from wheels only and enable the local commit gate:

```sh
uv sync --locked --only-group check --no-build
git config core.hooksPath .githooks
```

The hook runs the full check before each commit. Confirm the environment and hook with:

```sh
uv run --locked --no-sync --offline python -m tools.doctor
```

The check environment alone does not provide the numerical interpreter needed by the
full gate or training. Install it separately from the same lock, using a fresh path so
stale `.pth` files cannot import code from another checkout:

```sh
UV_PROJECT_ENVIRONMENT=.venv/numerical-clean uv sync --locked --no-default-groups --extra numerical
```

Set `BINARY_NUMERICAL_PYTHON` to that environment's absolute interpreter path, or create
the ignored, host-only `.host.toml` in the checkout root:

```toml
numerical_python = "/absolute/path/to/this/repository/.venv/numerical-clean/bin/python"
```

Keep credentials out of `.host.toml`. The full gate hides GPU visibility; CPU coverage
is defined in [numerical policy](../policy/numerical.toml), while hardware support and
measured limits belong to [support](support.md).

Environment installation does not supply model weights, captured datasets or a compiled
native runtime. The [data](data.md) and [benchmark](benchmarks.md) guides define their
inputs and preparation. Keep project assets under ignored `artifacts/`, following the
[storage map](storage.md).

## Validation and entry points

```sh
uv run --locked --no-sync --offline python tools/check.py
uv run --locked --no-sync --offline python tools/check.py --full
uv run --locked --no-sync --offline python -m w1a1_eagle preflight INTENT
```

The first check is partial; the second is the full commit gate. [Checks](checks.md#commands)
owns the lanes, coverage and repair guidance. Training preflight checks the supplied
intent and data on CPU before allocation; [execution](execution.md) defines run, resume
and reporting commands.

For training with periodic acceptance checks, use `python -m benchmarks campaign-preflight
SPEC` before `campaign SPEC [OUTPUT]`. [Benchmarks](benchmarks.md) owns the 60-prompt candidate
checks, saved baseline/leader references and interruption budgets. The
[qualification result](../results/reusable-acceptance-checks/result.md) records measured
costs and limits. Agent instructions start in [AGENTS.md](../AGENTS.md); task conventions
and closeout procedures are linked below.

## Repository interfaces

- Public entries are declared in `policy/commands.toml` and mapped in
  [architecture](architecture.md#entries-and-flows).
- `pyproject.toml` and `uv.lock` own dependency versions. [Setup](#setup) owns environment
  installation and host interpreter selection; [checks](checks.md#commands) owns validation commands.
- Checker rules, rejecting/accepting cases and required-test behavior are documented
  in [checks](checks.md#adding-things). A new file or ordinary test needs no rule-registry edit.
- Unfinished multi-session work uses [tasks](../tasks/README.md); durable research
  conclusions use the [result format](results.md).

## Experiments

Use maintained [benchmark](benchmarks.md) and [training](execution.md) entries. The task
records model/data identities, the question, allocation and stopping conditions. Model
or data-role changes and new training allocations require an authorized scientific scope.
Keep recovery state, provisional acceptance leader and full-comparison incumbent separate.
Frozen execution source and owner/birth/boot checks govern recovery, not the current checkout.
Persistent launch, progress inspection and pause budgets are specified in the benchmark guide.

The shared `benchmark-design` skill is for measurement design or repair. Ordinary candidate
runs use the existing commands and reference contracts. `correct` handles recurring agent
or workflow mistakes; isolated product fixes use their maintained commands and relevant tests.

## Investigation closeout

Follow the pinned [close-results procedure](../skills/close-results/SKILL.md) and
[project specification](results.md). This project copy governs even when a shared version
is installed; the link makes discovery independent of automatic skill loading.

The specification defines extraction of compact measurements, figure inputs and source
identities, result sections, FINDINGS updates and explicit artifact disposition. The
[retirement command](results.md#retirement-command) verifies retained evidence and exact
removal paths before authorized cleanup. Needed data/model inputs remain in ignored `artifacts/`;
discarded states lose exact resume/replay. Routine repairs normally need no result page.

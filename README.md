# BitDraft

BitDraft investigates whether a small, low-precision drafter can make a larger
language model answer faster. The drafter proposes several tokens; the target model
verifies them. Success means faster complete requests with the target's output
semantics preserved. Fast draft kernels or good training loss alone do not establish it.

The current development candidate is HE4 for a Qwen3.5-4B Q4_K_M target. Its
feed-forward weights use one bit, gate/up inputs use four bits, and down inputs use
eight bits. Other components retain their declared formats; it is not a whole-model
one-bit system. The [model](docs/model.md) explains what is trained and frozen.

The public pipeline covers data admission, CUDA training with optimizer recovery, GGUF
export, native request comparisons, tool workflows and periodic acceptance checks.
[Benchmarks](docs/benchmarks.md) gives entry points and workload contracts;
[support](docs/support.md) distinguishes measured coverage from remaining limits.
Released DFlash Q4KM remains the external comparator. Read [findings](FINDINGS.md)
before interpreting a candidate score as a serving improvement.

## Project storage

Datasets, models, checkpoints, benchmark records, archived evidence and project caches
live inside this checkout under ignored `artifacts/`. [Storage](docs/storage.md) maps the
folders and commands. Omitted run outputs use the appropriate folder automatically;
explicit output paths must remain inside `artifacts/`. Git tracks source and compact
results, so a Git backup alone does not preserve these artifact bytes.

## Setup

Requires CPython 3.11.15 and uv 0.11.32 on Linux x86_64; exact archive identities are in
[toolchain](policy/toolchain.toml). Install the check environment from wheels only:

```sh
uv sync --locked --only-group check --no-build
git config core.hooksPath .githooks
```

The second line enables the pre-commit hook, which runs the full check. Run
`uv run --locked --no-sync --offline python -m tools.doctor` to confirm the hook
is configured and executable. The check-only install does not provide the numerical
interpreter required by the full gate.

Numerical code runs in a separate environment installed from the same lock. Use a fresh
path so an old environment's `.pth` files cannot import packages from another checkout:

```sh
UV_PROJECT_ENVIRONMENT=.venv/numerical-clean uv sync --locked --no-default-groups --extra numerical
```

Set `BINARY_NUMERICAL_PYTHON`, or set the host-only `.host.toml` to its absolute interpreter:

```toml
numerical_python = "/absolute/path/to/this/repository/.venv/numerical-clean/bin/python"
```

The full gate disables GPU visibility. Versions live in [pyproject.toml](pyproject.toml)
and the lock; CPU fixture coverage lives in [numerical policy](policy/numerical.toml).
Keep credentials out of `.host.toml`. [Checks](docs/checks.md#commands) explains each lane.

## Commands

```sh
uv run --locked --no-sync --offline python tools/check.py
uv run --locked --no-sync --offline python tools/check.py --full
uv run --locked --no-sync --offline python -m tools.doctor
uv run --locked --no-sync --offline python -m w1a1_eagle preflight INTENT
```

The first is a quick partial check, the second is the full commit gate, the third
inspects the environment and hook. The last family runs and reports CPU/CUDA training;
see [execution](docs/execution.md).
For intermittent acceptance checks, use `python -m benchmarks campaign-preflight SPEC`
before `campaign SPEC OUTPUT`. The [benchmark guide](docs/benchmarks.md) explains60-prompt
candidate checks, reusable baseline/leader records and complete interruption budgets.
The [qualification result](results/reusable-acceptance-checks/result.md) records measured
pause/comparison costs, exact recovery and the limits of historical comparisons.

## More

- [Architecture](docs/architecture.md), [development](docs/development.md), [checks](docs/checks.md), [data](docs/data.md),
  [domain terms and metrics](docs/domain.md), [model](docs/model.md),
  [research context and source guide](docs/research.md)
- [Findings](FINDINGS.md): short result index; [result format and closeout](docs/results.md)
- [Supported behavior and limits](docs/support.md), [evidence access](docs/evidence.md)
- [Task convention](tasks/README.md); [close-results skill](skills/close-results/SKILL.md)
- License: not yet chosen.

- [Benchmarks](docs/benchmarks.md): prompt preparation, capture, native comparisons and training campaigns.

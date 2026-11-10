# Binary Eagle source port

The pre-benchmark CPU source path covers retained metadata admission, HE4 numerical
updates, transactional states, exact own-state resume/branch, bounded supervision
and verified GGUF export. End-to-end fixtures use retained source/intent snapshots.
Fresh-context readiness, routine repair and failure-diagnosis exercises are recorded
in [findings](FINDINGS.md); they establish no causal productivity gain.
Actual retained-artifact parity and native qualification remain pending. Benchmark redesign and high-level agent operations remain deferred.
This is a local repository with no remote; preserve the separate active GPU job.

[Agent first port overhaul](docs/port-overhaul.md) describes the authorized implementation and its remaining gates.

HE4 means W1A4 with A8 down projection and Q4K head. W1A8 is historical DSpark W1A8.
DSpark Q4KM is the historical baseline. Baseline means current external DFlash Q4KM drafter.
These identities are separate; the evidence does not establish HE4 beating Baseline.

## Setup

Use CPython 3.11.15 and uv 0.11.32 authenticated against the exact archive identities
in [toolchain](policy/toolchain.toml). Check disk first. This stage has a hard combined
1 GiB write cap and requires at least 2.5 GiB WSL space after writes. Install wheels only:

```sh
uv sync --locked --only-group check --no-build
```

The numerical profile uses a separate interpreter selected by `BINARY_NUMERICAL_PYTHON`,
local `.host.toml`, or `.venv/numerical/bin/python`. It requires the exact top-level
versions in [numerical policy](policy/numerical.toml). This host reuses existing Torch
libraries and an isolated pinned GGUF wheel; this is not a fresh hermetic environment
or native qualification. CPU numerical subprocesses mask CUDA devices.

Run from this repository root:

```sh
uv run --locked --no-sync --offline python tools/check.py
```

The command installs nothing, downloads nothing, formats nothing and uses no GPU.
Its ordinary run includes all specimens. `--self-test` shows stage 9 with bootstrap prerequisites;
it is an inspection convenience, never a reduced CI gate.

Use [CPU execution](docs/execution.md) for the public run/intent contract and doctor.

See [checks](docs/checks.md), [system map](docs/system-map.md), [findings](FINDINGS.md),
and [decisions](docs/decisions/README.md). License remains undecided. Hosted CI, the
optional image and authenticated policy review have not run. Native/runtime verify,
source parity and benchmark qualification are unavailable until separately approved.

# Binary Eagle foundation

Stage 0 contains CPU policy checks, synthetic contracts and historical evidence.
It contains no model, trainer, capture loader, ledger importer, native build or benchmark.
No GPU work or real-data validation is authorized. This is a local temporary repo with no remote.

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

Run from this repository root:

```sh
uv run --locked --no-sync --offline python tools/check.py
```

The command installs nothing, downloads nothing, formats nothing and uses no GPU.
Its ordinary run includes all specimens. `--self-test` shows stage 9 with bootstrap prerequisites;
it is an inspection convenience, never a reduced CI gate.

See [checks](docs/checks.md), [system map](docs/system-map.md), [findings](FINDINGS.md),
and [decisions](docs/decisions/README.md). License remains undecided. Hosted CI, the
optional image and authenticated policy review have not run. Native/runtime verify,
source parity and benchmark qualification are unavailable until separately approved.

# BitDraft

Trains and exports HE4 drafters: W1A4 weights, A8 down projection, Q4K head.
heldout24 and its aliases never enter training, calibration or selection.
[Domain terms](docs/domain.md) defines Baseline, candidates, formats and metrics.

Checkout: `/home/philip/BitDraft` on the RTX5080 host.
All project artifacts live in ignored `artifacts/`; [storage](docs/storage.md) maps the folders.
Remote: `git@github.com:philippesic/BitDraft.git` (`origin`). The license remains undecided.

## Navigation

- [README](README.md): setup and introduction. [Architecture](docs/architecture.md): source owners and entries.
- [Development](docs/development.md): change/validation route. [Checks](docs/checks.md): commands and repairs.
- [Data](docs/data.md): prompt roles, capture formats and corpus admission.
- [Execution](docs/execution.md): training, ownership, state/resume and clocks.
- [Benchmarks](docs/benchmarks.md): candidate acceptance checks, broader comparisons,
  saved baseline/leader records and workload contracts.
- [FINDINGS](FINDINGS.md): short conclusion index; linked results contain the evidence.
- [Tasks](tasks/README.md): unfinished-work convention. [Evidence](docs/evidence.md): archive recovery.

## Code and checks

- Strict typing; product code never imports `tools`, `tests` or `benchmarks`.
  Module owners and import boundaries are in [architecture](docs/architecture.md).
- The comment/docstring ban is enforced here. Necessary machine directives require
  exact entries in `policy/exceptions.toml`; [checks](docs/checks.md#comment-ban) owns the rules.
- Quick partial check: `uv run --locked --no-sync --offline python tools/check.py`.
- Commit gate: add `--full`; enable it with `git config core.hooksPath .githooks`.
  It includes CPU numerical fixtures and rule proofs, not GPU qualification.
- No hosted CI. Use the local hook and the [verification lanes](docs/checks.md#choosing-verification).

## Experiments and closeout

- GPU work follows the recorded task allocation. Inspect actual GPU clients and
  recorded process owners before starting or recovering work; [execution](docs/execution.md)
  and [benchmarks](docs/benchmarks.md) define the lease, source and recovery rules.
- Keep recovery state, provisional acceptance leader and full-comparison incumbent
  distinct. Use authenticated saved controls; exact comparisons and limits belong
  to [benchmarks](docs/benchmarks.md), measured support to [support](docs/support.md).
- For an investigation, follow [results and retention](docs/results.md) and the pinned
  [close-results procedure](skills/close-results/SKILL.md). Extract and verify compact
  measurements, conclusions and source identities; update FINDINGS and current docs.
- Every result states `Artifact disposition`. Retain needed data/model inputs and
  selected/recovery state in `artifacts/`; retire obsolete outputs through the
  [retirement command](docs/results.md#retirement-command) within authorized scope.
  Training states use their [Store retention](docs/execution.md#state-resume-and-branch) owner.
  Record lost exact replay/resume; a summary cannot replace an input still needed.
- `docs/` describes the present, `tasks/` unfinished work and `results/` durable learning.
  `evidence/` retains existing immutable receipts; retained raw records stay unchanged.

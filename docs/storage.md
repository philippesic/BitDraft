# Project storage

All BitDraft-owned files live in this checkout. Source, instructions, benchmark definitions
and compact results are tracked by Git. Large inputs, outputs, archived evidence and
project caches live under ignored `artifacts/`; ignoring them does not make them disposable.
Shared host tools, drivers, global agent policy and the machine GPU lease are host resources.

## Artifact map

| Location | Contents |
| --- | --- |
| `artifacts/datasets/` | Original teacher corpus, supplemental captures and the complete 6K FP16 corpus |
| `artifacts/models/` | Target/drafter inputs, vocabulary and reusable model assets |
| `artifacts/training/` | Training specifications, campaign runs and health records |
| `artifacts/benchmarks/` | Comparison definitions, workloads, results and saved reference registry |
| `artifacts/qualification/` | Measured qualification runs, role-safe DEV inputs and retained selected states |
| `artifacts/native/` | Pinned runtime sources, headers, libraries, binaries and helpers |
| `artifacts/bank/`, `artifacts/prompt-sources/` | Prompt banks, selected IDs, source text and source provenance |
| `artifacts/migration/` | Original imported HE4broad parent and its import receipts |
| `artifacts/archive/` | Historical Git readers/bundles, compact supporting records and preserved legacy material |
| `artifacts/receipts/` | Operational relocation, verification and retirement receipts |
| `artifacts/cache/`, `.venv/` | Project scratch/cache data and local Python environments |

The complete new corpus is `artifacts/datasets/train-bank-v8-fp16-01/corpus.json`.
Its reused 2K component lives at `artifacts/datasets/he4broad-supplement-01` and its remaining
Linux/Windows-produced records at `linux-captures/` and `windows-captures/` under the full
corpus folder. All three are physical project-local storage. The original 15,647-record
corpus is `artifacts/datasets/retained-he4broad-original`; its original final 220095 archive
and `artifacts/migration/import-01/states/` parent remain distinct from probe descendants.
The TRAIN expansion lives in one folder, `artifacts/datasets/train-expansion-01/`: read-only
public `sources/`, runtime-v6 `native-helpers/` with the span-capable capture helper,
`specs/`, session transcripts, the selected expansion bank, capture shards, the mixture
manifest with the 6K corpus, `logs/` and the `executed-sources/` that ran them.
Use [data](data.md) for roles/identity and [execution](execution.md) for state restoration.

## New output locations

Public execution commands accept an optional output. When omitted, they generate a unique
folder in the matching category: training campaigns in `artifacts/training`, captures and
conversions in `artifacts/datasets`, native builds in `artifacts/native`, prompt preparation
in `artifacts/bank`, and comparisons in `artifacts/benchmarks`. `record-reference` defaults
to the shared `artifacts/benchmarks/references` registry so observations deduplicate.
Records authenticate their source runs by absolute path, so those runs never move while
referenced. The current protocol's runs, establishment and candidate-only specifications
are in `artifacts/benchmarks/runtime-v7-depth3/` ([current protocol](benchmarks.md#current-protocol)).
The previous protocol's are in the registry's own `runtime-v6-depth3/` subfolder, where
they were first written; new runs go directly under `artifacts/benchmarks/`.
Outputs land in the checkout whose code runs: a worktree writes its own `artifacts/`, and
its `/home/philip/BitDraft` paths resolve there. Freezing branch source under this
checkout's `artifacts/` and running the snapshot with `--frozen` writes here instead.

A plain relative output name is placed in that category. An explicit path beginning with
`artifacts/`, or an absolute path within this checkout's `artifacts/`, selects its exact
location. Public commands reject outputs that escape it, including through symlinks.
Stage retained input downloads here too; do not introduce a persistent project dependency
on another checkout or an unrelated home/Windows folder. CPU runtime scratch is under
`artifacts/cache/tmp`; check scratch is under `.venv/check-temp`.

## Historical references

Immutable capture manifests, source identities and benchmark receipts retain their original
bytes, including old absolute path strings. `w1a1_eagle/storage.py` translates those known
legacy roots at the IO boundary to the corresponding location in this checkout. Mount
logical names and scientific identities are preserved. The rules travel with newly frozen source. Old execution records remain historical;
current readers translate their locators without changing their recorded bytes.
Archived absolute locators are historical references, not instructions to recreate external
storage. Current guides and newly emitted references use the project-local locations.

The old 5080 checkout and its `legacy-checkout-pending-delete` quarantine have been removed.
Historical source, useful measurements and unpublished edits remain independently archived;
current training and capture use the artifact locations above, without an old checkout.

Historical Windows runs and transfer deliveries have compact report, measurement,
diagnostic and source support under `artifacts/archive/development-history/`.
The old lossless raw-output archives are retired after member authentication and nested
archive inspection. Current Windows-produced TRAIN data remains physical data under
`datasets/`; it does not depend on historical archive extraction. See
[historical evidence access](evidence.md#windows-artifact-archives) and the
[retirement result](../results/historical-output-retirement/result.md).

Qualification storage retains the diagnostic DEV corpus, reusable initialization,
corrected original state/export, current comparison exports and authenticated saved-control
dependencies. Closed pilot checkpoints, unused exports, capture fixtures and raw profiles
are retired after compact evidence preservation. A completed historical pilot's selected
state is not an active production recovery requirement.

## Retention and moving the project

Follow [results and retention](results.md): keep working inputs and selected/recovery state,
extract verified compact findings and retire authorized obsolete outputs. Never delete
`artifacts/` merely because Git ignores it. Git preserves source and compact findings;
back up artifact bytes separately when they are needed. Move the whole checkout and use
its current artifact paths; the maintained readers locate the owning checkout rather than
requiring the old external directories. Host interpreter settings in `.host.toml` may need
to be updated after a checkout move.

For Windows storage audits, enumerate from an extended path such as `\\?\C:\...` so
deep descendants are included. Require the physical source folder to contain no files
or unrecorded links before removing it; a complete initial inventory alone is insufficient.

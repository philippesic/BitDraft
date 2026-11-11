# Training execution

## Commands

```sh
python -m w1a1_eagle preflight INTENT
python -m w1a1_eagle run INTENT OUTPUT --scope TASK
python -m w1a1_eagle report OUTPUT
```

`--json` before or after the subcommand prints JSON records instead of text; both forms
describe the same records. A run prints its admission before allocating anything and a
terminal report at the end. `report` only reads existing records. The bootstrap and
numerical worker are private entries listed in [commands](../policy/commands.toml).

## Intent

An intent (schema 2, validated by `Intent.load` in `w1a1_eagle/intent.py`) names the
objective, hypothesis, decision, risk and authorized scope; source, Q4_K vocabulary and
corpus artifacts; the mode (fresh, resume, continue or branch) and parent; geometry, seed and CPU/CUDA
device; update, attempt and cumulative wall limits; memory, state-size and disk limits;
checkpoint cadence; and the minimum-data and maximum-reuse policy. Policy values are
inputs, not recommendations. Unknown fields or schema versions are refused.

Artifact references are `path`, `sha256` and `bytes`; relative paths resolve from the
containing document. A corpus (schema 1) has `target_sha256`, `protected`,
`expected_train` and `records`. An optional `input_adapter` selects the modern index
instead; see [data](data.md). An editable manifest cannot establish provenance or
authorize a run.

## Owners

- `w1a1_eagle/checkpoints/store.py` owns tensor states, manifests and the latest pointer.
- `w1a1_eagle/lifecycle.py` owns the event journal and controller identity.
- `w1a1_eagle/data.py` owns roles and consumption.

Scratch files and read-only diagnostics need no owner. One controller at a time may
write an output root; a second one is refused. This lock covers the output root only,
and CUDA workers also obtain a host-wide exclusive GPU lease after checking active compute processes.

## Attempts and cost

Each attempt retains a copy of its source under `sources/<identity>` and its intent under
`intents/<identity>`, and runs from those copies, so later edits to the checkout or
intent do not change it. Loaded Python and native library identities are recorded at
start and checked again at the end.

A bootstrap arms an OS wall timer before importing Torch, so the time limit holds even
without a supervising agent. The controller runs the worker in its own process group
and identifies it by boot and process start identity. CPU workers hide CUDA; CUDA workers use the authorized visible device under their GPU lease.

Failed attempts keep their cost. There is no automatic retry. Before initialization, a
fresh retry with the same recipe may reuse the root and keeps the failed preflight cost.
After initialization, recovery needs an explicit parent and resume. If a controller is
lost, live owned work is refused; once the owner is gone, recovery charges the whole
remaining wall budget rather than guessing elapsed time.

## State, resume and branch

Every entered update is journaled before sampling, and exposure is journaled before the
update. Complete states are published through checksummed atomic manifests: writes go to
a temporary file on the same filesystem, are synced, authenticated and then the latest
pointer is replaced. A crash before publication keeps the parent; a crash after keeps a
complete parent or a complete child. This covers process crashes, not power loss.

- Fresh: proves the original authenticated initialization and no optimizer history.
- Resume: restores parameters, shared aliases, AdamW counters and moments, CPU RNG,
  sampler cursor, aggregate exposure counters and lineage, and must reproduce the next
  update exactly.
- Continue: changes recipe/data explicitly while retaining Adam, random state, cumulative
  clocks and parent lineage. A changed dataset starts a new dataset exposure counter;
  same-dataset continuation preserves it. Sampling restarts at cursor zero.
- Branch: resets optimizer and cursor but keeps parent lineage, cost and aggregate
  exposure counters. A changed recipe does not preserve per-position lifetime exposure.

No mode silently loads an old trained state. Without a retention declaration, state history is retained. An explicit `retention`
object bounds states/exports; pruning preserves latest and an in-store parent and leaves
compact retirement receipts. Cross-run selected states need separate retention. Each state write rechecks free space; this does not reserve capacity.

Tensor payload 4 stores learned parameters, Adam, RNG and counters plus a fingerprint of
every frozen buffer, dataset identity/exposure origin, observed labels and separate
successful-update, failed-update and data clocks. Payload 3 remains readable with unknown
new accounting fields; payloads 1/2 are refused. Frozen data is rebuilt from the required initializer and vocabulary
and refused on mismatch. Buffer guards catch ordinary in-place mutation before
publication; they are not a sandbox.

## Sampling and clocks

Optional `sampling` names an authenticated artifact with schema 1, policy
`task-position-v1`, locality `burst` (1–64), positive `weights` and `strata` mapping every
admitted task ID to a stratum. Sampling chooses a stratum, then a task uniformly, then an
eligible position. A burst retains the task while varying positions. This samples with
replacement; nominal window-equivalents are not completed epochs. Omitting it retains
the original seeded window permutation. The recipe binds the sampling artifact.

Successful model-update time starts after CPU capture materialization and includes input
transfer, forward/backward, optimizer and validation, with CUDA synchronization at both
boundaries. Data time covers sampling/materialization. Checkpoint/export/evaluation and
startup are separate from that clock; controller attempt time remains an operational
budget. Imported historical clocks stay in their receipts and must not be mixed with
new measurements. Admission adds known dataset exposure to planned upper bounds; it does
not reinterpret all historical updates as updates on a newly supplied corpus.

Optional `retention: {states: 3, exports: 2}` bounds payload retention and admission's
state-capacity estimate. It does not authorize deleting external inputs or selected states.

## Export

A run exports its final complete state to GGUF (native-format W1Ax version 2). The
exporter streams through the store under its byte budget, keeps the original Q4_K
vocabulary codes, checks frozen operands against export operands, and reads every tensor
back before publishing. Sign bits are little-endian 32-bit words, nonnegative zero is 1,
and unused tail bits are 0. This checks the format, not native loading.

## Journal

The journal is limited to 64 MiB, and execution stops at 56 MiB to leave room for the
final state, export and terminal records. It never deletes events to continue. That stop
is a storage limit, not completion. A torn final append is reconciled only after checking
ownership: its bytes are saved in an incident receipt, the intact prefix is validated and
no complete event is dropped. Corrupt complete events are refused.

## Supported formats

Intent 2, corpus 1, tensor payload 4 (reader also accepts 3), store manifest 1, event envelope 1 and GGUF W1Ax 2.
Earlier tensor schemas 1 and 2 and intent 1 have no reader. Migrating one needs a new
artifact with verified source lineage; keep the originals.

## Limits

These are CPU constructions with reduced geometry. They establish construction,
accounting and recovery behavior, not full-size parity with retained artifacts, native
correctness, useful learning or serving performance. The terminal report names the
controller outcome, worker stop, confirmed activation, starting state and risk, and marks
missing evidence as unknown.

# CPU execution

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
corpus artifacts; the mode (fresh, resume or branch) and parent; geometry, seed and CPU
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
not the GPU or other projects.

## Attempts and cost

Each attempt retains a copy of its source under `sources/<identity>` and its intent under
`intents/<identity>`, and runs from those copies, so later edits to the checkout or
intent do not change it. Loaded Python and native library identities are recorded at
start and checked again at the end.

A bootstrap arms an OS wall timer before importing Torch, so the time limit holds even
without a supervising agent. The controller runs the worker in its own process group
and identifies it by boot and process start identity. CPU workers hide CUDA, and GPU
requests are refused.

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
  sampler cursor, exposure and lineage, and must reproduce the next update exactly.
- Branch: resets optimizer and cursor but keeps parent lineage, cost and exposure.

No mode silently loads an old trained state. The store never deletes historical states
to make room. Each state write rechecks free space; this does not reserve capacity.

Tensor payload 3 stores learned parameters, Adam, RNG and counters plus a fingerprint of
every frozen buffer. Frozen data is rebuilt from the required initializer and vocabulary
and refused on mismatch. Buffer guards catch ordinary in-place mutation before
publication; they are not a sandbox.

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

Intent 2, corpus 1, tensor payload 3, store manifest 1, event envelope 1 and GGUF W1Ax 2.
Earlier tensor schemas 1 and 2 and intent 1 have no reader. Migrating one needs a new
artifact with verified source lineage; keep the originals.

## Limits

These are CPU constructions with reduced geometry. They establish construction,
accounting and recovery behavior, not full-size parity with retained artifacts, native
correctness, useful learning or serving performance. The terminal report names the
controller outcome, worker stop, confirmed activation, starting state and risk, and marks
missing evidence as unknown.

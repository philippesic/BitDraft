# Training execution

Use [`benchmarks campaign`](benchmarks.md) for training with intermittent native
acceptance checks. It coordinates the supervisor below, keeping recovery, provisional
fast leader and full-DEV incumbent distinct. Loss/top1 diagnostics do not measure acceptance.
Campaign V2 executes60 candidate-only requests against authenticated saved controls;
`benchmarks campaign-preflight SPEC` enforces their existence and compatibility. The
provisional leader uses acceptance. Historical speed remains descriptive. See
[reference records and pause budgets](benchmarks.md#reusable-references).

## Commands

```sh
uv run --locked --no-sync --offline python -m w1a1_eagle preflight INTENT
uv run --locked --no-sync --offline python -m w1a1_eagle run INTENT OUTPUT --scope TASK
uv run --locked --no-sync --offline python -m w1a1_eagle report OUTPUT
```

`--json` before/after the subcommand selects structured output. Preflight checks inputs
on CPU; run prints admission before allocation and a terminal report afterward. Report
is read-only. Private entries are listed in [commands](../policy/commands.toml).

## Intent

Schema2 (`w1a1_eagle/intent.py`) binds objective/hypothesis/decision/risk/scope, authenticated
initializer/vocabulary/corpus/parent, mode, geometry/seed/device, budgets, checkpoints and
data/reuse policy. Unknown fields are refused. Artifact references contain `path`,
`sha256`, `bytes`; relative paths resolve from their document. See [data](data.md) for
corpus/index/mixture formats. Editable manifests establish neither provenance nor authority.

## Owners

Optional authenticated `initialization` binds materialized source/vocabulary tensors
and immutable native codes from `benchmarks prepare-initialization INTENT OUTPUT`.
Origin, geometry, coverage, dtype, finiteness and byte budgets are checked before use.
The reusable basis avoids repeated dequantization; its identity enters the recipe.
Frozen codes are bound to model construction and guarded against replacement. Exports
retain tensor readback. This is an external derived artifact, not a training capture.
Retain its producer evidence and qualify tensor/export/update parity when adopting it.

The [store](../w1a1_eagle/checkpoints/store.py) owns states/manifests; the
[journal](../w1a1_eagle/lifecycle.py) owns lifecycle/controller identity;
[data](../w1a1_eagle/data.py) owns roles/consumption. Output roots admit one controller.
CUDA also takes a host-wide lease after checking existing compute processes.

## Attempts and cost

Each attempt freezes source/intent and records loaded Python/native identities, checked
again at exit. Checkout edits cannot alter it. A bootstrap arms an OS timer before Torch;
the controller owns the worker process group by boot/start identity. CPU hides CUDA.
V2 campaigns keep one supervised worker through native checks. It releases completed
gradients, maps trainable tensors and Adam moments from its current durable checkpoint,
retains frozen GPU buffers, then restores devices and CPU/CUDA RNG before another update. Tensor identities
and mutation versions are checked. Native child servers die with their evaluator parent.
Training capture caches are released before evaluation; the next sample is materialized
before the resume clock closes. Acceptance selection remains pending until that pause
passes its budget. Recovering a final screen at the update guard does not repeat it.
The worker performs full data admission before model allocation; public CPU preflight
remains available. Journal-capacity rotation preserves global attempt/update/wall guards.

Failed attempts retain cost; there is no automatic retry. An uninitialized same-recipe
retry may reuse its root. After initialization, recovery needs explicit parent/resume.
Live owned work is refused. Lost-controller recovery charges the remaining wall budget
when its owner is gone rather than guessing elapsed time.

## State, resume and branch

Updates and exposure are journaled before execution. State publication uses a temporary
file, fsync, checksum and atomic latest-pointer replacement. Process crashes preserve a
complete parent or child; power-loss durability is not established.
Continuations retain their external durable parent rather than rewriting an identical
initial state in every segment. Fresh initialization still publishes its first state.

- Fresh authenticates initialization and has no optimizer history.
- Resume restores parameters/aliases, Adam moments/counters, RNG, cursor, clocks,
  exposure and lineage. Recipe/dataset must match; next-update parity is required.
- Continue explicitly changes recipe/data while retaining Adam/RNG/clocks/lineage.
  A changed dataset starts its own exposure counter; sampling restarts at cursor zero.
- Branch resets optimizer/cursor while retaining lineage/cost/aggregate exposure.
  Changed recipes do not provide a per-position lifetime ledger.

Tensor payload4 fingerprints frozen buffers and stores trainable/Adam/RNG state,
accounting and dataset exposure origin. Reader also accepts3 with unknown new accounting;
1/2 are refused. Frozen buffers rebuild from authenticated initializer/vocabulary and
must match. Mutation guards are not a sandbox. No mode silently imports trained weights.

Without `retention`, state history remains. Explicit states/exports bounds preserve latest
and an in-store parent; global campaign retention also protects selected/recovery states.
Retirement receipts remain. Capacity is rechecked on writes, not reserved in advance.

At task completion, follow [compact closeout and retirement](results.md#compact-evidence-is-the-default).
Keep only the assigned task's needed inputs, recovery/selected state and compact evidence;
record the explicit disposition and loss of exact replay for obsolete outputs.

Before cleanup, resolve the inputs and state references used by live jobs and selected
results. Preserve the teacher corpus, initialization/vocabulary, frozen executed source,
latest recoverable Adam/RNG state, selected states/exports and raw execution receipts
needed by the assigned task. A readable result or ignored path does not make these
bytes disposable. Closed unselected states can be retired only within authorized scope,
using Store retirement receipts before removal. Record when an exact retired resume
point is no longer available. Keep shared caches until their ownership and dependencies
are known; an upstream cleanup default is not evidence they are disposable.

## Sampling and clocks

Optional authenticated `sampling` uses schema1, `task-position-v1`, burst1–64, positive
stratum weights and an exact task-ID→stratum mapping. Choose stratum, uniform task,
uniform eligible position; bursts reuse the task with varying positions. Replacement
sampling gives window-equivalents, not completed epochs. Omission retains seeded window
permutation. Recipe identity binds the sampler.

Model-update time includes input transfer, forward/backward, optimizer and validation,
with CUDA synchronization. Successful/failed update clocks are separate. Data time covers
sampling/materialization. Startup, checkpoint, export, diagnostics and native evaluation
stay separate; controller wall time is an operational budget. Legacy clocks cannot be
reconstructed. Dataset exposure starts from known lineage rather than charging every
historical update to new data. Workers publish accounting every100 updates.

Optional `development` binds120 fixed DEV windows and cadence; diagnostics preserve RNG
and report category loss/top1. `model_update_seconds` bounds synchronized optimizer work.
In V2 campaigns it is the cumulative optimizer-time stop, independently of evaluation
cadence. Stop at a complete update, publish state/export and perform the final screen.
Resume retains the consumed clock; successful and failed update seconds both count.
See [benchmarks](benchmarks.md) for native cadence, selection and health reporting.

## Export

Final complete states export through the store to W1Ax metadata2 in GGUF3, with original Q4K vocabulary
codes, frozen-operand checks and tensor readback. Export-time exponentials use CPU FP32
to match original arithmetic. Sign words are little-endian32-bit, nonnegative zero is1,
unused bits0. Bounded readback checks metadata, tensor names/types/shapes, nonoverlapping
aligned offsets and every payload hash. Independent SDK and native checks qualify it;
format conformance alone does not establish useful serving performance.

## Journal

Events stop at56MiB of the64MiB envelope, leaving room for final publication. This storage
stop is not completion. A torn tail is saved in an incident receipt and reconciled only
after owner checks; no complete event is dropped. Corrupt complete records are refused.

## Supported formats

Intent2, corpus1, tensor payload4/3, store1, event envelope1, W1Ax metadata2/GGUF3. Other tensor/intent
schemas need explicit migration with verified lineage; preserve originals.

## Limits

CPU checks cover reduced fixtures. Separate [migration](../results/he4-checkpoint-migration/result.md)
and [FP16 qualification](../results/fp16-storage-adoption/result.md) establish scoped
full-size GPU behavior. Neither proves useful learning or serving gains. Reports distinguish
controller outcome, worker stop, activation and scientific evidence; unknown stays unknown.

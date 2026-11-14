# Training execution

Use [`benchmarks campaign`](benchmarks.md) for training with intermittent native
acceptance checks. It coordinates the supervisor below, keeping recovery, provisional
fast leader and full-DEV incumbent distinct. Loss/top1 diagnostics do not measure acceptance.
Campaign V2 executes 60 candidate-only requests against authenticated saved controls;
`benchmarks campaign-preflight SPEC` enforces their existence and compatibility. The
provisional leader changes only on an acceptance lead beyond the screen's paired
prompt-sampling interval ([selection](domain.md#selection)). Historical speed remains descriptive. See
[reference records and pause budgets](benchmarks.md#reusable-references).

## Commands

```sh
uv run --locked --no-sync --offline python -m w1a1_eagle preflight INTENT
uv run --locked --no-sync --offline python -m w1a1_eagle run INTENT [OUTPUT] --scope TASK
uv run --locked --no-sync --offline python -m w1a1_eagle report OUTPUT
```

`--json` before/after the subcommand selects structured output. Preflight checks inputs
on CPU; run prints admission before allocation and a terminal report afterward. Report
is read-only. Private entries are listed in [commands](../policy/commands.toml).

## Intent

Schema 2 (`w1a1_eagle/intent.py`) binds objective/hypothesis/decision/risk/scope, authenticated
initializer/vocabulary/corpus/parent, mode, geometry/seed/device, budgets, checkpoints,
grouping, optimization and data/reuse policy. Unknown fields are refused. Artifact references contain `path`,
`sha256`, `bytes`; relative paths resolve from their document. See [data](data.md) for
corpus/index/mixture formats. Editable manifests establish neither provenance nor authority.

## Owners

Optional authenticated `initialization` binds materialized source/vocabulary tensors
and immutable native codes from `benchmarks prepare-initialization INTENT OUTPUT`.
Origin, geometry, coverage, dtype, finiteness and byte budgets are checked before use.
The reusable basis avoids repeated dequantization; its identity enters the recipe.
Frozen codes are bound to model construction and guarded against replacement. Exports
retain tensor readback. This is a derived artifact under `artifacts/models/`, not a training capture.
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
Training capture caches and background loads are released before evaluation; the next sample is materialized
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
Continuations retain their separately stored durable parent rather than rewriting an identical
initial state in every segment. Fresh initialization still publishes its first state.

- Fresh authenticates initialization and has no optimizer history.
- Resume restores parameters/aliases, Adam moments/counters, RNG, cursor, clocks,
  exposure and lineage. Recipe/dataset must match; next-update parity is required.
- Continue explicitly changes recipe/data while retaining Adam/RNG/clocks/lineage.
  States written before grouped updates (one window per update, unfused Adam) still
  load for continue and branch; their recipe cannot be resumed exactly.
  A changed dataset starts its own exposure counter; sampling restarts at cursor zero.
- Branch resets optimizer/cursor while retaining lineage/cost/aggregate exposure.
  Changed recipes do not provide a per-position lifetime ledger.

Tensor payload 5 fingerprints frozen buffers and stores trainable/Adam/RNG state,
accounting, dataset exposure origin and the reach position weights in force (null for
decay). Reader also accepts 4 without position weights and 3 with unknown new accounting;
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

One update is one group of windows from one sequence ([model](model.md#one-optimizer-update)).
`updates` counts optimizer steps and `cursor` counts groups consumed since the current
sampling stream began; both advance by one per update. Labels and exposure are counted
per window. The exposure journal event lists each group's block size (`slots`), anchors
and per-window trained labels (at most `slots` − 1 each).

Optional `grouping` sets `max_windows` (1–64, default 32), `activation_bytes` (default
3 GiB) and `device_bytes` (default 14 GiB). Its effective values always enter recipe
identity. `device_bytes` caps the CUDA worker's memory: at the cap PyTorch releases cached
blocks before failing, so varying group shapes cannot grow cached memory into WSL's
slow shared-memory spill. Without the cap, 300 real updates reserved 16.8 GiB while
allocating only 12.5 GiB. A sequence's group limit is
the largest window count whose estimated activation memory at that sequence's largest
anchor fits `activation_bytes`, capped by `max_windows` and never below one. Memory and
time grow with windows × slots, so a block of k slots scales both the estimate and the cap:
the limit is computed for k-slot rows and capped at `max_windows` × 16 / k (128 windows for
four slots, 170 for three). The device cap is unchanged. The estimate
in `w1a1_eagle/training.py` was fitted on the RTX 5080: about 4.2 MB per slot row plus
0.9 KB per row per context token and 117 KB per context token, on top of about 9.7 GiB
of resident state and gradients. The default keeps measured peak allocation near 12.7 GiB, below the
cap and the roughly 14.3 GiB where WSL spills into slow shared memory. On the frozen bank it allows 32 windows up to about 1,600 context tokens,
20 at 4,100 and 4 at the longest sequence (13,470); see
[grouped updates](../results/grouped-updates/result.md).

Without `sampling`, each pass shuffles every sequence's windows with the seed, splits
them into near-equal groups within that sequence's limit, then shuffles the groups. Every
window appears once per pass; passes repeat the same order. With random block sizes the
same seeded generator draws each group's block size before cutting it: the group takes
an even share of the sequence's remaining windows under that size's limit. The plan,
including every group's size, depends only on the seed, so a cursor determines its
group and block size. Random block sizes cannot be combined with `sampling`.

Optional authenticated `sampling` uses schema 1, `task-position-v1`, burst 1–64, positive
stratum weights and an exact task-ID→stratum mapping. Each group chooses a stratum, a
uniform task and distinct uniform eligible positions; the group size is the smallest of
burst, the sequence's group limit and its eligible count, so the retained burst-8 hybrid
specification gives at most 8 windows per update. Groups are deterministic from seed and
cursor. Replacement sampling gives window-equivalents, not completed epochs.
Recipe identity binds the sampler. Admission bounds planned label exposures by
updates × the largest (k − 1) × group limit over the recipe's block sizes, so update
budgets and visit policies sized for one window per update need re-planning.

While an update runs, one background thread loads and authenticates the sequence of the
next cursor's group, which depends only on seed and cursor; four reader threads read and
hash its eight feature files in parallel. It writes no journal event and
changes no counter, cursor or RNG: the next update still journals its attempt and exposure
before receiving that sequence. A failed background load is discarded and the update loads
synchronously, so errors surface as they would without prefetching. Host memory holds at
most the current sequence, one upcoming sequence and that load's read buffer, all in storage
precision. When
training stops for checkpoint, export or evaluation the worker releases both, waiting for
a load in progress.

Model-update time includes input transfer (features move in storage precision and become
FP32 on the device), forward/backward, optimizer and validation,
with CUDA synchronization. Successful/failed update clocks are separate. Data time is how
long an update waits for its sample: locating the group, waiting for or performing its
sequence load and selecting windows; background loading during the previous update is
not counted in any clock. Startup, checkpoint, export, diagnostics and native evaluation
stay separate; controller wall time is an operational budget. Legacy clocks cannot be
reconstructed. Dataset exposure starts from known lineage rather than charging every
historical update to new data. Workers publish accounting every 100 updates, with
the training loss of every update since the previous record.

Optional `development` binds 120 fixed DEV windows and cadence; diagnostics preserve RNG
and report category loss/top1 over all fifteen positions. A run that ends also measures
its final update unless the cadence already did; a campaign's intermediate boundaries do not.
`positions` adds, for each of positions 1–15, the windows reaching it, top-1 hits, and
prefix survivors (positions 1 through i all correct) with their rates. Training
diagnostics use one sixteen-slot block; `Diagnostics.measure` takes another block size
for the depth+1 blocks a server builds. Teacher-forced top-1 is not native acceptance.
`model_update_seconds` bounds synchronized optimizer work.
In V2 campaigns it is the cumulative optimizer-time stop, independently of evaluation
cadence. Stop at a complete update, publish state/export and perform the final screen.
Resume retains the consumed clock; successful and failed update seconds both count.
See [benchmarks](benchmarks.md) for native cadence, selection and health reporting.

## Block sizes and position weights

Optional `blocks` requires exactly `slots` and `position_weights`; its effective values
always enter recipe identity, and the default is today's fixed sixteen slots with decay.

```json
"blocks": {
  "slots": {"3": 0.5, "4": 0.333333, "5": 0.25, "16": 0.066667},
  "position_weights": "reach"
}
```

- `slots` maps block sizes from 2 to 16 slots (1 to 15 drafted positions) to positive
  relative weights. One entry trains every update at that size; several entries draw a
  size per group as described under [sampling](#sampling-and-clocks). All windows of an
  update use the drawn size and train on their first k − 1 labels.
- `position_weights` is `decay` (exp(−i/7) for zero-based position i) or `reach`. Reach
  weights are proportional to how often each position is reached when serving: position 1
  has weight 1 and position i the drafter's measured prefix survival through position
  i − 1 on the DEV diagnostic windows, smoothed as (survivors + 1) / (windows + 2) so no
  position's weight reaches zero. `reach` requires `development`; the weights refresh at
  each cadence diagnostic, including the one at cursor 0 before the first update, but not
  at a run's final measurement. The diagnostic journal event records the weights it set.

Reach weights are part of the trained state: tensor payload 5 stores the weights in force,
and resume restores them. A resumed run that starts on a cadence boundary measures again
and obtains the same weights from the same parameters. Continue and branch restart the
cursor at zero, so their first diagnostic refreshes the weights for the new recipe.

## Optimization

Optional `optimization` requires exactly `step_sizes`, `schedule` and `latent_scale`; its
effective values always enter recipe identity, so a stated default has the same identity
as an omitted field. Without it training uses today's defaults.

```json
"optimization": {
  "step_sizes": {"sign": 0.001, "scale": 1e-05, "midpoint": 1e-05, "gain": 1e-05, "clip": 1e-05, "norm": 1e-05},
  "schedule": {"decay": "cosine", "warmup_updates": 156, "start_update": 0, "final_update": 7800},
  "latent_scale": {"kind": "row", "multiple": 3}
}
```

- `step_sizes` gives every AdamW group a positive base step size.
- `schedule` multiplies all of them by a pure function of the absolute update count `u`:
  (u − start + 1) / warmup during the first `warmup_updates` after `start_update`, then 1
  (`constant`, `final_update` null) or a cosine from 1 to 0 at `final_update` (`cosine`).
  Admission refuses a run that starts before `start_update` or whose update budget passes
  `final_update`. The defaults are `constant`, no warm-up, start 0.
- `latent_scale` is `{"kind": "raw"}` (default) or `{"kind": "row", "multiple": m}`: fresh
  initialization sets latents to clamp(weight / (m × row mean |weight|), -1, 1) and every
  update clamps them back into [-1, 1] ([model](model.md#what-changes-during-training)).
  Continue and branch keep the parent's latent values; under `row` only the clamp applies.

Because the multiplier depends only on `u` and the recipe, resume repeats it exactly. The
schedule never stretches implicitly: a continuation that changes the planned length is a
new recipe whose schedule is restated in absolute updates, and from its first update the
multiplier follows that new schedule (for example, restating a finished 7,800-update cosine
as ending at 15,600 resumes at the multiplier of update 7,800 on the longer curve, about 0.5).
Continue and branch accept changed step sizes; continue keeps the Adam moments.

## Export

Final complete states export through the store to W1Ax metadata 2 in GGUF 3, with original Q4K vocabulary
codes, frozen-operand checks and tensor readback. Export-time exponentials use CPU FP32
to match original arithmetic. Sign words are little-endian 32-bit, nonnegative zero is 1,
unused bits 0. Bounded readback checks metadata, tensor names/types/shapes, nonoverlapping
aligned offsets and every payload hash. Independent SDK and native checks qualify it;
format conformance alone does not establish useful serving performance.

## Journal

Events stop at 56 MiB of the 64 MiB envelope, leaving room for final publication. This storage
stop is not completion. A torn tail is saved in an incident receipt and reconciled only
after owner checks; no complete event is dropped. Corrupt complete records are refused.

## Supported formats

Intent 2, corpus 1, tensor payload 5/4/3, store 1, event envelope 1, W1Ax metadata 2/GGUF 3. Other tensor/intent
schemas need explicit migration with verified lineage; preserve originals.

## Limits

CPU checks cover reduced fixtures. Separate [migration](../results/he4-checkpoint-migration/result.md)
and [FP16 qualification](../results/fp16-storage-adoption/result.md) establish scoped
full-size GPU behavior. Neither proves useful learning or serving gains. Reports distinguish
controller outcome, worker stop, activation and scientific evidence; unknown stays unknown.

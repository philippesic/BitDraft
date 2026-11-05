# Precision campaign: execution checkpoint

## Authorization — November 4, 2026, approximately 22:07 local

User: “go ahead and run this until you produce the full comparison chart, should
be quite extensive.” After completing that comparison, if the user has not
returned (roughly 11am Pacific), choose further work within scope: additional
combinations, more overall or targeted training, or new features. Extra work is
conditional on completing the first comparison early. A user return/check-in is
steering, not automatic cancellation; follow explicit pause/stop instructions.

This authorizes executing [PRECISION_SEARCH.md](PRECISION_SEARCH.md), including
necessary pipeline/native implementation, qualification, pilot-selected training
budgets, matched controls, component trials and charts. It supersedes that plan's
pre-execution “no new allocation” status for this campaign. Preserve the original
vanilla allocation and references; do not shorten/relabel their results. Do not
start a new goal. Follow the existing stage tiers for any later feature additions.

## Reconciled starting evidence

Actual local inspection November 4 at 22:07 PST found the baseline supervisor
finished with exit 0. Its result records 14,231 updates, 1,800.021702 measured
update seconds, 213,465 supervised tokens and checkpoints at zero/300/900/1800.
Evidence: runs/dflash-modern-train-20261104-14/{state.json,artifacts/result.json}.
The prior baseline supervising agent is interrupted; no live DFlash trainer,
remote_job or llama-server was observed. Native evaluation is still outstanding.
RTX5080 showed 1,352 MiB used and 0% utilization; that alone is not a GPU lease.
The new sole campaign worker must freshly check pause/process/ownership state
before taking GPU ownership. Do not stop unrelated processes.

Current vanilla manifest/STATUS contain older launch-only fields. Reconcile those
with completed training and actual evaluation evidence during execution; do not
invent native results from training completion.

## Deliverable and execution order

1. Recover the retained prototype work, verify actual CLI/adaptation/export, and
   finish native admission/operations/build for the essential package and A8/A4/
   A2/A1 precision contracts. Reference-only export is not native completion.
2. Export/evaluate preserved vanilla checkpoints and the matched Q4 comparator.
3. Qualify/train the essential foundation, including midpoint/gain removal
   controls. Use a short pilot to fix matched exposure, acceptance envelopes and
   resource budgets before selecting precision candidates.
4. Execute four anchors, down-through-A1, bounded component reductions and
   interaction/recovery trials per PRECISION_SEARCH.md. Keep unsupported, failed,
   unadapted and trained outcomes visible. Prioritize broad actionable coverage.
5. Produce reproducible machine-readable results, full comparison tables,
   acceptance-versus-latency plots, throughput and size/memory plots, recovery
   curves, native correctness and matching evidence. Identify frontier and
   deployment incumbent separately. Render the chart in the parent conversation.
6. Only after the comparison is complete, if the user is still absent, select
   further work by evidence and record the rationale, new bounded budget and
   ancestry. Prefer resolving uncertain promising candidates before speculative
   additions; any features follow the existing tiers and matched controls.

Commit coherent source/report progress, keep large artifacts out of Git, preserve
recoverable checkpoints, clean owned disposable work and processes. No pushes.
Do not report completion from scaffolding, CPU tests, a detached job launch or
acceptance alone. Persist through routine failures. The parent remains available
for discussion; delegated completion wakes it for integration and chart display.

## Owner checkpoint — November 4, 22:24 Pacific

Sole implementation/training/native owner: prepared branch
`t3/precision-frontier-campaign`. All four prototype commits recovered without
duplicating equivalent earlier picks; retained baseline benchmark also recovered.
Fresh local pause control clear; RTX5080 idle before adopted lane. Vanilla
checkpoint evaluation is running, supervised in local tmux
`binary-eagle-runtime` / `precision-vanilla-eval-20261104-01`. Canonical run path
`runs/precision-vanilla-eval-20261104-01`; ownership.json contains boot/process
birth/source pins. Authenticate those before TERM of supervisor; allow120seconds
and verify terminal state/group absence. No other campaign GPU job active.

Completed isolated CPU/CUDA compilation of candidate runtime under
`runs/precision-native-20261104-01`; source/base objects remain untouched.
New loader/graph supports affine FFN gains and midpoints, per-projection
A1/A2/A4/A8 and fixed/learned clipping metadata. Still UNQUALIFIED pending
actual GPU operation and native model checks. A1 uses signed +/-absmax per token
for dynamic mode, threshold0 (nonnegative positive), no zero code; all-zero
rows scale0. Fixed/learned mode uses +/-bound, with zero input positive.
Dynamic input STE is identity; fixed/learned uses clipped input STE plus step
gradient. A2 retains its separate three-level {-scale,0,+scale} grid.

Nine inherited CPU tests pass after A1 contract addition. Runner CUDA support,
baseline-parent conversion, matched cursor continuation and signal checkpoints
implemented but full CLI/GPU verification remains next. Do not mistake this
checkpoint for campaign completion. Vanilla registry now reflects actual
14,231 updates / 1,800.021702 update seconds and pending native evaluation.

## Qualified pipeline and pilot — November 4, 22:33 Pacific

Real CUDA qualification completed:36 operation cases (A1/A2/A4/A8,
dynamic/fixed/learned, aligned edge and actual gate/up/down geometries),12
forward/gradient cases, full-model2→4 exact parameter AND optimizer resume,
explicit A8→A1 adaptation with inherited cursor/reset optimizer, native GGUF
round trips and native target-only output equality. Receipts under
`runs/precision-campaign-20261104-01/`; native qualification panel contains
target/Q4/vanilla-zero/essential4/A1-adapt2, all responses correct. Native
runtime now frozen at `runs/precision-native-20261104-03` for the campaign;
its CUDA-only affine operations reject CPU fallback. Isolated build receipts
pin sources/libraries; no submodule gitlink changed. CUDA supports dynamic,
fixed and learned exported bounds through actual INT8-code DP4A execution.
A1/A2 do not imply packed-activation speedups or smaller physical buffers.

Vanilla2warmup+5alternating repetitions completed,3 previously consumed DEV
prompts. All outputs correct, but native W1 variants remain much slower than
Q4. These are retained under the original runtime as a distinct series; final
controls will be remeasured under the common campaign runtime.128-step essential
pilot showed healthy updates, steady ~0.126s/update after first-use overhead,
but little native acceptance recovery on the narrow pilot cohort. No winner
is selected from these smokes.

Frozen pre-selection exposure policy: essential foundation2048 additional
updates from preserved vanilla1800 (pilot128 included by exact resume).
Matched vanilla/no-midpoint/no-gain controls each get2048 updates from the same
vanilla parent, same TRAIN order/cursor, Adam reset and constant rates. Standard
precision window1024 updates /15360 supervised positions; points0/256/1024.
This is approximately129s warm update compute on the pilot, plus separately
accounted setup/I/O/evaluation. All four anchors, including extreme A1/A2,
receive this window regardless of zero-shot damage. Per-job ceiling3600 update
seconds guards failures; it is not an allocation for preserved vanilla.

Initial quality screens are diagnostic: G8/G4 useful-progress envelope at least
0.10 accepted draft tokens/round and0.05 accepted/proposed; exploratory G2/G1
at least0.02 accepted/round or an observed positive recovery trend. Failing
these labels an arm below this short-run envelope, not permanently inferior.
No elimination before the qualified A1/A2 recovery window. Deployment needs
full matched repeated throughput; smokes cannot promote. Preserve near-ties.

Finite format queue: FFN W1 with G,D in{8,4,2,1}; protected-weight trials
limited to Q8_0/Q4_0 for fusion, attention Q/K, attention V/output, and a
drafter-private Q4_0 head. Target remains immutable. Embedding reduction is
blocked pending qualified private-copy support; no shared target modification.
Head starts at actual target-shared GGUF format (record inventory), so a private
head's bytes and lost sharing must be charged. Gate/up widths/gains stay shared.
No adaptive head/fusion/attention master weights are unlocked. Component training
uses actual block-quantized/dequantized frozen weights, not ignored flags.

Campaign resource reservation: before precision selection reserve four anchors,
up to two component mutations/family,2 global interactions and2 one-window
recovery extensions with matched controls. Reserve four locked finalist
confirmations from the common frozen foundation before exploring mutations.
Standard window1024; confirmation2048 with matched optimizer-reset history.
A bounded first-tranche ceiling60000 physical updates and20000 measured GPU
update seconds includes controls, pilot, confirmations and extensions; it is
a maximum, not an instruction to fill every combination. Storage/actual recovery
will choose the finite adapted queue after zero-shot cost/sensitivity probes.
No feature tiers2+ before the initial comparison is complete.

Active local supervised run: `precision-foundation-20261104-01`, socket
`binary-eagle-runtime`; canonical run state/ownership receipts under runs.
It trains foundation +3 removal controls serially and then a consumed natural12
DEV panel (4 code,4 reasoning,4 prose). This does not consume sealed evidence.
Foundation checkpoint freeze precedes anchor selection. Source changes remain
in the prepared checkout; no other worker or GPU job is authorized concurrently.

Storage: redundant qualification copies were removed only after verifying exact
parameters AND optimizer against retained full four-update recovery. Exports,
logs/configs/proof receipts retained. Actual free-byte accounting is recorded
in `qualification-storage.json`; concurrent pilot output affects observed delta.
Research/vanilla checkpoints untouched. New archive helper verifies SHA256
through lossless compression/decompression before removing an uncompressed
owned checkpoint; archived checkpoints restore through mmap-backed scratch.

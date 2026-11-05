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

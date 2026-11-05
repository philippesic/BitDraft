# DFlash execution campaign checkpoint

## Authority — November 5, 2026, approximately 11:00 Pacific

User: “great make your way through this. exhaust the list then compare latencies
of our candidate and q4km”. This authorizes implementation, builds, qualification
and GPU measurements through the inventory in
[EXECUTION_OPTIMIZATION.md](EXECUTION_OPTIMIZATION.md), followed by the complete
matched latency comparison. The earlier planning-only statements are superseded.
Continue the existing active goal; no new goal or training campaign.

Cover the list beyond its small initial experiment set. Each listed opportunity
needs an explicit disposition: implemented/tested and retained, tested and
rejected, or concretely blocked/inapplicable with evidence. A small first win or
one completed profile is not completion. Conditional hardware/algorithm reserves
need bounded feasibility checks; do not turn them into an unlimited unrelated
algorithm search. Report remaining limitations rather than claim impossible
universal exhaustion or promise a win against Q4.

## Fixed subject and invariants

Primary W1 subject is the exact compact G4D8-learned-small-2048 checkpoint/export
from the chart that prompted the investigation. Preserve its hash and original
runtime04 as controls. Fixed-bound1024 may be a separately labeled qualification
control, not a replacement primary candidate. The Q4_K_M comparator, target,
verifier, precision, features, greedy depth-two policy and data ancestry stay fixed.
No new QAT, precision reductions or capacity/architecture changes. Exact and
approximate execution changes require separate qualification and labels.

Use local RTX5080 only, one operational owner, supervised jobs, unique run paths,
pause checks and authenticated process identities. Parent inspection at launch:
5080 clear pause flag,1352MiB,0%utilization, no live project trainer/server/job.
The operational worker must freshly verify these before taking ownership.
Preserve existing paid recovery archives and raw native model dictionary bases;
WSL and Windows space are limited. Reuse qualified artifacts and isolated builds.
No pushes or unpublished submodule gitlinks. Parent remains available for discussion.

## Required deliverables

- Observational traces of original W1 and Q4, actual N/shapes/dispatch, copies,
  quantizer/dot/head/attention/launch/capture/sync costs, with uncertainty stated.
- Qualified model-specific fixed-work replay and changing-context comparisons.
- Separate copy-only, known-bound-only and combined tests; cross-position DP4A
  versus affine-aware signed INT8 MMA; fusion and unused-anchor-head opportunities;
  evidence-led checks of remaining low-bit/FP16-code/residency/scheduling reserves.
- One ledger covering EVERY inventory item, dependencies, implementation/evidence,
  correctness, isolated and complete latency, memory cost and retain/reject/block.
- A tested retained combination and matched original-W1 / optimized-W1 / Q4
  comparison. Also remeasure Q4 after optimizations that apply to both drafters;
  preserve target/verifier arithmetic and consistent-decode setting throughout.
- JSON/CSV, readable report, static plots and interactive comparison HTML. Distinguish
  GPU region/event timings from host spans, replay from live execution, and clean
  latencies from profiled ones. Include native correctness, proposal/cache checks,
  full request throughput context, memory and paired uncertainty.
- Commit coherent progress and update ownership/recovery/checkpoints here and in
  STATUS/goal. Release owned processes and safe scratch at handback; keep useful
  results, caches, immutable source snapshots and recovery evidence. Completion
  requires actual comparisons, not prepared source or detached job metadata.

## Starting evidence and links

Frozen precision campaign: runs/precision-campaign-20261104-01/.
Runtime: runs/precision-native-20261104-04/.
Primary model metadata: runs/precision-campaign-20261104-01/finalists/
G4D8-learned-small/model-final.json; archived model can be hydrated with the
existing scripts/dflash_baseline/native_archive.py helpers.
Baseline settings: experiments/w1a8-standing-20261102/baseline.json.
Source plan: docs/EXECUTION_OPTIMIZATION.md. Operational worker owns subsequent
live-state updates; this initial entry records authorization, not a GPU launch.

## Implementation owner — November 5, 11:10 Pacific

Sole worker is the current `t3/dflash-execution-optimization-campaign` worktree.
Fresh local verification found pause clear, RTX5080 1352MiB/0%, no competing
trainer/server/supervisor. Canonical artifacts: `runs/execution-campaign-20261105-01`.
`ownership.json` freezes primary export/archive identity, baseline and initial
WSL/Windows headroom; `live-ownership.json` records actual job boot/startticks,
source checkout, supervisor groups and stop procedure. No training or push.

Unchanged original W1/Q4 observational Nsight Systems traces completed with
actual server execution and target-identical outputs. W1 dot grid-y is3;
quantizer grid-x is3. Both profile reports are retained in `profile-panel`.
These are diagnostic traces, not clean latency results. Profiling wraps the
server AFTER benchmark environment filtering, so Nsight injection survives
without carrying algorithm overrides into the target runtime.

Current supervised local session `execution-tranche1-20261105-01`, socket
`binary-eagle-runtime`, serially builds copy-only, bound-only, combined,
warp-bound, cross-position DP4A and signed INT8 MMA overlays. Frozen source/base
objects remain untouched. Actual independent CUDA qualification is required
before native comparisons. Target/verifier and consistent decode32 remain fixed.
Next: finish qualification, model-specific cache-restoring native replay and
clean native ablations; then fusion/dead-output and reserve checks. The campaign
is active and the inventory/comparison are not complete.

### First native gate — November5 11:12 Pacific

Six exact alternatives pass native output/cache and common intermediate byte
checks on model-specific replay with three changing contexts; independent CUDA
checks cover every admitted width/mode and N1/3/8/9. Evidence:
`tranche1-correctness.json`, `runtime-*/ops-N*.jsonl`, `replay-tranche1/`.
Full cache is cleared and rebuilt from identical native captured DEV features
before every repetition; Q4 rebuilds its own cache. Original trajectories are
reproducible. Separate audit callbacks never determine clean timing.

Early event-region medians: original graph1.421ms, copy1.412ms, bound2561.391ms,
both2561.385ms. These sequential early screens do not establish confidence or a
final gain. Warp-bound32 is rejected at2.293ms; dot variants built on it are
qualified but need fair faster-prep reruns. Exact changes are not promoted from
microbench alone. Native actual-request comparison remains outstanding.

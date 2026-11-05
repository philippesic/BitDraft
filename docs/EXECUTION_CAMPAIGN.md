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

### Broader execution gate — November5 11:28 Pacific

Actual N3 projection screen and SASS are retained in `micro/`, `mechanisms/`.
Signed INT8 MMA executes IMMA.16832.S8.S8. The tested INT4 PTX compiles on
SM120a but lowers to two signed INT8 IMMA sites; no native INT4 mechanism claim.
FP16-code and resident-expanded INT8 alternatives were timed with activation
conversion charged; down projection loses, resident gate/up gains modestly with
its explicit memory tradeoff. No historical N7 ratio is reused as modern evidence.

Tiled fixed-bound quantization executes its completion reduction/workspace and
passes native logits/cache checks. Early graph-event median1.264ms versus
original1.427ms; joint gate/up1.302ms. These are screens, not final paired effects.
The twelve candidate CONT nodes do NOT appear as copy kernels in the live draft
CUDA graph: backend allocation aliases them. Six named copies are visible only
in callback audit graphs. Copy-removal remains a small graph simplification,
not a claimed12-copy GPU saving.

Head-output pruning honors per-token requested-logit flags and standard output
index mapping; ordinary DFlash sampler never requests anchor logits. DSpark,
selector/LiliCorr and all-output diagnostics retain their requested outputs.
Final-layer crop occurs only after final attention/cache writes, with embedding
requests excluded. Both variants preserve requested mask logits and cache bytes.
Drafter-only generic fusion changes activation pack bytes in some contexts and
is an approximate rejected/deferred lane unless separately qualified. Target/
verifier fusion restriction remains32 throughout. Narrow fusion implementation
needed one failed admission-header build-order repair; failure logs preserved.
Next: combine exact wins, complete bounded reserves/edge checks, and perform
alternating matched live-request confirmation and charts. Campaign still active.

### Inventory covered; confirmation preparing — November5 11:40 Pacific

Thirty-eight explicit inventory entries are recorded in `ledger-start.json`.
Modern N1/N3 bitplane/LUT/FP16-code/resident INT8 screens completed; selective
LUT4 was also built and measured in native full forward, losing to the retained
joint/tiled combination. SiLU/down-pack fusion now actually dispatches;12 CUDA
fixtures show changed packed streams (including A1/subnormal cases), so it is a
separate approximate rejected lane despite matching the bounded primary native
trajectory. Prenorm fusion actually executes; no approximate path is in the
retained combination. Generic drafter fusion is likewise rejected.

Current retained exact candidate: `runtime-combined-last02` (tiled known-bound
pack, graph copy simplification, joint gate/up, requested head rows, final-layer
postattention crop). Early N3 graph-event screen1.174ms vs original1.428ms;
fair cropped Q4~1.194ms. These are sequential screens and do not establish a
win/tie against Q4. Requested W1/Q4 mask logits and model-specific cache bytes
remain identical. Target/verifier code paths/precision/settings are unchanged.

Additional edge receipts `edge-checks03/result.json` cover all4bits/3modes,
strided fallback, signed-zero/subnormal streams, explicit ties, tiny/high finite
bounds, and known-bound nonfinite/overflow rejection. Original max-float-bound
CPU NaN-payload comparison needs a native-original cross-check; ordinary extreme
finite1e30 bound passes. Failure artifacts remain intact.

Next: final independent joint/overflow checks, alternating cache-restoring replay
across3consumed captureDEV domains, then5alternating actual-request repetitions
on consumed natural12 with paired plain/host-trace panels. Final artifacts and
release are still outstanding; do not infer completion from the prepared runtime.
Current task storage~2.8GiB, WSL~6.7GB/Windows~11.5GB free. Bounded budget revised
to3.5GiB including final evidence; no expanded full models on disk. Paid models,
archives and shared dictionary basis remain protected.

### Final confirmation live — November5 approximately noon Pacific

Independent retained joint kernel passes every bit/mode at N1/3/8/9 with actual
joint dispatch. Maximum finite clipping-bound outputs also match original GPU
bytes, including overflow NaN payloads. The CPU-only max-bound payload mismatch
is resolved by that direct native comparison; no arithmetic change was needed.

All60final fixed-work replay cells finished:540forwards including warmups,
180measured graph regions,30paired per-domain/repetition original/optimized
comparisons, all required logits and model-specific cache bytes exact. Replay
assembly initially failed only because the pin helper receivedstr rather than
Path; complete preserved progress was independently re-audited and assembled,
without replacing or rerunning GPU observations. `replay-final/result.json`
records that recovery. One28.61ms host outlier remains in the data (its graph
event1.171ms); no outlier trimming.

Live supervised session `execution-native-final-20261105-01`, socket
`binary-eagle-runtime`, is in its last actual-request repetition. Source and
current authenticated identities are in `live-ownership.json`; raw results in
`native-final/`. Candidate/source/baseline pins remain fixed. Proposal-token
tracing is confined to diagnostic companions; all throughput uses plain requests.
After this panel: short final Nsight traces, standalone patch reconstruction,
all-output/embedding fallback checks, artifact publication and owned release.
No completion claim or deployment promotion yet.

## Completed — November5, approximately12:20 Pacific

The38-item execution inventory is dispositioned in
[ledger](../experiments/execution-campaign-20261105/ledger.md) and
[report](../experiments/execution-campaign-20261105/report.md). The retained exact
runtime is `runs/execution-campaign-20261105-01/runtime-combined-last02`.
The primary learned-small2048 model/export remains byte-pinned and unchanged.

Final paired results: GPU forward1.428→1.183ms, reduction17.2% [95%15.5–18.8%].
Actual full-request throughput171.90→176.21tok/s, +2.51% [1.37–3.50%].
Q4 GPU forward1.178ms: candidate/Q41.004× [0.989–1.022], unresolved difference.
Q4 request244.39tok/s remains incumbent; W1176.21is0.721× [0.701–0.738].
Fair eligible-change Q4 was separately measured, with no resolved improvement.
Acceptance/counters remain unchanged between each original/optimized model.

All60final replay cells completed:540forwards,180measured warm graph regions,
required mask logits/cache bytes exact across3changing contexts/3captureDEV
sources/5alternating reps. Main request panel has700responses (600measured
plain+diagnostic), all target-identical. Corrected actual selected-proposal
tracing uses the token passed to common_sampler_accept, as consumed by DFlash;
its deterministic natural12 panel matches each original/optimized stream and
all five timing-repetition counters. Earlier ignored sampler-return traces are
preserved as superseded diagnostics, not used as proposal proof. No target/
verifier change or unbounded HF/native parity investigation.

The last A4 four-plane reserve was actually implemented and independently
qualified. Five alternating paired replay repetitions give planes/joint0.9938×
[0.9772–1.0116], leaving its small possible extra gain unresolved. The simpler
qualified joint DP4A runtime remains retained. Other reserve, fusion, storage,
allocator/graph/cache/attention and scope dispositions are in the ledger.

Standalone patch reconstruction actually passes all-output, anchor, embedding
fallback and cache checks. Rebuild using `scripts/execution_campaign/reproduce.py
NEW_UNIQUE_RUN_DIR`; the patch and exact base/source pins are tracked under
`experiments/execution-campaign-20261105/runtime/`. No submodule gitlink or push.
Final JSON/CSV, five PNG/SVG pairs, offline interactive HTML and theme-compatible
inline HTML are in `runs/execution-campaign-20261105-01/charts-final/`.
The inline chart previews without console errors. Earlier chart series remain
preserved, including the original precision/held-out campaign.

All27owned supervisor groups are absent; all server processes reaped. Fresh
RTX5080 observation:1352MiB desktop baseline/0% utilization, no GPU processes.
Pause flags unchanged. `release.json` records terminal identities/statuses.
Owned failed/superseded binary caches and duplicate scratch were safely removed;
all sources, raw numerical/native evidence, paid checkpoints/exports/archives,
shared dictionary basis and useful qualified runtimes remain. WSL free measured
6,150,549,504→7,312,474,112bytes (+1,161,924,608); native Windows C free
11,549,904,896→11,549,904,896 (unchanged). No VHDX compaction/reclaim claim.
`retained-storage.json` records exact paths/retention and before/after accounting.

Current next: parent review and integration of this committed isolated branch.
No live operational GPU job or automatic follow-up remains. Keep the unmerged
worktree/branch until parent integration; the broader nine-model goal stays open.


## Parent review and integration — November 5, 2026

Integrated the committed execution branch after code/report/ledger review.
Independent checks: all34publication sizes/hashes match;700responses all correct;
600formal rows reproduce reported rates/counts; original/optimized native pairs
retain token counts and accepted/proposed/round/prefix counters;180measured GPU
regions each have one warm graph;30required-logit/cache comparisons pass.
Proposal-selection and all-output/embedding fallback audits pass. Python scripts
parse, and applying the standalone patch to authenticated frozen base files
reproduces every optimized source hash. No repeat GPU test was needed.

The inline chart previews at728px with no console errors. Parent will display it
in the conversation. The final readout must distinguish17.2% lower GPU forward
cost and11.0% lower full-draft host cost from a win over Q4: optimized W1/Q4 GPU
ratio1.004 has95% interval0.989–1.022, so they are unresolved. No full-request
throughput promotion. Exact model/config, no QAT and all earlier frozen artifacts
remain unchanged.

Fresh parent observation confirms5080 at1352MiB/0% and no project operational
server/trainer/supervisor. Retained inventory/release receipts account for worker
cleanup; paid archives/raw basis/checkpoints and qualified runtimes are protected.
Safe merged-worktree cleanup and canonical integrated evidence mapping are
recorded separately in runs/execution-campaign-20261105-01/parent-review.json.
Nothing pushed. The bounded execution request is complete; broader research
continues only within the current goal and subsequent user direction.

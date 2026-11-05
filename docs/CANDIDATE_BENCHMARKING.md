# DFlash candidate training and benchmarking

This is the entry point for candidate comparisons. The user clarified on
November 4 that the sequence is **essential-feature foundation first, precision
selection second, experimental feature tiers afterward**. Exact scope is in
[CANDIDATE_STAGES.md](CANDIDATE_STAGES.md): the proposed common foundation includes
signs, row scales, row midpoints, input-channel gains and existing norm gains.
The agreed [adaptive precision search](PRECISION_SEARCH.md) defines four activation
families, bounded component trials, matched adaptation and frontier reporting.

The [remaining-component recovery plan](REMAINING_PRECISION.md) specifies the
head/embedding/KV/scalar follow-on, private-copy controls, recovery checkpoints,
training eligibility and slow-starter retention. Its execution budget remains
pilot-defined; the historical vanilla allocation is not reused for new arms.

The already-authorized vanilla W1A8 run below remains a minimal control. Its
midpoint/gain exclusions apply to THAT frozen reference, not to the separate
essential-feature candidate. Preserve the live delegated run and its budget;
do not retrofit it or relabel it as the expanded foundation. Future training
allocations remain separate. This protocol supplements [EVALUATION.md](EVALUATION.md).

## Reference registry

The initial reference is registered in
[baseline manifest](../experiments/dflash-w1a8-baseline-20261104/manifest.json).
Read its status before attempting a benchmark. Null checkpoint or evaluation
fields mean not yet produced, never zero performance. The preparation worker
owns the experiment report and manifest; STATUS links to the current evidence.

| Reference | Role | Replacement policy |
| --- | --- | --- |
| Vanilla W1A8 zero-update export | Measures conversion damage and starting execution cost | Freeze exact model, initialization and export bytes |
| Vanilla W1A8 initial trained checkpoint | Frozen minimal-control parent; not the essential precision foundation | Freeze parameters, optimizer, RNG, schedule, cursor and exposure |
| No-feature continuation from that parent | Controls for the benefit of extra training | Produce a matched continuation for each comparison budget |
| Essential-feature foundation checkpoint | Frozen parent for precision anchors and final attribution comparisons | Freeze separately after qualification; do not relabel vanilla |
| Unchanged essential/family continuation | Controls extra training for each precise parent and exposure | Reuse only with matching ancestry, data/order and training policy |
| Candidate tradeoff frontier | Useful size, memory, latency and acceptance alternatives | Retain evidence and uncertain near-ties; no external-throughput win required |
| Deployment incumbent | Best sufficiently supported deployment candidate so far | Promote with matched native correctness/throughput evidence; keep old references |
| DFlash Q4_K_M | External primary throughput baseline | Never replace with the incumbent |
| Target-only | Correctness reference | Same target/runtime as each comparison |

Frozen external targets/settings are in
[baseline.json](../experiments/w1a8-standing-20261102/baseline.json).
Do not alter that file to install the new W1A8 reference. Historical DSpark/Qwen3
F16 results remain secondary evidence on their original track, not modern
Qwen3.5 baseline numbers or a substitute DFlash training checkpoint.

## Initial checkpoint allocation

The coordinator-selected initial allocation is **1,800 seconds (30 minutes) of
actual training-update wall time**, with zero-update and approximately
300/900/1800-second checkpoints. Save at complete update boundaries; report
actual times and any one-update overshoot. This allocation belongs to the vanilla
run only; the precision-search plan grants no additional allocation. This is a practical first screening budget, not a
measured optimal horizon or a convergence claim. The user requested
preparation and an initial run, not a new long-QAT campaign.

Measure forward/backward/optimizer update time consistently, synchronizing at
accounting boundaries so asynchronous GPU work is not omitted. Separately record
setup, captures, compilation, export, checkpoint I/O and evaluation wall time.
Preserve optimizer/RNG/data/schedule state at each checkpoint. Also record updates,
training tokens/windows, unique data coverage and effective epochs; 30 minutes
alone does not establish useful exposure. Fix a continuation-compatible learning
rate policy before the run; do not silently reset the schedule when branching.

Before the full initial allocation: qualify a real optimizer update, meaningful
sign/scale changes, resume, export and bounded native correctness. Calibrate step
cost and actual memory. Do not substitute scale-only EAGLE or historical DSpark
for vanilla DFlash because their adapters are available. If the model, data or
native contract is blocked, record exact evidence and continue useful preparation;
do not claim an initial checkpoint exists. At 30 minutes, evaluate and stop this
bounded training run. Additional allocations remain a separate decision.

## Initial engineering scope

Start with the modern Qwen3.5 4B Q4_K_M target on RTX5080, using its frozen DFlash
Q4_K_M comparator and correctness runtime. This lane is a coordinator-selected
first qualification target; the broader 4B/9B × Q4_K_M/NVFP4 goal remains intact.

Use the compatible released DFlash checkpoint, not a DSpark checkpoint relabeled
as DFlash. Inspect its actual layer count, target taps, mask layout and shared
embedding/head. Initial vanilla coverage recommendation: binarize its FFN
(gate/up/down) projections, use row scales and A8 matrix inputs; preserve fusion,
attention and head/embedding at declared known-working precision. Train selected
binary signs and row scales plus existing drafter normalization gains. Record the
exact coverage, sign initializer/update rule and activation arithmetic in the
manifest before training. A legacy five-layer/15-FFN contract does not qualify a
six-layer/18-FFN released DFlash model. Necessary adapters are authorized within
baseline preparation; no silent architecture/coverage reduction to fit old code.

Keep midpoint, input-gain, group-scale, clipping-learning, rotation, correction,
confidence and alternate-objective features off for this reference. Use the
original compatible DFlash objective, including its existing early-position
weighting, and matching target-native features/labels. Distinguish QAT arithmetic
simulation from the actual packed native inference export. No proprietary or
unverified recipe is inferred. Direct sign-flip optimizers and midpoint variants
remain proposed comparisons, not selected additions to vanilla.

## Branch comparisons

1. Branch the no-feature continuation and single-feature candidates from the same
   parent. Match training examples/order and exposure where meaningful; also
   compare measured compute because equal updates need not have equal cost.
2. Initialize additions with equivalent behavior when possible: midpoint zero,
   input gain one, residual correction zero. Verify actual numerical behavior;
   topology/precision changes may not have an exact no-op initialization.
3. Compare a feature with the no-feature continuation at the same additional
   budget, not only with the starting parent. Save intermediate learning curves.
4. Compare sign optimizers and initialization choices from zero updates as well;
   a trained parent can hide initial-conversion behavior and optimizer inertia.
5. Promote a repeatably useful candidate to the working incumbent, retaining its
   fixed-baseline and external-Q4 results. For the next addition, again run an
   unchanged-incumbent continuation. Test combinations against their components.

Deployment promotion requires matched native output correctness and useful
complete-request throughput, not just lower training loss, reconstruction error or higher acceptance
percentage. Short-run promotion is provisional. Extend promising candidates and
selected slow starters when authorized; no universal early-to-long predictor is
established and a short loss does not permanently invalidate a feature.

For precision searches, follow PRECISION_SEARCH.md: exploratory mutations branch
from matched immediate parents; final precision-attribution arms use the frozen
essential foundation, identical exposure/data order and training policy. Different
curricula are different recipes even with equal total tokens. Initially train only
foundation groups and qualified quantizers; other master weights remain frozen,
with their selected quantized arithmetic present in the training forward pass.
Frontier retention is not deployment promotion and may preserve a smaller, slower
candidate. Freeze finalists/recipes on DEV before opening held-out results.

## Measurement contract

- Reuse explicitly consumed development cohorts for screening; preserve sealed
  final inputs and apply all TRAIN exposure/alias exclusions. New DFlash taps or
  histories require compatible recapture, not mislabeled existing EAGLE caches.
- Freeze target, verifier, hardware, prompt suite, sampling, context, KV precision,
  output limit, proposal policy and runtime within a comparison. Label any shorter
  DEV workload separately from the historical 256-token full matrix.
- At zero/5/15/30-minute points, retain validation loss and token agreement where
  available, sign flips and persistent changes, scale/norm changes, gradient
  health, data exposure and resources. Native evaluation after serialized training
  stages must not compete with the training process for GPU resources.
- Measure fixed-workload draft latency including packing, head and required
  catch-up; separately measure verification and full request throughput.
- Report accepted/proposed counts, first-rejection/prefix survival, emitted tokens
  per round, total verification rounds, decode speed and full-request speed.
- Use two warmups and at least five alternating measured repetitions for the
  benchmark comparison, following EVALUATION.md. A smaller diagnostic is labeled
  smoke only. Timing repetitions are not independent prompts or training seeds.
- Keep model/source/data hashes, commands, runtime settings, raw result paths and
  checkpoint ancestry in the experiment manifest/report. Report an unresolved
  difference as unresolved rather than declaring a winner from noise.

## Execution and ownership

Follow [AGENT_OPERATIONS.md](AGENT_OPERATIONS.md). One worker owns the local5080
lane, checks the pause flag/processes before experiments, and launches supervised
jobs through scripts/remote_job.py with unique run directories. Preserve unrelated
work and healthy jobs. No2080Ti operation, no SSH and no pushes are authorized by
this task. A user pause overrides this allocation.

The user explicitly authorized a subagent for initial baseline preparation and
training. It may implement necessary adapters, prepare compatible TRAIN captures,
qualify the native path and run the bounded allocation. It must not launch feature
sweeps or a major long-QAT allocation. Update the baseline manifest/report and
current durable state with checkpoint locations, source pins, birth identities,
stop procedures and actual status; prepared metadata is not a completed run.

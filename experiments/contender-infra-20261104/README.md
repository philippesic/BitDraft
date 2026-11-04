# Contender workflow qualification

**The shared workflow is ready for candidate comparisons on the qualified current
reference.** Training, cooperative interruption/resume, export, staged execution,
matched native benchmarking, comparison auditing and local deployment packaging
were exercised. New architectures still require their own model/state/export/
kernel adapters and resource admission; this report does not claim an unknown
design is already long-QAT ready.

Use the [workflow guide](../../scripts/contender_pipeline/README.md) and
[machine-readable qualification](qualification.json). Raw evidence and immutable
source snapshots are under `runs/contender-infra-20261104-01/`; checkpoints remain
under `/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01/`.
No architecture, precision, feature recipe or major long-QAT allocation changed.

## What works

| Need | Evidence obtained |
| --- | --- |
| Common execution and budgets | Immutable argv plans, ordered dependency/stage resumes, cumulative per-candidate/shared costs, reserved deadlines/grace, carried prior costs, output checks and bounded logs. Actual supervised GPU stages and final packaging run through it. |
| Training and recovery | Real SIGTERM after update5762; resume to5808 matches continuous5760→5808 exactly in parameters, Adam, CPU/CUDA/Python RNG, cursor, exposure and contract. Meaningful scale changes occurred. |
| Changed workflow fidelity | Bounded audit snapshots + packed F32 storage produce the same parameters, Adam, RNG, cursor/exposure and exported GGUF bytes. Scientific contract matches; only the explicit helper/data freeze differs. |
| Data arrays | All3807 source arrays authenticated;600,442,880bytes in10bounded shards. All3603current windows and2135required embedding rows match the admitted resident baseline exactly. |
| Larger metadata mechanism | An LRU of admitted member manifests, stable sample IDs, FIT-only window admission and sampler resume across epochs. Actual current2879FIT windows indexed;24sampled windows materialized across12manifest loads with one active manifest and a shared bounded row reader. |
| Smoke and repeated comparison | Original target-only, frozen DFlash, initializer, paid5760 and continuous/resumed exports;12consumed DEV cases,2warmups and5alternating repetitions. Independent raw-request/settings/model/library/metrics/release audit. |
| Measurement repeatability | Identical exports have identical accepted/proposed tokens and rounds for all70paired measured/warmup requests. Speed ratio0.99970,95% interval0.98560–1.01311: no spurious distinction established. |
| Final confirmation protections |73related source/data/recovery/confirmation tests plus new synthetic reservation and winner-bundle tests. Historical alias/exclusion pipeline remains authoritative; no dedicated untouched input payload opened. |
| Local product artifact | Drafter copied into a bundle with settings, explicit target/runtime dependencies and evidence; copied model runs natively and passes14exact-output requests. Bundle relocation/corruption and false-winner rejection tested. |

**119 CPU test methods pass:**35new workflow tests,73related existing tests and
11current-reference checks. They cover stale/missing/wrong outputs, changed
inputs, finite/count checks, corruption, timeouts, signals, separately-sessioned
child cleanup, hard-crash recovery, PID reuse, exclusive locks, log I/O errors,
log size bounds, cumulative budgets and protected confirmation consumption.

There are **476exact target-matching native responses**, including360measured
requests. All34native process groups released. Continuous, resumed and optimized
exports are byte-identical. These are infrastructure/repeatability controls, not
three independent research candidates or a new feature/generalization result.

## Setup performance

In one bounded CPU comparison, after shared prior reads and with the optimized
path run first, the snapshot auditor took39.97s and the original took103.77s.
Both rebuilt exactly the same union manifest and accounted for the same
728,842,240row bytes,995,468,999hashed bytes and21,320,631metadata bytes.
The optimization retains row geometry, hashes, finite checks and source ancestry;
it removes repeated filesystem access using bounded immutable copies.
[Raw measurement](audit-cost.json). This is setup validation, not GPU/decode speed,
and one measurement does not establish a universal speed ratio.

The current48-update optimized training/export rehearsal took131.02s inside its
trainer, with3.12GB peak allocated CUDA memory. Setup still has a real cost;
experiments should amortize it with useful training chunks. Array and metadata
caches are bounded; their limits do not bound the OS file cache, arbitrary caller
references or all Python RSS. The shared-manifest probe peaked at about659MiB RSS.

## Deliberate failures and corrections

The first benchmark adapter omitted the required initializer control. Existing
admission rejected it before inference. The adapter now carries all four reference
controls and the complete repeated panel passed. That failed attempt remains in
cost accounting and raw evidence; its output was not promoted.

An initial exhaustive CPU comparison used repeated original filesystem readers
and was superseded with the already-admitted immutable resident baseline plus
bounded thread count. All-window comparison then completed. The final metadata
stream shares its immutable row reader across manifest eviction to avoid repeated
array-index construction/authentication. Earlier diagnostic evidence is retained.

Hard-crash/log-failure/corruption tests were synthetic; the real GPU interruption
was a cooperative complete-update stop. Arbitrary device failure mid-optimizer
step is not treated as a valid checkpoint. Use a known completed checkpoint and
an explicit recovery plan.

## Costs and release

The task executed144physical updates covering the same48-update reference
trajectory; it introduced no research treatment. Update wall totals105.05s.
Cumulative supervised command wall is1113.09s (**18.55minutes**), including setup,
training, export, native comparisons, packaging and the rejected first benchmark
attempt. Update wall is contained in that total. Overlapping CPU audits/tests
outside supervision are separate; this is not a claim of18.55minutes pure GPU
utilization or total project elapsed time.

All supervised jobs and owned native groups are stopped, with no GPU compute
process left. Useful checkpoints, raw evidence, source snapshots and reusable
shards are preserved. Identical immutable exports/bundle copies share hardlinks
where possible, with every path and byte preserved; measured disk free-space
receipts are in qualification.json. No push occurred.

## What remains tied to our design decision

- The reference GPU path is4B Q4_K_M, fixed-sign row-W1A8, depth2, mode32, greedy,
  single-request execution. All four selected target/DFlash artifact sets rehash;
  this task did not run new architecture/precision comparisons on the other three.
- Grouped scales, full-sign state/export, different heads, new kernels or proposal
  graphs require their chosen adapters and the same admission/recovery tests.
- Larger TRAIN source selection/capture, complete alias exclusion, appropriate
  diversity and the chosen long-run horizon remain explicit recipe work. The
  streaming mechanisms are tested; an arbitrary large corpus is not prequalified.
- The bounded rehearsal trainer retains its7680cap and original19200schedule
  semantics. Choose and qualify the actual long-run adapter/allocation explicitly.
- Different context/concurrency, stochastic verification or deployment hardware
  require their own scope-specific runtime checks. Current smoke/contender tests
  do not establish those product claims.
- No early training metric is validated as a universal predictor. The workflow
  preserves uncertainty, selective continuation and independent confirmation;
  it does not permanently reject backlog ideas from short losses.

These are candidate/workload decisions and implementation boundaries. The common
execution, accounting, data access, measurement, recovery and packaging path now
has runnable tools and direct evidence instead of per-experiment orchestration.

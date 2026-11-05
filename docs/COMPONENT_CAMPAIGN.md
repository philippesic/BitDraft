# Remaining precision component campaign

## Execution authority — November 5, 2026

User: “ok go do it”, following the incremental head/embedding/KV/scalar plan and
an estimate of9–10minutes of update time per approximately4,000-update arm.
This authorizes implementation, qualification, pilot, matched recovery training,
component screens, combinations and measured comparisons in
[REMAINING_PRECISION.md](REMAINING_PRECISION.md). Earlier planning-only statements
are superseded. The estimate excludes setup/export/evaluation and assumes the
existing trainable groups; additional head/embedding-master QAT needs a pilot.
Continue the existing research goal. No pushes or target/verifier changes.

## Subject, sequence and completion

Freeze B0, the learned-small2048 compact G4/D8 model with the retained optimized
execution runtime. Start with shared Q6_K, exact private Q6_K and private Q4_K
head arms. Continue private/tied embeddings, drafter K/V ladders and scalar/norm
storage, then useful combinations and a small confirmed frontier.

The first primary recovery arms get the planned cumulative0/256/1024/4096
readouts; do not silently treat1,024updates as completion of the4,096-update
comparison. Calibrate resource cost and total budget before launching the broad
queue. Give viable damaged quantizations matched recovery with an unchanged
continuation, retaining at least two slow-starter opportunities. Additional
format ladders and targeted-QAT rescue use the finite evidence-led plan, not a
Cartesian product. Distinguish frozen-head upstream compensation from actual
head-master QAT. Missing trainable/native support is blocked, not failed quality.

Completion requires real native exports/admission/evaluation, matched recovery
curves and latency/memory/throughput charts covering the four component groups,
plus combined candidates or concrete evidence why an arm cannot be implemented.
A prepared adapter, CLI, passing CPU test or healthy detached launch is progress,
not completion. Do not reject an arm merely from initial acceptance damage.

## Operational boundary and launch observations

One delegated worker owns GPU work; parent remains available for discussion.
Parent at13:07Pacific observed5080 idle1352MiB/0%, pauseclear, no project trainer/
server/supervisor. Fresh owner checks still required. Long jobs use remote_job.py
and unique local tmux/run directories with process birth/boot/source pins and
stop/recovery instructions. No SSH or2080Ti. User pause overrides experiments.

Resource observations: WSL free7,968,272,384bytes; Windows C free11,541,807,104bytes;
WSL totalRAM approximately20GB, available approximately18GB. Native/CPU state and
private vocabulary matrices must be budgeted explicitly. Preserve all earlier
paid checkpoints/archives/raw model dictionary basis and immutable source data.
Before broad training, choose a safe bounded NEW-campaign checkpoint retention
and compression policy, keeping active state recoverable, frontier states resumable,
and useful native exports/curves/source/data ancestry. Do not promise that a
large master-head+optimizer fits or discard valuable states to force the sweep.
If space truly blocks an arm, record the exact prerequisite and continue useful
independent arms. No disk expansion or personal configuration edits implied.

## Reusable evidence

- docs/REMAINING_PRECISION.md: exact ladders, controls, recovery and tying rules.
- experiments/execution-campaign-20261105/{report.md,runtime/optimized.patch}:
  retained exact DP4A/gate-up/tiled-quantizer/output-pruning implementation.
- runs/execution-campaign-20261105-01/runtime-combined-last02 and
  runtime-reproduction01: actual qualified optimized runtime binaries.
- runs/precision-campaign-20261104-01/finalists/G4D8-learned-small/model-final.json:
  frozen candidate and archived native export/checkpoint ancestry.
- scripts/dflash_baseline: current training, native archive and export components.
- experiments/w1a8-standing-20261102/baseline.json: frozen external Q4 comparator.

The operational worker owns subsequent checkpoints here, manifest/report and
STATUS/active-goal updates. Keep large outputs/raw logs out of Git, commit coherent
source/evidence progress, clean safe owned scratch/processes at handback and
preserve healthy active training if the user asks a question during work.

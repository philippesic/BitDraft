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

## Implementation owner checkpoint — November5,13:20Pacific

Implementation runs in the supplied T3 worktree, without delegates or pushes.
Fresh RTX5080 check: idle1352MiB/0%, no project process; pause clear.
Canonical artifacts: `/home/philip/binary-eagle-decoding/runs/component-campaign-20261105-01`.
Full B0 archive authenticated against the inference sidecar: exact learned-small2048
parameters/Adam/RNG/order,5,100,543,059raw bytes and4,207,188,154archive bytes.
All prior artifacts remain protected.

The new model-only runtime overlay reuses the immutable optimized CUDA library,
adds private/tied embedding admission and F16 metadata conversion to F32 graph
operands. Q4_K uses the pinned GGML native quantizer from targetQ6_K, notQ4_0.
Preparation completed; native admission and full-model arithmetic remain to test.
Draft-cache source audit finds implicit Hadamard rotations:K128,V64, followed by
32-value blocks, native scale-half rounding and read dequantization. Simulation
implements these semantics, pending native fixture/history qualification.

Live pilot: local tmux socket `binary-eagle-runtime`, session
`component-pilot-20261105-01`; supervisor record and stdout under the CURRENT
WORKTREE `runs/component-pilot-20261105-01/`, not the canonical artifacts directory.
Exact supervisor/group birth identities are in `state.json` and arm ownership.
Stop the authenticated supervisor withSIGTERM, allow90seconds for recovery save,
then verify its group and GPU release; never stop unrelated processes.
Pilot is32updates, NOT the primary4096allocation; full checkpoint and exact
resume/native gates precede the broad queue.

Resource policy: recover B0 into owned RAM-backed scratch, stream NEW full
checkpoints without a raw disk duplicate, keep bounded rolling NEW recovery
states and selected frontier resumability, archive native exports against shared
component bases. Windows campaign checkpoint destination is
`/mnt/c/binary-eagle-component-20261105-01`. Report real device/tensor/cache bytes
separately from archival compression. Training exposure repeats the24original
TRAIN sequences unless compatible broader captures are explicitly admitted;
overlapping windows never count as fresh labels. Budget remains pilot-defined.

## Native qualification and resource pilot —13:34Pacific

Full pilot:32actual updates in4.882seconds, median0.143224seconds/update;
4096updates project to586.65seconds of update time, excluding setup/export/DEV/
checkpoint I/O. Dense consumed-DEV CE6.89435→6.62556; no acceptance claim from CE.
Actual full-head training fits the5080; exact allocator peak is in
`arms/H0-pilot/result.json`. Full F32 parameter/Adam/CUDA-RNG recovery and identical
native re-export passed. CUDA cache Q8_0/Q4_0/Q5_0 fixtures reproduce native
write/read rounding exactly on2432values each and have finite STE gradients;
K128/V64 Hadamard involution and scalar-gradient checks pass.

Native short64-token captureDEV smoke completed for H0/H6/H4/E6/E4/HE4, all six
K/V pairs, four projection-scalar groups, norms and combined scalars. All final
target responses are correct. H0/H6/E6 acceptance counts agree55/263. Q4_K
private head55/263, E458/257, corrected tied HE457/259. These are unwarmed
smokes, not formal latency/quality conclusions. Real cache logs confirm requested
formats, K128/V64 rotations and buffer allocations. Tied allocation initially
followed the CPU input path; corrected runtime03 stores one output-placedCUDA
Q4_K tensor and aliases the embedding to it. Runtime01's unsupported bool-template
link and runtime02's slow tied placement remain superseded diagnostics; qualified
old runtime sources/binaries are unchanged.

NEW native states now use authenticated tensor-slice archives, referencing exact
frozen tensor bytes and storing only new bytes. Full reconstructed file hashes
are verified before retiring raw files. File compression is not model memory.
CPU archive conversion was initially concurrent with a native reader and hit a
file-lifecycle race; final native panel completed correctly, conversion was
rerun serially and passed. All future archive conversions/native consumers serialize.
The storage report is `consolidation.json`; no pre-campaign artifact was touched.

Windows checkpoint conversion reached its512MiB free-space floor and preserved
the original full state. The incomplete NEW partial was authenticated as disposable
and removed, restoring measured Windows free space to6,299,725,824bytes.
A bounded streamed byte-shuffled retry now precedes the H0/H4 recovery stage.
Live supervisor: current worktree `runs/component-head-recovery-20261105-01`,
local tmux socket `binary-eagle-runtime`, matching session. Birth/group identity
is in its state.json. Stop authenticated supervisor SIGTERM with90s grace,
then verify group/GPU release. The stage includes full4096updates,256/1024
readouts, recoverable rolling checkpoints and actual native readouts, not only
the pilot. The budget is `budget.json`: maximum114688provisional updates
(including controls/reserves/combinations/confirmation), approximately4.56hours
pure update time at pilot speed; actual cost is tracked separately.

Remaining: full head recovery, finite Q3 probe/conditional bridge/R2, embeddings,
KV long-context/history qualification and recovery, scalar recovery, interactions,
locked frontier confirmations and charts. Entire component campaign remains OPEN.

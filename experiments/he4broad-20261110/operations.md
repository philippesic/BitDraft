# HE4broad fresh training on the November 9–10 captures

State: **RUNNING; eight-hour cumulative update-time continuation**. Philip authorized this run November 10 in
thread `6a9bee6a-3f4d-4d2f-87c5-73a5e841c6c6`. The prerequisite campaign is complete and its RTX 5080 release is verified. Stage 6 release was verified against its pushed release proof and the completion message.

## Current eight-hour update-time continuation

Actual launch: local tmux socket `binary-eagle-runtime`, session
`he4fresh-update8h-20261110-01`; supervisor
`runs/he4fresh-update8h-supervisor-20261110-01/state.json`. Active source is
`runs/he4broad-20261110-01/source-update8h-20261110-01/`, selected by
`active-source.json`. Full parameters/Adam/RNG restore passed from the 20,671
update parent; the cursor continued from 20,671 and completed new updates.
All three locks belong to the authenticated workflow; storage is local.
The existing health event is enabled at 30-minute intervals; its first resumed
check was scheduled for November 11 00:14 UTC / November 10 16:14 Pacific.
Durable health observations and comparisons are under the run's `health/` directory. Raw launch/process/source
receipts are in `update8h-{preflight,resume-receipt}.json` under the run.
The source and clock implementation were committed before freezing/launch.

GPU telemetry note from the November 11 01:17 UTC health check: WSL NVML can
return an empty process list while the authenticated trainer owns `/dev/dxg`
handles and updates advance with active GPU utilization. An empty list alone is
not release evidence. Cross-check process boot/birth/parent and GPU handles,
three lock owners, stage/logs and advancing counters; preserve a healthy job.
The raw observation and independently authenticated check are in `health/`.

Philip subsequently authorized sustained resume to **28,800 TOTAL synchronized
optimizer-update seconds**, including the saved 2,873.748 seconds. His explicit
resume ends the migration pause. The original training-wall cap is removed:
input I/O, setup, checkpoint/export and native evaluation do not consume this
update allocation. Rejected update time remains charged, including repeated
rollback before another checkpoint. Stop after the complete update crossing the
cap; report any one-update overshoot.

Resume only the complete saved 20,671-update state, preserving parameters, all
Adam moments/group steps, RNG, order and cursor. The original recipe/config hash
stays unchanged. The source-pinned `scripts.he4fresh.clock.POLICY` and
[continuation policy](continuation-policy.json) explicitly override the original
execution clock; do not interpret the historical recipe's 43,200 wall seconds
as today's stop condition. No learning-rate/objective/model change is made.

New checkpoint/model labels are `update-3600` through `update-28800`; native
panels are `formal-update-*`. They cannot overwrite old wall-hour results or
resume receipts. The fixed target, external Q4 comparator, target-only
correctness, 2 warmups / 5 alternating reps / 256 tokens remain unchanged.
Historical zero, 60-second smoke and five wall-hour panels are preserved.

Use the source selected by `active-source.json`, and the exact supervisor state
and tmux session recorded by `gpu-owner.json`. The new source closure preserves
the already-qualified local loader and the mathematical update path. CPU checks
cover a wall clock far beyond the former cap, update-time stopping, distinct
panel names and cumulative repeated-rollback charges. Authenticate exact process
boot/birth and three locks before recovery. Preserve a healthy pinned source;
never create duplicate trainers. Check disk/headroom and local storage each time.
Re-enable the existing bound 30-minute health/repair event at real launch; keep
the old availability event disabled. A future explicit pause still takes
precedence. On completion, verify all new hourly panels plus historical results,
full-state GPU restore/native-byte recovery, durable exposure/time accounting,
artifact inventory, GPU/lock release, compact results and scoped commit/push.

## Earlier pause and completed localization (historical)

The following records describe the completed migration pause, before the above
explicit resume authorization. Their statements about keeping training paused
and health disabled describe that earlier phase.

Philip requested a safe pause, full local data transfer, lighter per-window
validation, a short old/new loading smoke, and retirement of old one-off models.
The complete paused state has 20,671 updates, 20,702.123 training-wall seconds
and 2,873.748 synchronized update seconds. The production recovery receipt was
unchanged by the scratch test. Sustained training remains **PAUSED**; the existing
30-minute health event remains disabled during this explicit pause.

All captures and current checkpoints/exports/accounting are now on local ext4.
`teacher-mount/` and `archive-mount/` retain their names to preserve the exact
checkpoint configuration, but are ordinary local directories, **not mounts**.
`local-data.json` records VERIFIED_LOCAL_READY; all 15,647 captures passed
one-time complete checksum/array/finite/alignment admission. Runtime reads local
feature bytes without repeated hashes, small label-array parsing or full-array
finite scans. It retains cheap length checks and the same global window order.

The original `source/` closure is unchanged. The separately frozen
`source-local-20261110-01/` passed six CPU loader checks and the actual matched
GPU diagnostic. `active-source.json` selects it with QUALIFIED_LOCAL_SOURCE.
A future authorized sustained resume must use `scripts/he4fresh/launch.py`
through `scripts/remote_job.py` and local tmux, preserving the existing complete
parameters, Adam, RNG, order, cursor and cumulative budget. Never reset optimizer
or adopt either diagnostic arm. Do not silently resume while pause is requested.
The health prompt already understands local storage and this source selector;
re-enable the existing event only on authorized real production launch.

Both scratch arms restored the same paused checkpoint, trained the same next
32 windows and produced exactly equal final parameters/Adam/RNG. Loader time
fell 26.695 → 0.485 seconds. The full 32-update loop fell 46.821 → 15.214 seconds,
including different first-update startup costs. Excluding the first update,
loader + update time fell 29.986 → 4.480 seconds (6.69×). This is a short loader
diagnostic, not an inference evaluation or sustained-run projection.
[Measured charts and qualifications](localization.md).

[Model retention policy and evidence](../storage-cleanup-20261110/report.md)
records 443 retired local model artifacts and the protected HE4/Csmallbroad,
foundation/B0 and W1A8/DSpark baselines. All 60 displaced evidence files from the
canceled cold offload were restored and verified; its obsolete copies and
private remote tree were removed. No retained artifact depends on that offload.

The old 2080 Ti dataset tree was removed only after the local admission and
matched GPU smoke passed, with no data consumers found. The remote current-run
checkpoint backup remains. The comparison mount was unmounted, both diagnostic
workers exited, all three GPU locks are free and the GPU has no compute clients.

The remaining sections preserve launch history and recovery requirements. Their
SSHFS/no-bulk-copy descriptions concern the **superseded original phase**;
Philip's explicit November 10 localization instruction replaces that storage
choice. The production recipe, target, objective and original budget stay fixed.

## Prerequisite and events

The prerequisite is Stage 6 thread
`mcp:launch:5cc79481d4d97b44b4c6cf5f1de491084f7f89c30d5802c9dbc55da66ceb13b2`.
It has been asked to message this training thread after all remaining phases,
audits and results are complete/pushed and its GPU processes are released.
The hourly availability fallback was disabled after real training started and the
30-minute training health event was enabled. Exact event receipts are
in `runs/he4broad-20261110-01/schedules.json`.

On a completion message or hourly event, check the prerequisite's latest evidence
under `/home/philip/vllm-semantic-cache-ext/evidence/stage6-5080-20261109/`,
actual GPU processes, host process birth/boot identities, pause control and
`runs/contender-gpu.lock`. Preserve the other campaign and its jobs.
The 30-minute training health/repair event is now enabled; its first check is
November 10 at 13:07:59 UTC. Both events are bound to this thread. No new worker thread is needed.

## Authorized recipe

- Fresh zero-update HE4 quantization of the authenticated base DFlash drafter,
  with fresh Adam. Do not load previous trained HE4 or HE4broad parameters or
  optimizer state. Retain a zero-update initialization/export proof.
- Reuse the qualified HE4 architecture: 18 W1 FFN projections, gate/up A4,
  down A8; Q4_0 fusion/attention; private tied Q4_K head/embedding; F16 draft KV;
  learned row scales/midpoints, input gains, clips and existing norms.
- Reuse the existing 15-position decay-weighted hard-target CE, seed 8101 and
  constant default group rates (sign 0.001; other groups 0.00001), unless a
  necessary compatibility repair is explicitly documented.
- Up to 43,200 actual synchronized update seconds. Keep setup, data transfer,
  checkpoint/export and evaluation wall time separate. Preserve cumulative
  accounting across repairs, with no fresh budget on resume.
- Every hour, serialize training/checkpoint/export and native acceptance plus
  complete-request throughput evaluations. Use the fixed Qwen3.5 4B Q4_K_M target,
  external DFlash Q4_K_M comparator and target-only correctness control. Use the
  existing consumed DEV cohort, with TRAIN overlap/exclusions checked. Record
  accepted/proposed counts, throughput, correctness, update/time and exposure.
  Label any reduced early diagnostic as smoke; retain formal zero/final panels.
- Keep early progress/loss and completed native results available on request.
  No deployment promotion is authorized by this run.

## New training data

Use the completed combined dataset on `pstack-2080`:
`/home/philip/he4-capture-2080ti/supplemental/combined-manifest.json`.
The final supplemental audit passes. The manifest declares 15,647 sequences,
1,568,435 overlapping windows, 1,787,054 label positions, of which 1,786,275
participate in training windows. These are distinct from repeated exposure.
It includes both the original 10,150 captures and all 5,497 supplemental captures.

The target hash matches the existing local Qwen3.5 4B Q4_K_M target. There are
eight pre-block-input taps (2,6,10,14,18,22,26,30), F32 features, int32 token and
label arrays, 15-label windows, F16 target KV and native greedy labels. Capture
stops at the first EOG label with no post-EOG rows. Preserve the manifest's
exclusion ancestry and authenticate every record/file before exposure.
Metadata copies belong under the run's ignored `input-metadata/`; compact input
pins belong in the tracked manifest. The old 40-sequence dataset is not eligible
as a silent fallback.

## Qualified preparation and clock

The new runner is `scripts/he4fresh`; the November 8 runner and its frozen runs
remain unchanged. Metadata admission on the data host verified all receipt,
metadata/source hashes and identity exclusions. The new bounded adapter uses the
verified native label positions, including both boundary windows omitted by the
older Capture range. Four CPU regressions cover final EOG-window alignment,
corruption rejection, metadata/array disagreement and bounded materialization.
First/last real captures and consumed-DEV prompt disjointness pass.

The read-only `teacher-mount` SSHFS mount exposes the existing remote captures.
The writable `archive-mount` exposes `/home/philip/he4broad-training-20261110-01`
on `pstack-2080`, with about 426 GiB free at setup. Compressed full states and
exports stay there. Write/fsync/rename and a lossless state round trip passed.
The current native model alone is copied into a bounded local cache for timed
serving, then removed after its panel. Never bulk-copy the capture trees here.
SSHFS was unpacked from the configured Ubuntu package under this run's tools/;
no personal settings or system packages were changed. Both mounts require the
existing private-key/verified-host SSH alias. Reconnect uses 15-second keepalives.

Training is bounded by **43,200 training wall seconds**, including feature I/O
and hashing, and also by 43,200 synchronized update seconds. Setup/checkpoint/
export/native evaluation are reported separately. This realizes the requested
up-to-12-hour training allocation without silently extending it for remote I/O.
Readouts occur at zero, an extra 60-second qualification, and every training wall
hour. All windows have a globally shuffled seed-8101 order; no sequence grouping
or smaller data fallback is used. Actual update time/exposure is reported so
remote I/O is never called model training throughput. Formal native panels run
at zero and every hour; the extra 60-second panel is labeled smoke.

The first metadata-index attempt had an anchor offset caught by the EOG-window
regression before any GPU exposure; it is retained as
`archive-mount/index-rejected-anchor-offset.json`. The corrected index hash and
source closure are pinned in the recipe/prepared receipts.

## Launch, monitoring and repair

Use the existing local `scripts/remote_job.py` supervisor and detached tmux on
socket `binary-eagle-runtime`. Record supervisor/child boot and birth identities,
source/config/data pins, commands, stage and stop procedure in this run directory.
One authenticated workflow owns the contender lock and serializes GPU work.

The dedicated 30-minute event checks progress, finite loss/gradients, checkpoint
integrity, expected eval/I/O transitions, ownership, disk and OOM/error evidence.
Repair routine failures within scope. Resume only complete authenticated states
from this new run, including parameters, Adam moments/steps, RNG and data cursor;
never reset to the old trained candidates. Preserve healthy jobs and failed
attempt evidence. At completion, verify actual full-state GPU restore/native
recovery, all requested hourly panels and correctness, record retention, stop
owned jobs, verify GPU release, publish/push compact results and disable events.

The training module loads source BF16 tensors and immediately applies the HE4
quantization/simulation. Fresh Adam and zero cursor are asserted; previous trained
candidate parameters and Adam are never read. Every hourly segment restores only
this run's full checkpoint, checking parameters, moments, group steps and RNG.
If a failed stage leaves uncheckpointed updates, recovery preserves their log and
charges their time to the existing budget before rolling back to the last state.

## Live run and stop/recovery procedure

The tmux session is `he4fresh-20261110-01` on `binary-eagle-runtime`.
Supervisor state is `runs/he4fresh-supervisor-20261110-01/state.json`;
its stdout is `stdout.log` in that directory. Full initial process birth/boot
identities, including both owned SSHFS processes, are in
`runs/he4broad-20261110-01/launch-receipt.json`. Current workflow identity and
three-lock ownership are in `gpu-owner.json`; current stage/command is in
`stage.json`. `health/latest.json` preserves read-only observations. The snapshot
helper is `.venv/bin/python -m scripts.he4fresh.health` in the canonical checkout.

Live progress and recoverable state receipts are under
`runs/he4broad-20261110-01/archive-mount/candidate/` (`live.json`, `result.json`,
`recovery.json`, `readouts.json`, `updates.jsonl`). The physically remote directory
remains available at the archive path on pstack-2080 if a local mount is lost.
Native hourly evidence stays under the local run (`formal-0`, `formal-3600`, etc.),
with an extra `smoke-60`. Stage logs/commands have unique attempt timestamps.

For an authorized stop, first match the supervisor's current boot/start identity
to its saved launch receipt, then SIGTERM that supervisor. It gives the complete
owned child group 600 seconds to stop between updates and save a paused full state.
Verify terminal supervisor/workflow receipts, original process-group absence and
all three GPU locks released. A pause flag alone is not release. Never stop another
project's owner or blindly signal a historical PID. Leave healthy work running.

After a failure, preserve its logs/incident receipts. Revalidate mounts, pins,
checkpoint/optimizer/RNG/cursor and budget, and ensure the old group is gone.
Use a fresh supervisor run ID and a free tmux session with the canonical Python,
`scripts/remote_job.py`, and
`--stop-grace-seconds 600 -- .venv/bin/python scripts/he4fresh/launch.py`.
The launcher enters the pinned source; completed readout/panel stages are skipped.
A failed native attempt is renamed and preserved before retry. Do not modify the
healthy pinned source. If a code repair becomes necessary after failure, preserve
that exact old source closure and explicitly pin the repaired closure before resume.

Repeated recovery before another complete checkpoint needs reconciliation of the
already retained `recovery-incidents/*/accounting.json` receipts as well as the
current uncheckpointed tail; the dedicated agent event must verify that rejected
exposure and both clocks remain charged to the original allocation. No blind
fresh initialization or budget reset is an authorized repair.

## Verified launch outcome

Four adapter regressions pass. All 75 zero-update formal native requests and
all nine early trained smoke requests match target output. The first trained
checkpoint retains 56 updates; actual GPU restoration verified parameters and
all Adam groups/moments/steps, restored RNG/cursor/order and completed update 57.
Later updates are advancing in the first hourly segment. Proofs are linked from
the manifest; the failed pre-exposure index attempt remains separate evidence.
The 30-minute event is enabled and the hourly availability fallback is disabled.
Readiness is verified launch/resume, not completion of the 12-hour allocation.

## Early learning-curve investigation

The [read-only setup audit](setup-audit.md) checks dataset exposure, immutable
optimizer state, binary-gradient support and saved native responses/counters.
It identifies input materialization as the dominant training-wall cost and the
three-prompt development panel as too narrow for broad saturation claims. It
does not change the active recipe, pinned source, clock allocation or schedule.

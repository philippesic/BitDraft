# HE4broad fresh training on the November 9–10 captures

State: **QUEUED, NOT TRAINING**. Philip authorized this run November 10 in
thread `6a9bee6a-3f4d-4d2f-87c5-73a5e841c6c6`. Wait for the other project's
complete campaign and verified RTX 5080 release. Its idle assistant turn or
an idle GPU between phases is insufficient. No Binary Eagle GPU job was launched.

## Prerequisite and events

The prerequisite is Stage 6 thread
`mcp:launch:5cc79481d4d97b44b4c6cf5f1de491084f7f89c30d5802c9dbc55da66ceb13b2`.
It has been asked to message this training thread after all remaining phases,
audits and results are complete/pushed and its GPU processes are released.
An hourly fallback is enabled in this training thread. Exact event receipts are
in `runs/he4broad-20261110-01/schedules.json`.

On a completion message or hourly event, check the prerequisite's latest evidence
under `/home/philip/vllm-semantic-cache-ext/evidence/stage6-5080-20261109/`,
actual GPU processes, host process birth/boot identities, pause control and
`runs/contender-gpu.lock`. Preserve the other campaign and its jobs.
Enable the already-created 30-minute training health/repair event at launch;
disable the hourly availability event only after that succeeds. Both events are
bound to this thread. No new worker thread is needed.

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

## Preparation still required

The old `scripts/he4broad` runner is pinned to the November 8 paths, 40 sequences,
eight hours, and its old readout cadence. Do not execute it unchanged. Adapt in
a new campaign module or explicit configuration without modifying frozen runs.
The new metadata references verification receipts and arrays, requiring an
authenticated adapter; the old eager Capture list would load the whole dataset
into RAM. Use bounded loading/cache/streaming while preserving the selected
window order and exact feature/label semantics.

Storage is currently tight: approximately 2.4 GiB free on WSL root and 12 GiB on
Windows when inspected. The original capture tree alone is 215 GiB, with the
supplemental tree additional. Do not bulk-copy it here or reuse the old physical
reservation scheme, which needs many GiB per hourly checkpoint. Qualify a bounded
remote data/cache and recoverable checkpoint storage plan before GPU launch;
the configured Tailscale SSH alias works. Keep useful existing artifacts intact.
Storage setup is preparation, not measured training or an excuse to change data.

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

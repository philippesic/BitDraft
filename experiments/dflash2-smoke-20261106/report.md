# DFlash2 smoke and Csmallbroad continuation

Exact continuation retry is active after final-state verification rejected attempt02. No native performance conclusion yet.

The user authorized a quick native DFlash2 comparison across the frozen four
Qwen3.5 target cells and one1,800-second actual-update continuation of retained
Csmallbroad. Parent owns main integration and the queued external message.

## Contract

Two same-process warmups per cell and two alternating measured process
repetitions on the three already-consumed code/reasoning/prose DEV captures.
Context4096, output cap256, greedy T0/seed1234, one request at a time,
cache_prompt=false, flash attention on, F16 target KV and consistent decode32.
Baseline depths2 for4B Q4 and3 otherwise. Target-only references are recomputed
in each target cell; NVFP4 MTP metadata does not enable MTP execution. This is a
smoke screen, not the full formal panel or an incumbent promotion.

All methods within a target cell use one common qualified runtime. A changed
runtime requires fresh comparators. Accepted/proposed/prefix counts diagnose
proposal quality; full request tokens/time establishes observed performance.
Clean latency, instrumented host spans, GPU forward and residency remain separate.
Unsupported cells remain explicit rather than receiving fabricated zeroes.

## Continuation

Restore the original Csmallbroad4096 state with104 master parameters, full Adam,
RNG/order/cursor and identical eligibility/learning rates. Train on the original
broader40 native TRAIN sequences,3234 overlapping windows/3794 unique label
positions. R1 output compensation, frozen head, no added capacity. One1800-second
synchronized forward/backward/update budget; setup, qualification, evaluation,
export and save are excluded. Native readouts near0/300/900/1800seconds. Preserve
all original frontier states and HE4. Before/after is a continuation recipe
comparison; no extra unchanged30-minute control is allocated.

Raw root: `/home/philip/binary-eagle-decoding/runs/dflash2-smoke-20261106-01`.
Supervisor: canonical `runs/dflash2-smoke-train-20261106-02/state.json`.
Owned tmux socket `binary-eagle-runtime`, session
`dflash2-smoke-train-20261106-02`. Graceful stop must authenticate process and boot
identity, signal its supervisor, allow checkpoint save, then verify release.

## Storage

[Retirement receipt](diagnostic-retirement.json) records the exact obsolete
one-step diagnostic retired under the user cleanup authorization. It is now
unresumable. Its source/configuration/results/native exports survive. Neither
original retained frontier nor B0/captures were deleted. Windows free space rose
from4.71GB to9.59GB before the run. This is physical free space, not guest VHD
compaction. New final state is reserved under
`/mnt/c/binary-eagle-dflash2-smoke-20261106-01`.

## Preserved failed attempt

Attempt01 completed300.008221seconds/1562actual updates, then its first readout
failed because the new adapter omitted the export state's requiredversion field.
Its masters were not saved; these updates are lost and do not belong to the
completed continuation. Source and receipts survive in Git and canonical
`candidate-attempt01`. Attempt02 restarts from the original paid4096fullstate.
The corrected adapter supplies version and lineage and checkpoints at a failed
readout boundary. Report failed-attempt compute separately from successful1800s.

## Release and native format qualification

[nchapman/Qwen3.5-4B-DFlash2](https://huggingface.co/nchapman/Qwen3.5-4B-DFlash2/tree/8d8bb62873a4fe6165e644163e9ff76e06ab78ba)
provides the genuine five-layer DFlash2 checkpoint with dynamic convolutions,
rank256/top-k16 selector and full248320vocabulary. The authenticated weights are
also the [vLLM CI copy](https://huggingface.co/vllm-project/Qwen3.5-4B-DFlash2/tree/50d7919bb821601f23301bc72d2c34f4046a055d).
This is a community checkpoint trained against a GPTQ Qwen3.5-4B verifier;
transfer to these fixed native target quants is measured here, not assumed.

The adapter flattens speculators metadata, preserves tensor bytes and converts
auxiliary hidden-state indices to the existing native converter's layer convention.
Native taps remain2,10,17,24,29. The ordinary draft uses its original eight taps.
Native metadata/tensor audit confirms all selector/conv tensors, unchanged
vocabulary, Q4_K_M draft quantization and identical sharedQ6_K vocabulary bytes
across4B Q4 and NVFP4 targets. The build uses retained runtime04 and its CUDA
correctness path; every target cell still requires fresh runtime/output qualification.
No9B checkpoint was found in the bounded HF/model-owner inventories recorded in
`release-audit/bounded-9b-search-summary.json`; this is an explicit no-matched-
release blocker, not proof that every possible repository lacks one.

## Completed continuation budget

[Machine-readable receipt](continuation.json):9,923additional updates,1,800.014628s
of synchronized forward/backward/optimizer update time,0.014628sone-update
overshoot. Cursor/Adam advanced4096→14019. Exposure148,845supervised positions
from the same3,794unique labels/3,234overlapping windows/40TRAIN sequences;
there are no fresh labels. Original optimizer groups/learning rates, schedule,
RNG and order/cursor were restored. Process wall2140.846s; setup/qualification
155.947s and final streamed checkpoint60.659s, with readouts and loop/state
construction overhead separate from update accounting. Earlier failed attempt:
300.008s/1562lost updates; it does not contribute to the final trajectory.

Native exports at0/300/900/1800actual-update seconds survive. The final4.2GB-class
streamed full state is retained under the new Windows directory; its complete
archive/raw hashes and exact bytes are in the receipt. Trainer exited0 and GPU
returned to its1102MiBdesktop baseline before the measurement supervisor launched.
Actual next-update/native byte recovery remains pending at this milestone.

## Attempt02 rejected by final recovery

The above1800.014628-second trajectory is **INVALID for exact resume**. Final
archive inspection found Adamstep14020, progress14019. PyTorch's non-capturable
Adam loading preserves CPU scalar tensors by reference: scratch qualification
advanced the caller's step4096→4097 before the main trainer loaded it. Moments
and disk parent were unchanged, but the main trajectory used the wrong Adam
bias-correction counter. Decrementing the final counter would not repair history.
The worker rejected this trajectory and preserved its full4.245GBstate, raw logs,
configs and all native byte streams. Owned redundant raw native exports were
losslessly archived against their frozen zero export to keep headroom; complete
roundtrip hashes were checked. `invalid-attempt-retention.json` records them.

The fix clones CPU scalar optimizer tensors before scratch restore, checks the
caller's complete Adam fingerprint remains unchanged, validates startingsteps4096,
and verifies every live update's actual Adamsteps equal its progress count. A CPU
Adam regression passed. Attempt03 restarts from the immutable original4096state;
this is a corrective execution of the same selected continuation, not an extra
control/architecture experiment. Attempt02's1800.014628s and attempt01's300.008221s
are failed/invalid cost, separate from the accepted1800-second trajectory.

Attempt03 stores only one final full state (or one paused state), keeping
intermediate native exports in authenticated lossless archives against its zero
export. No original paid frontier, parent, data or old runtime/source was removed.

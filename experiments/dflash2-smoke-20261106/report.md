# DFlash2 smoke and Csmallbroad continuation

Execution in progress; no performance conclusion yet.

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

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
Supervisor: canonical `runs/dflash2-smoke-train-20261106-01/state.json`.
Owned tmux socket `binary-eagle-runtime`, session
`dflash2-smoke-train-20261106-01`. Graceful stop must authenticate process and boot
identity, signal its supervisor, allow checkpoint save, then verify release.

## Storage

[Retirement receipt](diagnostic-retirement.json) records the exact obsolete
one-step diagnostic retired under the user cleanup authorization. It is now
unresumable. Its source/configuration/results/native exports survive. Neither
original retained frontier nor B0/captures were deleted. Windows free space rose
from4.71GB to9.59GB before the run. This is physical free space, not guest VHD
compaction. New final state is reserved under
`/mnt/c/binary-eagle-dflash2-smoke-20261106-01`.

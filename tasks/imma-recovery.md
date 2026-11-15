# Recover IMMA and preserve evidence

## Goal

Test the November 4 tensor-core implementation on S4 and relocate retained artifacts
to the main checkout with SHA-256 verification and successful replay.

## Authorized

Owner requested recovery, 4/8-slot exact tests, served confirmation if faster,
verified relocation and removal of verified worktree copies. No subagents, merge,
results entry or FINDINGS edits. Ten quiet minutes before heavy work; GPU lease only
for GPU work. Preserve other jobs and pause CPU work for their leases.

## Done when

Recovered/adapted variant measured and exact-checked, served test if applicable;
all retained artifacts copied, individually verified, relocated specifications
replayed, old copies removed through verified inventory, receipt retained. Gate
passes and branch pushed.

## State

Found the actual definition in native-source-support.tar.gz, original path
runs/w1a8-algorithms-20261020/residual30-native03/ggml/src/ggml-cuda/w1a8_algorithms.cuh.
SHA-256 4472b75a9b82c95040681a30a014823316dfc9f5e9625201ec06611971ef2a6e.
Archive index and recovered source retained under artifacts/benchmarks/imma-recovery.
Four warps split the reduction dimension; vector loads fetch four sign words at
once. Adaptation replaces seven-slot constants with bounds and uses exec_affine.
Signed and original unsigned-long-row modes, plus one-warp ablation planned.
Blocking ten-minute quiet waiter is active; only light work until it finishes.

## Next

Compile and qualify, then served confirmation if faster. Relocate all original,
RSR and new evidence with receipts, test replay, gate and push.

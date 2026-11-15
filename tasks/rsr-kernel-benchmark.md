# RSR-core GPU qualification

## Goal

Test the pinned public binary RSR-core GPU kernel on S4 gate, up and down against
the dispatched runtime, with exact integer and final-byte checks at 1, 2, 4, 8 slots.

## Authorized

Owner requested item 3 after the original kernel benchmark was delivered and pushed
at 5f2f1556. No subagents, no default runtime or model changes, no results entry.
Only light reading/editing during other GPU jobs. Wait for ten quiet minutes, and
pause CPU work whenever another GPU owner starts. GPU execution holds the lease.

## Done when

All tables and slot counts measured with integer equality, index sizes and k reported;
answer whether four slots beat the dispatched path. Full gate passes, commits pushed.
Retain source/licence, indexes, captures, build identities, timings and failures in
ignored artifacts. Remove this task only after completion.

## State

Pinned source 99bcccded70416839f1ef1fc6d58726c507d1fb9 is independently fetched under
`/home/philip/BitDraft/artifacts/native/rsr-core-upstream-20261114`; MIT, IndexLab 2026.
Installer/build and selected device source reviewed, no upstream code executed.
Authored scripts remain separate. Source review is retained under the main checkout's
`artifacts/benchmarks/rsr-core-audit-20261114` and this worktree's `artifacts/benchmarks/rsr-kernels`.
Read-only kernel/licence inputs are hardlinked into this worktree's artifacts for
maintained builder authentication. No upstream installer or Torch JIT import is used.

Implementation in progress: literal v5.9 device kernel, its rectangular wrapper's
256/512-thread policy, own offline counting-sort index construction, exact signed
conversion W=2B-1, original DP4A with unit affine parameters as integer reference,
complete original affine output checks. Test blocks 4, 8, 12; single-vector upstream
kernel repeated per slot. One-slot input is the first slot of a captured two-slot DEV
activation. CPU preprocessing uses a separate executable with no CUDA dependencies.

## Next

Finish tests/docs and build after the blocking quiet wait. Measure one slot first,
then 2/4/8, preserving failures. Compare joint gate/up against real fused dispatch as
well as individual gate/up/down calls. Commit and push checkpoints.

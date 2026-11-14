# One-bit kernel benchmark

## Goal

Measure exact one-bit feed-forward kernel alternatives on S4 at 2, 4, 8 and 16
slots, compared with Baseline Q4_K through runtime-v7's ordinary dispatch.

## Authorized

User-requested implementation and GPU measurement on this branch. No subagents,
no default runtime/reference changes, no PR, no results page or FINDINGS edit.
Only light editing while another GPU lease owner is active. Require ten consecutive
minutes of free lease before builds/checks/timing; pause CPU work if another owner
starts. Hold the lease only for GPU execution. No waiting timeout.

## Done when

Maintained documented command and tests pass the full gate; every required variant
has measured times and full output-byte comparisons; coherent commits pushed.
A clear four-slot winner additionally requires the separately named served experiment.

## State

Owner: implementation agent, branch `t3/benchmark-w1a1-kernels-3`.
Scaffold checkpoint `040afa06` is committed and pushed. Its full gate passed with
130 CPU tests and 92 rule proofs. The leased GPU smoke completed at 24.749504 µs
for one gate matrix at four slots with synthetic codes; it is not final performance
evidence. Its process elapsed time was 0.47 seconds.

The maintained harness build is `artifacts/native/kernel-benchmark-v7-03`; earlier
builds remain as receipts. The quick check passed 144 tests before adding Q6_K
payload fixtures. Actual DEV inputs for all six layers and four slot counts are in
`artifacts/benchmarks/w1-kernels/dev-capture-01/result.json`. The full reference and
Baseline sweep in `reference-02/result.json` passed all 120 byte checks. Baseline
down matrices in layers 2 and 5 are Q6_K; the other feed-forward matrices are Q4_K.
Initial warmup was increased to 10000 launches after a noisy preliminary smoke
case; both preliminary and final samples are retained. The full extended gate passed with 147 tests and 92 rule proofs;
Cross-slot DP4A in `cross-01/result.json` passed all 48 group checks but was slower
than the dispatched reference at every tested slot count. Original MMA and the
authored tensor-core variants remain unqualified. Required variants plus broadcast-MMA
and one-warp-MMA are compiled. No waiter is currently active.

Read-only inputs are hardlinked into this worktree: runtime-v7 source, compatibility
headers, libraries and receipt, S4 export, Baseline, target and DEV workload. No
original input or default runtime was edited. Draft checkpoint backups are retained
under `extensions-backup*`; current source contains later edits and is authoritative.
`artifacts/benchmarks/w1-kernels/next-steps.txt` contains the original execution sequence;
its initial scaffold commit, push, smoke and helper build steps are now complete.
Specifications, smoke outputs and operational logs are under
`artifacts/benchmarks/w1-kernels/`; maintained helper build outputs are under
`artifacts/native/kernel-benchmark-v7-03/`. New measurement/capture outputs remain
under `artifacts/benchmarks/`. Original exports, models and runtime-v7 are read-only inputs.
Retain measurement receipts and exact build/input identities; generated copies and
binaries are reproducible scratch, eligible for retirement after final verification.
The maintained-command checkpoint includes the passed full gate and reference sweep.
The original smoke overlapped another CPU build and is excluded from performance
evidence; `smoke-clean.log` repeats it after that build ended.

## Next

Run the full extended gate and push the maintained-command checkpoint, then qualify
successive variants with full byte identity and repeated timing. Commit and push
each meaningful checkpoint. Profile MMA separately
from clean timing; apply the conditional served-runtime step if a four-slot winner
emerges.

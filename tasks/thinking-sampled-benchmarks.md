# Thinking and sampled scenario benchmarks

## Goal

Extend three-level DEV evaluation with thinking, sampling, grouped training diagnostics,
per-scenario decisions, saved controls and a protected request confirmation reserve.

## Authorized

Implementation sub-agent assignment on `t3/thinking-sampled-benchmarks`; commit locally,
no push or merge. About 1.5 GPU hours on the shared RTX 5080 under the host lease; wait
for the agentic suite owner. New artifacts stay under the main checkout's
`artifacts/benchmarks/scenario-levels-01/`, under 15 GB. Do not consume heldout24 or aliases.
Do not run confirmation requests. Preserve existing references and the campaign screen.

## Done when

Thinking and sampled 40-request cohorts and their 12-request slices have both-precision
three-arm evidence and references. Final export `405e326b…` has references on the four
routine cohorts alongside S4. The unchanged 120 training diagnostic windows have separate
20-window agentic/thinking additions with measured cost. Scenario reports and verdicts,
120 protected fresh confirmation requests, result/FINDINGS documentation and full gate pass.

## State

First implementation checkpoint `8d9e772d`; full gate passed at that checkpoint. New
report validation tests pass; the latest full check passed 136 tests and all 92 proofs
but found one import-order issue, now fixed. Q4KM thinking suite completed in 797.6 s,
all three arms exact on all 40 requests. Both runtime-v7 sampler verifications passed.
The 160-window diagnostic measurement preserves all original 120-window numbers
exactly; supplementary groups cost 10.6 s. The complete resident training pause is
not yet qualified. Confirmation has 120 frozen requests and 12 validated, unexecuted
specifications; `confirmation/protection-v2.json` is the future exclusion artifact.

Only the NVFP4 thinking panel currently has GPU allocation. On completion release it
and notify the agentic owner: their all60 target-only calibration has next priority.
Keep `gpu-yield` present; do not queue more GPU work until coordination permits it.
Inspect `queue-owner.json`, `controllers/`, `queue-source.txt`, `logs/queue.log` and actual
GPU clients before resuming. A ten-minute healthcheck monitors this task.
All paths are under the main checkout's `artifacts/benchmarks/scenario-levels-01/`.
Frozen source and scripts reproduce the work. Raw panels/captures, source snapshots,
models and registry dependency closures are retained pending compact result extraction.
This task authorizes no deletion of shared data or existing references.

## Next

Register NVFP4 thinking evidence. After the agentic calibration and coordinated release,
run both sampled suites and both-precision final-drafter routine checks. Qualify the
complete extended training pause, register references, finalize result/FINDINGS and
measured-cost documentation, run the full check and commit locally. Account standalone
time receipts as well as queue-progress when enforcing the 1.5-hour GPU budget. Preserve
failures and report any allocation limit honestly.

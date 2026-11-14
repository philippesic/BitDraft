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

Implementation checkpoints `8d9e772d` and `40c4d5bd`; the latter full gate passed 136 tests
and all 92 proofs. Current memory-in-scenario reporting changes pass the configured partial
check. Both thinking suites completed exactly on all 40 requests in all three arms:
Q4KM 797.6 s, NVFP4 630.2 s. Twelve shared reference records cover both suites and slices.
Both sampler verifications passed. The 160-window diagnostic measurement preserves all
original 120-window numbers exactly; supplementary groups cost 10.6 s. The complete
resident training pause is not yet qualified. Confirmation has 120 frozen requests and
12 validated, unexecuted specifications; use `confirmation/protection-v2.json` as its
future exclusion artifact. Prior protocol-v3 reports were regenerated through the maintained
scenario-report command; neither format establishes faster under the new rule.

The lead authorized only Q4KM sampled next after agentic target-only calibration ended.
Bounded controller PID4168453 runs `q4km-sampled-suite`; inspect `queue-owner.json`,
`logs/queue-q4km-sampled.log` and actual clients. It has no subsequent allocated job.
`gpu-yield` was restored after it started. On completion notify the agentic owner and
give them the next slot if waiting. Do not start NVFP4 sampled or routine checks without
coordinating. The launcher accepts named jobs after its frozen source argument.
`report-source.txt` points to current source. The ten-minute healthcheck next wakes
at 19:30 UTC on November14.

All relative paths above are under the main checkout's
`artifacts/benchmarks/scenario-levels-01/`. Conservative compute used is about 1816 s of
about 5400 s; task artifacts occupy 3.0 GiB. `compact-draft/` and `result-draft.md` are interim
extractions, not the final result. Frozen source and scripts reproduce the work. Raw
panels/captures, source snapshots, models and registry dependency closures remain retained.
No shared data or existing reference deletion is authorized.

## Next

Finish the bounded Q4KM sampled allocation. After coordinated release, run NVFP4 sampled
and both-precision final
routine cohorts, one named allocation at a time. Register all complete evidence with
`scripts/register_completed.py`. Qualify the complete extended training pause using
`scripts/qualify_pause.py` after Q4KM screen final is registered. Keep total new artifact
size below 15 GB; its two disposable states need an explicit Store retention decision.
Finish result/FINDINGS and measured-cost docs; regenerate compact evidence and maintained
reports; run full gate and commit locally. Count standalone receipts and queue-progress
as well as raw panel times. Preserve failures and report any allocation limit honestly.

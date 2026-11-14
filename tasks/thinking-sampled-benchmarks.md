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

Implementation checkpoints `8d9e772d`, `40c4d5bd`, `2e64d3c3`; latest committed full gate
passed 136 tests and 92 proofs. Pending small report changes fix historical-direction
escalation and describe potentially unequal sampled paths; focused tests pass.
Both thinking suites completed exactly on all 40 requests in all three arms: Q4KM 797.6 s,
NVFP4 630.2 s. Q4KM sampled completed all 360 rows in 1658.5 s (1661.9 s outer allocation).
HE4 194.1 tok/s, target 153.3, Baseline 203.4; HE4/target 1.266 (1.233–1.310), HE4/Baseline 0.954
(0.945–0.963). All verifier/settings/response checks passed. Eighteen canonical reference
records cover completed thinking and Q4KM sampled suites/slices. Native policy uses greedy
draft proposals and target sample-and-match; sampled paths can coincide under fixed seeds.
`sampler-policy-check.json` records source/argument inspection; `sampled-path-check.json`
checks completed raw paths without making equality a sampled validity requirement.

The 160-window diagnostic measurement preserves all original 120-window numbers exactly;
supplementary groups cost 10.6 s. Complete resident pause qualification remains pending.
CPU admission for its two-update isolated branch passed: cosine end extends 80209→80211,
with multipliers 1.982e-9 and 4.956e-10. The old idea of updating beyond the existing cosine
end was rejected during inspection before execution. `pause-admission.json` records this.
Confirmation has 120 frozen requests and 12 validated, unexecuted specifications; use
`confirmation/protection-v2.json` for future exclusions. Previous protocol-v3 reports were
regenerated through scenario-report; neither format establishes faster under the new rule.

Q4KM sampled controller 4168453 and lease 4168457 exited. Agentic owner has priority now
for Q4 tail 3, NVFP4 target-only 60, then their reference panels (last lease 4179157). Keep
`gpu-yield` and coordinate before more GPU work. The launcher accepts named jobs after
its frozen source argument and allocates only those names. `report-source.txt` points to
current source; inspect actual clients and recorded owners before launch. A ten-minute
healthcheck remains (next 20:00 UTC November 14 at last update).

All relative paths above are under the main checkout's
`artifacts/benchmarks/scenario-levels-01/`. Conservative compute used is about 3478 s of
about 5400 s; task artifacts occupy 3.0 GiB. `compact-draft/` and `result-draft.md` remain
interim extractions. Frozen source and scripts reproduce the work. Raw panels/captures,
source snapshots, models and registry dependency closures remain retained. No shared data
or existing reference deletion is authorized.

## Next

After coordinated release, run NVFP4 sampled and both-precision final routine cohorts,
one named allocation at a time. Register complete evidence, check saved specifications
and sampled path agreement with scripts under `scripts/`. Qualify the complete extended
training pause using `qualify_pause.py` after Q4KM screen final is registered. Check the
remaining ~32-minute GPU budget against observed NVFP4 and routine costs before proceeding.
Keep new artifacts below ~15 GB; its two disposable pilot states need an explicit closed
Store retention decision and receipt after compact extraction. Finish result/FINDINGS and
measured-cost docs; regenerate compact evidence and maintained reports; full gate and local
commit. Preserve failures and report allocation limits honestly. No push or merge.

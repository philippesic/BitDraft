# HE4broad supplemental data and continuation

## Goal

Generate the frozen 2K supplement and run two hours of actual model-update time on
the hybrid corpus, continuing original220095 with Adam/RNG. The benchmark/reference
qualification is [complete](../results/reusable-acceptance-checks/result.md).

## Authorized

RTX 5080 work, scoped FP16 capture, exactly the frozen 2000 TRAIN selection, and continuation
of original 220095 including Adam/RNG. Preserve other jobs. Reclaim obsolete large artifacts
after ownership, dependency and evidence checks. No hosted CI, new remote, full 6K capture,
heldout24 use or substituted qualification descendant. Philip now authorizes two hours
of model updates on the hybrid dataset; capture, data loading, checks and saves are separate.

## Done when

All 2000 new records pass the capture/product loader. The 17647-task mixture is role-safe,
uses task-uniform sampling and passes public preflight. Production continues original 220095,
preserving Adam/RNG, and finishes7200 optimizer seconds with validated native results.
Keep operational/scientific health receipts and preserve unfavorable or failed results.
The existing operational update guard is 220095+100000; do not reset or extend it silently.

## State

Artifact root `/home/philip/binary-eagle-artifacts`; FP16 supplement capture is running
in `datasets/he4broad-supplement-01` from the exact frozen2000-ID specification. The GPU
allocation belongs to this task. No production training worker has started yet.
FP16 storage passed scoped HE4/Q4KM qualification. Original 15,647 records physically live
in `datasets/retained-he4broad-original`; the old path remains a compatibility symlink.
`training/retained-corpus-intent.json` maps directly to the physical original data.
Original complete 220095 and its normalized parent remain preserved.

Reference registry: `benchmarks/references`. Qualified controls exist for Q4KM and NVFP4;
`qualification/lean-checks/broader-qualification-assertions.json` identifies the complete
candidate-only suites. Production uses 60×128-token screens, 1800–3600 optimizer seconds
between checks and a 120-second complete pause. Saved target/baseline/initial references
avoid rerunning those models. Recovery, acceptance leader and full-suite rankings differ.

The two-hour cumulative optimizer stop is being integrated with resident campaign checks.
The obsolete full 600×2
schedule stays disabled. `qualification/prepare-hybrid-production.py` is a disabled older
recipe; revise it to V2 before use, preserving its refusal until replacement validation.
Closed unselected handoff checkpoints were retired with Store receipts; originals,
selected states, exports, raw executions and frozen source remain preserved.

## Next

1. Use `qualification/capture-supplement-spec.json` for only the frozen 2000 TRAIN IDs
   into `datasets/he4broad-supplement-01`, resuming verified records after routine failures.
2. Prepare the role-safe 17647-task mixture and V2 production intent/campaign through
   maintained preflight. Bind original 220095, current executed source, saved controls,
   the verified initial export/reference, explicit global budgets and the correct dataset.
3. Validate newly captured size extremes before production. Then use
   `training/he4broad-hybrid-01` and `training/he4broad-hybrid-campaign.json`; keep hourly
   health receipts externally. Reclaim closed unselected artifacts if storage is needed.

# Current project status

## HE4broad source-initialized QAT

November 8 user-authorized new recipe: **RUNNING**, 211 completed updates, 36.792 / 28,800 actual update seconds. Source BF16 initialization; broader authenticated 40/3234/3794 TRAIN set; fresh Adam at unchanged default rates. Hourly recoverable states and native exports retained. [Report](../experiments/he4broad-20261108/report.md) · [Live progress](../runs/he4broad-20261108-01/candidate/live.json) · [Ownership and stop procedure](../experiments/he4broad-20261108/operations.md). External DFlash Q4_K_M remains the deployment incumbent; no promotion.

## Completed: unchanged HE4 overnight continuation

November 7–8: exact HE4-confirm resume advanced from 4,096 to **173,193 updates**
using **21,600.017 additional update seconds**. The recipe and data stayed frozen.
On consumed DEV3, acceptance fell **22.61% → 17.85%** and complete-request
throughput fell **170.56 → 160.51 tokens/s** (0.9411×; paired timing interval
0.9338–0.9470). External DFlash Q4_K_M reached **251.58 tokens/s** and remains
the deployment incumbent. This fixed-data continuation did not demonstrate useful
native headroom; it does not establish a general HE4 capacity limit.
[Report, intermediate readouts and limits](../experiments/he4-continuation-20261107/report.md).

Actual parent/final GPU resume and native-byte recovery pass. Independent audit
reconciles 169,097 updates and 87 measured native request rows; all 127 requests
including warmups match target output. Eight new full states, nine native exports,
the original parent and all required sources/captures remain retained.
[Audit and hashes](../experiments/he4-continuation-20261107/audit.json) ·
[Retention](../runs/he4-continuation-20261107-01/milestone-retention.json) ·
[GPU release](../runs/he4-continuation-20261107-01/gpu-release.json).
All owned jobs and the contender lock are released. No further GPU work is assigned.
No mixed-weight work or deployment promotion was performed.

## Development priority: HE4; Csmallbroad fallback

User decision, November 6: prioritize **HE4** for future binary-drafter development
toward beating external DFlash Q4_K_M request throughput. Preserve **Csmallbroad**
as the compact fallback, especially if memory or longer-context costs become
limiting. [Decision and exact retained recipe](DECISIONS.md#2026-11-06-prioritize-he4-retain-csmallbroad-fallback).
This selects the development direction; it grants no additional GPU run or
training budget and does not promote HE4 over the external deployment incumbent.
Deferred next precision idea: [selective ternary/2-bit FFN weights](DECISIONS.md#2026-11-06-revisit-mixed-ffn-weight-precision-later),
with HE4's activation widths held fixed. Record only; no sweep launched.

## Completed: precision campaign and DFlash2 follow-up

Both user-authorized investigations are complete. Source, compact findings,
figures and reusable runners are integrated; paid models/captures/raw evidence
remain outside Git. External DFlash Q4_K_M remains the throughput incumbent.
Those investigations assign no further GPU work; the separate bounded HE4
allocation above is the current task.

[Component campaign](../experiments/component-campaign-20261105/report.md):
28 full 4,096-update windows, 118 readouts, 1,680 formal measured requests and
actual retained-state GPU resume/native recovery. HE4 is the faster-draft
endpoint; Csmallbroad is the smaller-cache endpoint. Neither beats external Q4.

[DFlash2 smoke and continuation](../experiments/dflash2-smoke-20261106/report.md):
genuine DFlash2 measured for 4B Q4_K_M and NVFP4 with five FFN down matrices
stored/executed F32. Original Q4 and F16 paths are invalid due to nonfinite logits.
Both 9B target baselines ran; no matched DFlash2 release was found in the bounded
search, and no 9B candidate weights exist. Those cases are explicitly unsupported
or untested. Ordinary DFlash Q4 leads the consumed-DEV smoke in all four cells.

Csmallbroad's accepted continuation used 1,800.073 actual update seconds and
9,998 updates, with exact resume from its original state. Actual next GPU update
and native byte recovery pass. Before/after throughput has no consistent gain.
Export failure and optimizer-state aliasing caused two rejected attempts costing
another 2,100.023 update seconds, separately retained and excluded. The scalar
aliasing regression is fixed. Parent recomputed all 120 measured native request
rows and the accepted update log, verified seven handoff hashes, and passed eight
regression checks. [Review receipt](../runs/dflash2-smoke-20261106-01/parent-review.json).

## Handoff

All owned GPU jobs and passive successors are released. The two component
frontiers, original parents and new fully resumable continuation remain retained.
See [preservation](../runs/dflash2-smoke-20261106-01/milestone-retention.json) and
[GPU release](../runs/dflash2-smoke-20261106-01/gpu-release.json).
User-authorized November 6 cleanup retired 34 superseded optimizer checkpoints:
63.57 GiB in WSL and 61.59 GiB directly on Windows C. Observed free space is now
about 66.20 GiB in WSL and 63.16 GiB on C. The five current full states/parents
remain hash-verified and unchanged; selected native exports were reverified.
[Retirement paths and preservation](../experiments/storage-cleanup-20261106/report.md).
Historical checkpoint references to these explicitly retired files are not live
resume claims. Preserve the retained states, source/dictionary dependencies and
captures. This cleanup grants no new training allocation.

The exact queued shader benchmark message was delivered to the
[requested thread](<t3-thread://v2/de4b951c-7cc7-4d2d-bbb0-1f3e6d7f79f7/aba88598-10c1-4c3c-b760-9aa7a0353548>)
after review/integration/push and fresh GPU release checks. Delivery started a new
turn there. Delivery identity and the unchanged message are in the
[handoff receipt](../runs/component-campaign-20261105-01/pending-thread-message.json).
Do not resend. That separate shader investigation now owns its next actions;
that handoff does not assign this thread any shader work. The separate HE4
allocation above is the current binary-model GPU task.

## References

[Benchmark protocol](CANDIDATE_BENCHMARKING.md) · [Stages](CANDIDATE_STAGES.md) ·
[History](HISTORY.md) · [Operations](AGENT_OPERATIONS.md) · [Git and artifacts](GIT_AND_ARTIFACTS.md).

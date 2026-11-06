# Development history

These records explain previous work, not current instructions. [STATUS](STATUS.md)
and its linked campaign define current work. Historical command/source paths may
require reconstruction and must not be used to resume jobs without fresh checks.

| Topic | Retained record |
| --- | --- |
| Recent precision and execution results | [November 5 status snapshot](history/STATUS_HISTORY_20261105.md) |
| Earlier development chronology | [Earlier status history](history/STATUS_HISTORY.md) |
| Decisions and rejected alternatives | [Decision archive](history/DECISIONS_20261105.md) |
| Original EAGLE assumptions and scope | [Original overview](history/PROJECT_OVERVIEW_20261105.md) |
| Former goals and checkpoints | [Goal archive](history/goals/README.md) |
| Old operational recipes | [Runbook archive](history/README.md) |

When revisiting an idea, identify its original model/runtime/hardware, what was
actually tested, outcome and limitations, and what changed enough to justify a new
attempt. Append meaningful results to the relevant history; do not copy monitoring
updates into multiple documents. Preserve raw evidence outside Git.

- [Retained precision charts, 2026-11-05](../experiments/precision-history-20261105/README.md): historical export with original source identities.

## Retired Mac workspace (2026-11-05)

Historical Mac source, native-runtime changes, worktree snapshots and compact
reports are preserved on this machine at
`/home/philip/project-archives/mac-retirement-20261105/binary-eagle-decoding/`.
See its README and verification receipt. This is outside Git; it is not current
campaign source. Large obsolete Mac artifacts were approved for retirement;
this archive does not promise exact old training-state recovery.


## 2026-11-06: remaining component precision campaign

Completed 28 physical 4,096-update windows and formal heldout/context panels.
HE4 lowers clean draft latency; Csmallbroad reduces cache/model payload. Neither
beats external DFlash Q4_K_M request throughput. Both retained full states
actually resume and reproduce evaluated native weights. Parent validated the
formal packet and raw heldout totals. [Report](../experiments/component-campaign-20261105/report.md).
DFlash2 matrix smoke and one 30-minute candidate continuation are authorized next;
the shader-thread handoff waits until that follow-up finishes.

## DFlash2 smoke and exact continuation (2026-11-06)

[Full comparison](../experiments/dflash2-smoke-20261106/report.md): genuine4B
DFlash2 required a declared F32 down fallback;9B had no matched release in the
bounded search. Ordinary DFlash Q4 remained fastest in all four cells. Exact
Csmallbroad+1800s continuation and final full-state GPU resume passed; native
before/after throughput did not improve consistently. Two failed/invalid attempts
and their additional compute are preserved. GPU ownership was released; parent
review/message delivery is recorded separately from worker completion.

## Storage retirement and deferred precision work (2026-11-06)

Recorded selective ternary/2-bit HE4 FFN weights as a deferred experiment, with
activation widths held fixed. No new training was launched. User-authorized cleanup
retired 34 superseded optimizer states: 63.57 GiB inside WSL and 61.59 GiB directly
on Windows. Current HE4/Csmallbroad states and required parents remain verified;
native exports, source and measurements remain. [Exact scope and outcome](../experiments/storage-cleanup-20261106/report.md).

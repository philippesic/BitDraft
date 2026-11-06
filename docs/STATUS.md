# Current project status

## Component campaign complete; DFlash2 follow-up next

The remaining head/embedding/drafter-KV/scalar campaign is complete on RTX 5080,
Qwen3.5-4B Q4_K_M: 28 full 4,096-update windows, 118 readouts and 1,680 formal
measured requests. Parent re-executed the formal packet validator, recomputed
heldout totals and verified handoff file hashes. Actual retained-state GPU
resume and exact native re-export passed. The worker released its GPU jobs.

[Report](../experiments/component-campaign-20261105/report.md) ·
[Compact results](../experiments/component-campaign-20261105/results.json) ·
[Full charts](../runs/component-campaign-20261105-01/charts/comparison.html) ·
[Completion receipt](../runs/component-campaign-20261105-01/campaign-complete.json) ·
[Parent review](../runs/component-campaign-20261105-01/parent-review-20261106.json).

HE4 has the faster draft: 0.995 ms clean GPU forward versus 1.179 ms external Q4;
heldout request throughput is 167.06 versus 236.67 tokens/s. Csmallbroad is the
smaller retained endpoint, saving 68.74 MiB of model/cache payload against its
matched broader-data control and reaching 168.95 request tokens/s. External Q4
remains the deployment incumbent. Controls, complete training recipes and
cohorts remain distinct; full HF/native graph parity is still unresolved.

## Authorized next action

Execute [DFlash2 smoke](DFLASH2_SMOKE.md): genuine DFlash2 across the frozen
4B/9B × Q4_K_M/NVFP4 target matrix, ordinary DFlash Q4_K_M comparisons, plus
one additional 1,800-second actual-update continuation of the best retained
candidate. Select Csmallbroad for that continuation using the native request
throughput criterion among retained binary candidates; preserve HE4 as the
faster-draft comparison. This is artifact/recipe selection, not an isolated
precision claim or proof of a significant win over HE4. Measure before/after
on consumed DEV; the old fresh24 cohort is now consumed and cannot be relabeled
sealed. Do not start another broad training sweep.

The previous campaign source checkout and pinned runtimes remain preserved.
Parent is integrating and preparing one follow-up GPU owner. Fresh ownership,
pause and storage checks precede launch; old running receipts from three WSL
restarts are historical, not live authority. See [component record](COMPONENT_CAMPAIGN.md)
and [operations](AGENT_OPERATIONS.md). All retained states/parents/data and
unpublished source must remain recoverable. The explicitly retired earlier
optimizer states are listed in the [retirement record](../experiments/precision-campaign-20261104/checkpoint-retirement-20261105.md).

After the DFlash2/candidate follow-up finishes and is reviewed, release the GPU
and send the exact queued shader benchmark message. The destination, message
and deduplicated key are in the [pending handoff](../runs/component-campaign-20261105-01/pending-thread-message.json).
The component campaign's completion alone does not trigger delivery.

## Reference and history

[Candidate protocol](CANDIDATE_BENCHMARKING.md) · [Stages](CANDIDATE_STAGES.md) ·
[History](HISTORY.md) · [Decisions](DECISIONS.md) · [Git and artifacts](GIT_AND_ARTIFACTS.md).

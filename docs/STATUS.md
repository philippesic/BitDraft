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

## DFlash2 follow-up complete; parent review pending

The implementation operator in `t3/dflash2-smoke-baseline-candidate-train`
completed the frozen four-cell smoke and released all owned GPU groups/passives.
Genuine DFlash2 runs on both4B targets with a declared five-matrix F32 down
fallback; the original Q4/F16 paths are invalid (nonfinite logits). No matched9B
release was found in the bounded inventory search, so both9B DFlash2 rows are
explicitly unsupported. Ordinary DFlash Q4 remains fastest in all four cells.

Csmallbroad's accepted continuation completed1800.073 actual synchronized update
seconds/9998updates,4096→14094. Final native byte recovery and actual GPU next
update pass with all104masters/Adam/RNG/cursor preserved. Before/after request
throughput changes−2.19% on4B Q4 and+1.54% on4B NVFP4; no consistent gain or
promotion. Two rejected implementation attempts cost another2100.023update
seconds and are preserved/separately reported. HE4 is unchanged;9B candidate
weights do not exist. This is consumed DEV smoke, not a fresh formal panel.

[Full report/charts/table](../experiments/dflash2-smoke-20261106/report.md) ·
[JSON](../experiments/dflash2-smoke-20261106/results.json) ·
[Continuation](../experiments/dflash2-smoke-20261106/continuation.json) ·
[GPU release](../runs/dflash2-smoke-20261106-01/gpu-release.json).
Raw/source/full states are retained with dependency inventories. Windows and guest
free space are low; original component states/source/runtime/charts remain intact.

Parent owns review/integration and the exact queued shader message after review.
**No external message has been sent by this worker.** No GPU job is active or
assigned by this status. See the handoff receipt for branch/source/push identity.

## Reference and history

[Candidate protocol](CANDIDATE_BENCHMARKING.md) · [Stages](CANDIDATE_STAGES.md) ·
[History](HISTORY.md) · [Decisions](DECISIONS.md) · [Git and artifacts](GIT_AND_ARTIFACTS.md).

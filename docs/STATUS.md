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

## Active DFlash2 follow-up

The sole follow-up GPU operator is the tool-created worktree
`/home/philip/.t3/worktrees/binary-eagle-decoding/t3-e6fc8d06`, branch
`t3/dflash2-smoke-baseline-candidate-train`. Parent owns review/integration and
external message delivery after completion and GPU release. No external message
has been sent by this worker.

The corrected exact continuation completed1,800.072727s/9,998additional updates,
4096→14094, with a live Adam-step identity check at every update. Native readouts
and a full state are saved; trainer exited0 and released the GPU. The two earlier
failed/invalid attempts remain preserved and excluded from this trajectory.
[Continuation](../experiments/dflash2-smoke-20261106/continuation.json) ·
[Attempt history](../experiments/dflash2-smoke-20261106/attempts.md).

Final archive/native-byte/GPU-resume verification and the four-cell smoke now run
under canonical `remote_job.py`, local tmux `dflash2-smoke-measure-20261106-02`.
[Supervisor](../runs/dflash2-smoke-measure-20261106-02/state.json) ·
[Smoke record](../experiments/dflash2-smoke-20261106/report.md).
One operator, one GPU lane. No external message has been sent; parent owns that
after completion, review and final release. No performance conclusion yet.

A genuine Qwen3.5-4B DFlash2 release has been authenticated and converted;
selector/convolution metadata and tensor coverage are present. No compatible9B
release has yet been found. Runtime qualification and all four final cell statuses
remain pending; prepared artifacts are not measurements. Continue the bounded
smoke with target-only exact greedy references, two warmups and two alternating
repetitions on consumed DEV. Preserve HE4 unmodified. Do not start any other GPU
job while this measurement owner holds the lane.

The exact queued shader message remains pending until this follow-up finishes,
parent review completes and the GPU is released. Parent will deliver it once.

## Reference and history

[Candidate protocol](CANDIDATE_BENCHMARKING.md) · [Stages](CANDIDATE_STAGES.md) ·
[History](HISTORY.md) · [Decisions](DECISIONS.md) · [Git and artifacts](GIT_AND_ARTIFACTS.md).

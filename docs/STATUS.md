# Current project status

## Completed: precision campaign and DFlash2 follow-up

Both user-authorized investigations are complete. Source, compact findings,
figures and reusable runners are integrated; paid models/captures/raw evidence
remain outside Git. External DFlash Q4_K_M remains the throughput incumbent.
No new binary-model GPU work is assigned.

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
Windows C and WSL have low free space; do not infer another training allocation
from completion. Preserve paid states, source/dictionary dependencies and captures.
Prior explicitly authorized retirements remain recorded in their campaign receipts.

The exact queued shader benchmark message is ready for parent delivery now that
review/integration and GPU release are complete. Delivery status/identity lives in
[handoff receipt](../runs/component-campaign-20261105-01/pending-thread-message.json).
Do not resend if a successful delivery receipt is recorded.

## References

[Benchmark protocol](CANDIDATE_BENCHMARKING.md) · [Stages](CANDIDATE_STAGES.md) ·
[History](HISTORY.md) · [Operations](AGENT_OPERATIONS.md) · [Git and artifacts](GIT_AND_ARTIFACTS.md).

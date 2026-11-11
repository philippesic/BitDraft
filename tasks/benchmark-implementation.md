# Benchmark implementation and HE4broad continuation

## Goal

Finish the benchmark/data/training pipeline under Philip's revised runtime requirements,
then generate the selected2K supplement and leave original HE4broad continuing on the hybrid.

## Authorized

RTX5080 implementation/testing, scoped FP16 qualification,2K capture, optimizer-preserving
hybrid continuation and hourly health/repair. Preserve other jobs. Reclaim obsolete large
artifacts after ownership/dependency/evidence checks. No hosted CI, new remote, full6K
capture or heldout24 consumption. New workload sizes are being discussed; do not restart
the rejected oversized protocol or its disabled schedule.

## Done when

Revised benchmarks meet measured wall budgets and preserve target semantics;2K captures
are validated and joined to15647 retained records; original220095 continues with Adam/RNG;
monitoring, docs/findings and full checks are complete. No remote: commits remain local.

## State

Owner root. Artifact root `/home/philip/binary-eagle-artifacts`.
Code, bank-v8/supplement-v2 and scoped FP16/migration/recovery qualification are implemented.
Q4KM completed full600×2, harness90×2 and live12×3 for all3 arms. All matched target
behavior; live completion8/12 per arm, with an HE4 slowdown despite request gains.
The oversized suite full-qualification-01 was intentionally stopped during NVFP4 after
Philip rejected its cost. Complete Q4 evidence stays valid; partial NV timing cannot rank.
The obsolete hourly schedule is deleted. No production capture or hybrid training has begun.
Original run/data physically lives in datasets/retained-he4broad-original under the artifact
root; old path remains a symlink. Original220095 and selected controls are preserved.
Runtime breakdown: qualification/benchmark-runtime-breakdown-q4km.json. Stopping receipt:
benchmarks/full-qualification-01/operator-stop.json. Schedule: training/hourly-schedule.json.

## Next

1. Agree lean workloads from the requested runtime breakdown. Training acceptance check:
   at most a couple minutes every30–60 training minutes. Full comparison must be much
   shorter than the rejected2h38/target. Keep breadth and real tool cases; remove default
   excessive generations/repeats. Measure complete pause/export/eval/resume cost.
2. Update definitions, guards, docs and qualification to the accepted budgets. Reuse valid
   preserved evidence explicitly. Do not silently resume old600×2 or sampled120×3 runs.
3. Then capture the frozen2K subset in qualified FP16, validate the real hybrid, continue
   original220095, verify healthy progress/recovery and create fresh hourly health upkeep.

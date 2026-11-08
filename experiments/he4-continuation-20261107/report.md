# HE4 overnight continuation, November 7–8, 2026

This continuation does not demonstrate useful native headroom on this consumed-DEV panel.

The unchanged HE4 trajectory used **21600.017 synchronized update seconds**, advancing from **4096 to 173193 updates** (169097 additional updates). Acceptance changed by **-4.76 percentage points**; complete-request throughput relative to HE4-4096 was **0.9411×**, with paired timing interval **[0.9338, 0.9470]**. Relative to external DFlash Q4_K_M it was **0.6380×**. These are preliminary consumed-DEV measurements.

| Arm | Accepted / proposed | Acceptance | Request tokens/s |
| --- | ---: | ---: | ---: |
| Target-only | 0 / 0 | — | 184.78 |
| External DFlash Q4_K_M | 1705 / 2610 | 65.33% | 251.58 |
| HE4 parent, 4,096 updates | 935 / 4135 | 22.61% | 170.56 |
| HE4 continued, 173,193 updates | 790 / 4425 | 17.85% | 160.51 |

The final panel uses the existing component native runner, runtime04, Qwen3.5-4B Q4_K_M target, F16 target and draft K/V, context 4096, greedy depth 2, 256-token output reserve, two excluded warmups per server cell, and five alternating measured repetitions over code/reasoning/prose DEV prompts. All native requests match target-only outputs. Acceptance is total accepted/proposed; request speed is total output tokens / complete request wall seconds, including prompt processing. Prefix, decode and resource counters are independently reconciled in [audit.json](audit.json), with the original raw receipts retained.

The frozen recipe remains W1 across all 18 FFN matrices, A4 gate/up and A8 down inputs, Q4_0 attention/fusion, one tied private Q4_K vocabulary matrix, F16 draft K/V and F32 compensation/norm storage. The parent's 24 TRAIN sequences, 1,426 windows, 1,762 unique label positions, saved order/cursor/RNG, Adam moments/groups and constant learning rates were preserved. Repeated exposure is not new data.

## Periodic readouts

The intermediate native panels used 64-token reserves and one repetition, with no warmups. They remain smoke diagnostics and are separate from the final 256-token panel above.

| Additional update seconds | Parent acceptance | Continued acceptance | Parent request tokens/s | Continued request tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 1800 | 21.07% | 19.85% | 121.11 | 117.98 |
| 7200 | 21.07% | 21.07% | 120.21 | 119.93 |
| 14400 | 21.07% | 17.88% | 121.34 | 117.70 |

These screens show no convincing native gain; the 7,200-second checkpoint matches parent acceptance on this shorter workload. No checkpoint was promoted. The fixed-budget final checkpoint determines the main before/after comparison.

| Additional update seconds | Total updates | DEV cross entropy | DEV top-1 agreement |
| ---: | ---: | ---: | ---: |
| 0.000 | 4,096 | 7.1205 | 9.63% |
| 300.021 | 6,442 | 7.5279 | 11.85% |
| 1800.101 | 18,179 | 7.5332 | 11.85% |
| 3600.051 | 32,270 | 7.7065 | 15.56% |
| 7200.119 | 60,455 | 7.6649 | 13.33% |
| 10800.036 | 88,645 | 7.9795 | 13.33% |
| 14400.102 | 116,819 | 8.5157 | 10.37% |
| 18000.029 | 145,006 | 8.4448 | 10.37% |
| 21600.017 | 173,193 | 8.5477 | 9.63% |

DEV loss/agreement are the inherited QAT surrogate, distinct from native acceptance. The continuation exposed 2,536,455 supervised positions over the same 1,762 unique capture-label positions: 118.581 passes over the 1,426 TRAIN windows, or 1,439.532 supervised-position passes because adjacent 15-label windows overlap. These are repeated exposures, not new data.

## Recovery and resources

Exact parent archive/raw/native hashes were verified before training. Native byte recovery and an actual scratch GPU update from 4096 to 4097 passed before the update budget. Qualification did not mutate parent Adam counters. The final full checkpoint was decoded, re-exported to identical native bytes, and passed an actual next GPU update. Qualification time is separate from the 21,600 update-second budget.

Peak Torch allocated/reserved memory: 9.869 / 10.037 GiB; peak host RSS: 13.439 GiB. Trainer setup/qualification: 225.720s; checkpoints: 180.848s; trainer process wall: 23126.147s. One-update budget overshoot: 0.017170s. Readout/export costs are recorded separately at each point. All intermediate checkpoints and native exports are retained in the new run directory.

Final checkpoint: `/home/philip/binary-eagle-decoding/runs/he4-continuation-20261107-01/checkpoints/HE4-plus21600-173193.pt.bsz`. Final native export: `/home/philip/binary-eagle-decoding/runs/he4-continuation-20261107-01/candidate/model-21600.gguf`. Full hashes and ancestry are in [machine-readable results](results.json); run accounting, source closure, owner identities, commands, per-update logs and raw native receipts are under `/home/philip/binary-eagle-decoding/runs/he4-continuation-20261107-01`. The original parent and frontier registry were preserved.

## Independent audit and retention

The independent audit recomputed all 169,097 update rows, checked finite losses/gradient norms and cumulative time, confirmed the main first update exactly matches the qualification probe, verified all eight checkpoint and nine native-export hashes, and reconciled all 87 measured request rows across the four native panels. All 127 requests including excluded warmups match the target. The contender pipeline passed and its supervisor exited successfully.

| Arm | Verification rounds | Emitted tokens/round | Server decode tokens/s | Prefix survivors at positions 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| HE4 parent | 2,070 | 1.461 | 183.99 | 695 / 240 |
| HE4 continued | 2,215 | 1.366 | 172.13 | 570 / 220 |
| External Q4 | 1,305 | 2.318 | 282.47 | 995 / 710 |

Prefix survivors use the existing per-request rounded native rates; server decode excludes prompt processing, while complete-request speed includes it. The increased number of verification rounds is consistent with the lower measured acceptance.

All eight new full states, all nine native exports, the original parent, source/capture dependencies, raw requests and reusable caches remain retained. Authenticated owned process groups are absent, no GPU compute process remains, and the contender lock was reacquired and released. [Retention and exact keep set](../../runs/he4-continuation-20261107-01/milestone-retention.json) · [GPU release receipt](../../runs/he4-continuation-20261107-01/gpu-release.json). Large artifacts remain local; Git synchronization of source/results does not back up those weights. No other artifact was retired.

## Interpretation limits

The measured result is negative for this fixed-data continuation; it does not establish a general HE4 convergence or capacity limit. Three previously consumed DEV prompts, one training trajectory, and five alternating process repetitions; timing repetitions are not independent prompts or training seeds. No new heldout, context matrix, or other target cell was opened. Intermediate 64-token panels are one-repetition smoke diagnostics and cannot be pooled with the final 256-token comparison. Inherited QAT/native arithmetic parity limitations remain; native complete requests determine throughput. External DFlash Q4_K_M was rerun in the matched final panel and remains the deployment incumbent. No mixed-weight experiment or deployment promotion was performed.

[Native table](native.csv) · [DEV trajectory](trajectory.csv) · [Protocol](../../docs/CANDIDATE_BENCHMARKING.md) · [Run plan](manifest.json).

[Comparison figure](comparison.png) · [Portable SVG](comparison.svg) · [HTML view](comparison.html) · [Independent audit](audit.json).

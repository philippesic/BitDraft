# Fresh HE4broad on the combined 2080 Ti captures

Status: **COMPLETE**. 220,095 recoverable updates; 28800.105 synchronized update seconds, 49625.032 training wall seconds (including 20817.785 data I/O seconds). Philip's resumed allocation is **28,800 total synchronized update seconds (8 hours)**, including the existing 2873.748 seconds. Rejected update time also counts. There is no training-wall cap; data I/O, setup, checkpoints, export and native eval do not consume the update allocation and remain separately reported.

Fresh base-drafter HE4 quantization, fresh Adam, combined 15,647 sequences and 1,568,435 windows. Global seeded window shuffle; exact own-state resume. All hourly native panels use depth 2, greedy, F16 KV, context 4096, two warmups and five repetitions per cell with 256-token output reserve. The extra 60-second diagnostic uses one repetition and 64 tokens. Network model transfer is excluded from native request timings.

Historical numeric points are training-wall seconds; new `update-*` points are cumulative synchronized update seconds. The original recipe/config is retained verbatim for exact state compatibility; the explicit continuation policy supersedes its old clock/budget.

| Readout (clock in label) | Actual update seconds | Updates | DEV CE | DEV agreement |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0 | 7.825455 | 0.7407% |
| 60 | 17.736 | 56 | 7.029570 | 6.6667% |
| 3600 | 511.046 | 3662 | 5.886574 | 14.0741% |
| 7200 | 1009.754 | 7281 | 5.983110 | 12.5926% |
| 10800 | 1508.975 | 10934 | 5.819820 | 17.7778% |
| 14400 | 2012.362 | 14556 | 5.524400 | 19.2593% |
| 18000 | 2521.241 | 18187 | 5.553204 | 20.0000% |
| paused-20671 | 2873.748 | 20671 | 5.429714 | 19.2593% |
| update-3600 | 3600.016 | 26173 | 5.317311 | 22.2222% |
| update-7200 | 7200.065 | 53763 | 5.133850 | 24.4444% |
| update-10800 | 10800.002 | 81455 | 4.904629 | 26.6667% |
| update-14400 | 14400.125 | 109165 | 5.025072 | 22.9630% |
| update-18000 | 18000.036 | 136860 | 4.680093 | 31.1111% |
| update-21600 | 21600.024 | 164674 | 4.603074 | 27.4074% |
| update-25200 | 25200.108 | 192372 | 4.450846 | 30.3704% |
| update-28800 | 28800.105 | 220095 | 4.457087 | 28.8889% |

| Panel | Arm | Draft acceptance | Request tokens/s | Correct |
| --- | --- | ---: | ---: | --- |
| formal-0 | target | — | 184.002 | True |
| formal-0 | q4 | 65.3257% | 247.398 | True |
| formal-0 | HE4broad | 3.1858% | 127.546 | True |
| smoke-60 | target | — | 133.435 | True |
| smoke-60 | q4 | 59.6491% | 144.728 | True |
| smoke-60 | HE4broad | 10.0649% | 106.985 | True |
| formal-3600 | target | — | 184.818 | True |
| formal-3600 | q4 | 65.3257% | 248.243 | True |
| formal-3600 | HE4broad | 27.0513% | 179.438 | True |
| formal-7200 | target | — | 184.745 | True |
| formal-7200 | q4 | 65.3257% | 249.562 | True |
| formal-7200 | HE4broad | 28.8714% | 183.027 | True |
| formal-10800 | target | — | 184.696 | True |
| formal-10800 | q4 | 65.3257% | 249.637 | True |
| formal-10800 | HE4broad | 31.2585% | 187.443 | True |
| formal-14400 | target | — | 185.401 | True |
| formal-14400 | q4 | 65.3257% | 249.030 | True |
| formal-14400 | HE4broad | 31.7007% | 187.439 | True |
| formal-18000 | target | — | 185.175 | True |
| formal-18000 | q4 | 65.3257% | 249.238 | True |
| formal-18000 | HE4broad | 32.5549% | 189.175 | True |
| formal-update-3600 | target | — | 184.902 | True |
| formal-update-3600 | q4 | 65.3257% | 249.874 | True |
| formal-update-3600 | HE4broad | 36.5994% | 197.484 | True |
| formal-update-7200 | target | — | 184.421 | True |
| formal-update-7200 | q4 | 65.3257% | 248.044 | True |
| formal-update-7200 | HE4broad | 40.7855% | 205.718 | True |
| formal-update-10800 | target | — | 183.893 | True |
| formal-update-10800 | q4 | 65.3257% | 248.037 | True |
| formal-update-10800 | HE4broad | 44.7950% | 213.585 | True |
| formal-update-14400 | target | — | 184.946 | True |
| formal-update-14400 | q4 | 65.3257% | 249.414 | True |
| formal-update-14400 | HE4broad | 45.0949% | 215.956 | True |
| formal-update-18000 | target | — | 184.004 | True |
| formal-update-18000 | q4 | 65.3257% | 248.696 | True |
| formal-update-18000 | HE4broad | 46.0064% | 216.667 | True |
| formal-update-21600 | target | — | 184.409 | True |
| formal-update-21600 | q4 | 65.3257% | 246.117 | True |
| formal-update-21600 | HE4broad | 46.3141% | 215.472 | True |
| formal-update-25200 | target | — | 184.351 | True |
| formal-update-25200 | q4 | 65.3257% | 248.677 | True |
| formal-update-25200 | HE4broad | 48.8487% | 222.065 | True |
| formal-update-28800 | target | — | 174.403 | True |
| formal-update-28800 | q4 | 65.3257% | 228.353 | True |
| formal-update-28800 | HE4broad | 46.9355% | 199.505 | True |

Consumed DEV3, one training trajectory. Timing repetitions are not fresh prompts/seeds. No heldout opened. External Q4 remains deployment incumbent; no promotion.

[Results and pins](results.json) · [Operations](operations.md) · [Local-data migration](localization.md).

[Final validation, interpretation and retention](final-report.md).

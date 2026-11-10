# Fresh HE4broad on the combined 2080 Ti captures

Status: **RUNNING**. 10,934 recoverable updates; 1508.975 synchronized update seconds, 10800.552 training wall seconds (including 9290.798 data I/O seconds). Cap: 43,200 training wall seconds and 43,200 actual update seconds. Setup, checkpoints, export and native eval are separate.

Fresh base-drafter HE4 quantization, fresh Adam, combined 15,647 sequences and 1,568,435 windows. Global seeded window shuffle; exact own-state resume. All hourly native panels use depth 2, greedy, F16 KV, context 4096, two warmups and five repetitions per cell with 256-token output reserve. The extra 60-second diagnostic uses one repetition and 64 tokens. Network model transfer is excluded from native request timings.

| Training wall point | Actual update seconds | Updates | DEV CE | DEV agreement |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0 | 7.825455 | 0.7407% |
| 60 | 17.736 | 56 | 7.029570 | 6.6667% |
| 3600 | 511.046 | 3662 | 5.886574 | 14.0741% |
| 7200 | 1009.754 | 7281 | 5.983110 | 12.5926% |
| 10800 | 1508.975 | 10934 | 5.819820 | 17.7778% |

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

Consumed DEV3, one training trajectory. Timing repetitions are not fresh prompts/seeds. No heldout opened. External Q4 remains deployment incumbent; no promotion.

[Results and pins](results.json) · [Operations](operations.md).

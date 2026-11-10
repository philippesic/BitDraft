# Fresh HE4broad on the combined 2080 Ti captures

Status: **RUNNING**. 56 recoverable updates; 17.736 synchronized update seconds, 60.893 training wall seconds (including 43.140 data I/O seconds). Cap: 43,200 training wall seconds and 43,200 actual update seconds. Setup, checkpoints, export and native eval are separate.

Fresh base-drafter HE4 quantization, fresh Adam, combined 15,647 sequences and 1,568,435 windows. Global seeded window shuffle; exact own-state resume. All hourly native panels use depth 2, greedy, F16 KV, context 4096, two warmups and five repetitions per cell with 256-token output reserve. The extra 60-second diagnostic uses one repetition and 64 tokens. Network model transfer is excluded from native request timings.

| Training wall point | Actual update seconds | Updates | DEV CE | DEV agreement |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0 | 7.825455 | 0.7407% |
| 60 | 17.736 | 56 | 7.029570 | 6.6667% |

| Panel | Arm | Draft acceptance | Request tokens/s | Correct |
| --- | --- | ---: | ---: | --- |
| formal-0 | target | — | 184.002 | True |
| formal-0 | q4 | 65.3257% | 247.398 | True |
| formal-0 | HE4broad | 3.1858% | 127.546 | True |
| smoke-60 | target | — | 133.435 | True |
| smoke-60 | q4 | 59.6491% | 144.728 | True |
| smoke-60 | HE4broad | 10.0649% | 106.985 | True |

Consumed DEV3, one training trajectory. Timing repetitions are not fresh prompts/seeds. No heldout opened. External Q4 remains deployment incumbent; no promotion.

[Results and pins](results.json) · [Operations](operations.md).

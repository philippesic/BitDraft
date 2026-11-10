# HE4broad source-initialized QAT, November 8, 2026

November 10 retention update: zero and final full states remain; the seven
superseded hourly optimizer states were retired. All measurements and native
readouts remain. [Retention record](../storage-cleanup-20261110/report.md).

Status: **COMPLETE**. Completed **226,853 updates** using **28800.038 / 28,800 actual update seconds**. The last recoverable checkpoint covers 226,853 updates. Setup, evaluation, serialization and native panels are outside this budget.

This new recipe starts from authenticated source BF16 DFlash tensors and target Qwen3.5-4B Q4_K_M tensors. No B0, HE4-4096 or overnight continuation parameters/Adam state are loaded. Exact source latent tensors, mean-absolute row scales, zero midpoints/log gains and HE4 initial clipping bounds are asserted before training. Existing saved JSON tensor digests establish distinct trainable scales from both B0 and HE4-4096. The private tied Q4_K vocabulary is installed from its authenticated target-derived matrix.

The architecture matches HE4: all 18 FFN projections W1; gate/up A4, down A8; Q4_0 attention/fusion; one private tied Q4_K head/embedding; F16 draft KV; F32 scales, midpoints, gains, clipping and norms. TRAIN asserts 40 sequences, 3,234 windows and 3,794 unique label positions, with the exact Csmallbroad/H0broad manifest. consumed_DEV is evaluation-only. Fresh Adam uses the unchanged optimizer defaults: signs 0.001; scale/midpoint/gain/clip/norm 0.00001; constant schedule, weight decay zero. Every group takes a real Adam step each window. Window shuffle seed 8101, initial cursor zero. Loss is the existing 15-position decay-weighted hard-target CE.

| Requested update seconds | Actual update seconds | Updates | DEV CE | DEV top-1 agreement |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0 | 7.825455 | 0.7407% |
| 3600 | 3600.054 | 28271 | 7.723593 | 13.3333% |
| 7200 | 7200.112 | 56664 | 7.636782 | 11.8519% |
| 10800 | 10800.005 | 85029 | 8.360915 | 16.2963% |
| 14400 | 14400.098 | 113404 | 8.220396 | 17.7778% |
| 18000 | 18000.084 | 141799 | 8.426917 | 10.3704% |
| 21600 | 21600.112 | 170170 | 8.499314 | 13.3333% |
| 25200 | 25200.053 | 198528 | 8.479428 | 14.0741% |
| 28800 | 28800.038 | 226853 | 8.899076 | 13.3333% |

Native panels use runtime04, consumed DEV3, depth 2, greedy sampling, context 4096 and F16 target/draft KV. Formal panels use two excluded warmups per server cell, five alternating repetitions, and 256-token output reserve. Smoke panels use one repetition, no warmups and 64 tokens. Training exits before each native panel.

### smoke-0

| Arm | Acceptance | Complete-request tokens/s | Correct |
| --- | ---: | ---: | --- |
| target | — | 136.926 | True |
| q4 | 59.6491% | 147.173 | True |
| HE4broad | 2.2599% | 93.658 | True |

### formal-0

| Arm | Acceptance | Complete-request tokens/s | Correct |
| --- | ---: | ---: | --- |
| target | — | 183.516 | True |
| q4 | 65.3257% | 248.523 | True |
| HE4broad | 3.1858% | 127.358 | True |

### smoke-14400

| Arm | Acceptance | Complete-request tokens/s | Correct |
| --- | ---: | ---: | --- |
| target | — | 138.108 | True |
| q4 | 59.6491% | 143.826 | True |
| HE4broad | 22.0077% | 117.108 | True |

### smoke-28800

| Arm | Acceptance | Complete-request tokens/s | Correct |
| --- | ---: | ---: | --- |
| target | — | 142.030 | True |
| q4 | 59.6491% | 151.964 | True |
| HE4broad | 20.9125% | 118.532 | True |

### formal-28800

| Arm | Acceptance | Complete-request tokens/s | Correct |
| --- | ---: | ---: | --- |
| target | — | 185.444 | True |
| q4 | 65.3257% | 249.662 | True |
| HE4broad | 24.5658% | 174.113 | True |

Full checkpoints (including zero and every hourly state) are retained under `/mnt/c/Users/philip/binary-eagle-decoding-artifacts/he4broad-20261108-01`; exports, immutable source closure, commands, process identity and update log are under `/home/philip/binary-eagle-decoding/runs/he4broad-20261108-01`. Storage is physically reserved before CUDA work; unused reservation files are released at shutdown. SIGTERM is honored between complete updates. An update exception preserves the last good checkpoint and is excluded from completed-update accounting.

Final checkpoint: `/mnt/c/Users/philip/binary-eagle-decoding-artifacts/he4broad-20261108-01/HE4broad-28800-226853.pt.bsz`. Final export: `/home/philip/binary-eagle-decoding/runs/he4broad-20261108-01/candidate/model-28800.gguf`. Full hashes and native-byte recovery proof are in [results](results.json).

Consumed DEV3 only; one source-initialized seed and training trajectory. Timing repetitions are not new prompts or training seeds. No heldout24/fresh24 opened. QAT CE/agreement are diagnostics; native throughput uses complete request wall time. External DFlash Q4_K_M remains the deployment incumbent; no promotion claimed.

[Results and hashes](results.json) · [Completed-update audit](audit.json) · [Manifest](manifest.json).

# Findings

Old reports are attributed historical measurements, not fresh reproductions.
Question/date: what survives foundation, 2026-11-09. Host: historical RTX 5080
unless the source says otherwise. Raw data remain external; remote availability
is unverified. Exact selected object identities are in [evidence](policy/evidence.toml).

## F001

Four names are distinct; incumbent is external DFlash Q4KM. HE4 is W1A4, A8 down, Q4K head.

Limits: No cross-era speed ratio or retrospective DSpark baseline relabeling..

Evidence: [experiments/w1a8-standing-20261102/baseline.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/w1a8-standing-20261102/baseline.json) [docs/CANDIDATE_BENCHMARKING.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/docs/CANDIDATE_BENCHMARKING.md) [experiments/dspark-dflash-catchup-20261104/README.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dspark-dflash-catchup-20261104/README.md) 

## F002

Retained HE4 and Csmallbroad are tradeoff frontiers, not deployment winners. Component heldout request rates: HE4 167.06, Csmallbroad 168.95, Baseline 236.67 tok/s. HE4 0.706 [0.690,0.724] versus Baseline; Csmallbroad 0.714 [0.696,0.734]. HE4 GPU forward improves 14.88% versus H0; private vocabulary adds 341.02 MiB.

Limits: 24 then-fresh clusters, 5 process repetitions; now consumed evaluation, not fresh training/selection. Improvement versus H0 is not superiority over Baseline..

Evidence: [experiments/component-campaign-20261105/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/component-campaign-20261105/report.md) [experiments/component-campaign-20261105/results.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/component-campaign-20261105/results.json) 

## F003

Unchanged HE4 continuation worsened: acceptance 22.61→17.85%; complete-request throughput 170.56→160.51 tok/s; paired 0.9411 [0.9338,0.9470]. Added 169,097 updates on 1,762 unique labels produced 1,439.532 position exposures.

Limits: One fixed-data trajectory; DEV3, 5 process repetitions; no general capacity/convergence conclusion. Smoke cap 64 differs from formal 256..

Evidence: [experiments/he4-continuation-20261107/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/report.md) [experiments/he4-continuation-20261107/audit.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/audit.json) [experiments/he4-continuation-20261107/trajectory.csv](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/trajectory.csv) [experiments/he4-continuation-20261107/native.csv](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/native.csv) 

## F004

Fresh-source HE4broad ended at 24.5658%, 174.113 tok/s versus target 185.444 and Baseline 65.3257%, 249.662. 226,853 updates; 40 sequences/3,234 windows/3,794 unique labels; 896.889 exposures. DEV top-1 peaked 17.7778% at hour 4, final 13.3333%; final CE 8.899076.

Limits: DEV3, one seed/trajectory; formal serving at 0/8 h only. Hour-4 smoke cannot establish best formal-serving checkpoint. No HE4 win..

Evidence: [experiments/he4broad-20261108/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/report.md) [experiments/he4broad-20261108/audit.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/audit.json) [experiments/he4broad-20261108/results.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/results.json) [experiments/he4broad-20261108/manifest.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/manifest.json) 

## F005

Null-topic recovery is unresolved. Supplied partition: 9,856=12+23+5,505+4,316; corrected admission would be 9,821 under identical exclusions. ~5,497 recaptures differ by eight. Old one-null fixture cannot catch erroneous merging of two nulls.

Limits: Remote bug/counts are Philip-supplied, not audited on 2080 Ti. Preserve a compact attributed arithmetic/fixture observation; no claim a new capture has been admitted..

Evidence: [scripts/capture_nine_model_train_data.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/scripts/capture_nine_model_train_data.py) [tests/test_dspark_full_pool_capture.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/tests/test_dspark_full_pool_capture.py) 

## F006

Csmallbroad accepted continuation: 9,998 updates/1,800.073 update seconds; request changes −2.19% on Q4 target and +1.54% on NVFP4. Proxy CE worsened while top-1 rose. Two failed attempts consumed another 2,100.023 update seconds; aliasing scalar Adam state invalidated one attempt.

Limits: Two process repetitions and DEV3 smoke; no consistent gain/ranking. The regression source was read, not executed here..

Evidence: [experiments/dflash2-smoke-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/report.md) [experiments/dflash2-smoke-20261106/parent-review.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/parent-review.md) [experiments/dflash2-smoke-20261106/attempts.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/attempts.md) [tests/test_gpu_resume_optimizer_state.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/tests/test_gpu_resume_optimizer_state.py) 

## F007

Original DFlash2 Q4 and F16 native paths had nonfinite logits. Accepted fallback replaced exactly five down matrices with released BF16→F32 values; 76 other tensors unchanged. No matched 9B release established by bounded search.

Limits: Mixed Q4/F32 result, not fully Q4 DFlash2; bounded historical search is not a present universal availability claim..

Evidence: [experiments/dflash2-smoke-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/report.md) [experiments/dflash2-smoke-20261106/results.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/results.json) 

## F008

Execution improvement can coexist with model loss: learned-small GPU forward −17.2% [15.5,18.8], full-request +2.51% [1.37,3.50], yet 176.21 versus Baseline 244.39 tok/s. Some fusions changed packed bytes and were rejected.

Limits: GPU replay, host trace and full HTTP request clocks differ; natural12 consumed DEV; learned-small is not relabeled HE4. Old DSpark head precision differs..

Evidence: [experiments/execution-campaign-20261105/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/execution-campaign-20261105/report.md) [experiments/execution-campaign-20261105/ledger.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/execution-campaign-20261105/ledger.json) [experiments/execution-campaign-20261105/historical-latency-context.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/execution-campaign-20261105/historical-latency-context.md) 

## F009

Historical DSpark binary screen: no alternative beat retained W1A8 pipeline; A1 XOR request 137.95 versus A8 197.45 tok/s. A8-trained weights with A1 activation override were not trained W1A1. Tested binary-MMA PTX lowered to integer MMA.

Limits: Qwen3 F16/DSpark D5, two repetitions; no modern DFlash inference, no generalized hardware claim..

Evidence: [experiments/binary-methods-20261104/README.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/binary-methods-20261104/README.md) [experiments/binary-methods-20261104/analysis.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/binary-methods-20261104/analysis.json) 

## F010

November 6 deliberately retired 34 optimizer archives; five protected full states survived. HE4/Csmallbroad/B0/foundation and accepted Csmallbroad continuation remain separate identities; preserve compression bases/dictionaries/source/capture closure.

Limits: Retirement receipt supersedes older preservation statements for exact retired paths. Current metadata is not fresh multi-GiB hashing or an independent backup. Historical 66.20 GiB free space is stale..

Evidence: [experiments/storage-cleanup-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/storage-cleanup-20261106/report.md) [experiments/storage-cleanup-20261106/summary.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/storage-cleanup-20261106/summary.json) [docs/DECISIONS.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/docs/DECISIONS.md) 

The provisional 2080 Ti observation is 10,150 sequences, 970,121 windows and
1,111,811 labels (Philip supplied, not audited). The 5,505 versus approximately
5,497 recovery difference remains open; no recovered prompt is admitted here.

## F011

A fresh-context readiness exercise at the worker-control checkpoint correctly
refused an actual HE4 launch, reported 34 CPU tests and all 125 specimen pairs,
and distinguished fake-worker/storage checks from numerical and native evidence.
It also found stale capability prose that the link checker did not detect.

Limits: one unaided readiness/no-change task, shared-host access, no complete dirty
tree digest, no timing comparison or causal productivity conclusion. This does
not qualify later source increments or actual training.

Evidence: [readiness receipt](docs/evidence/agent-readiness.json).

## F012

The CPU source path now composes admitted synthetic captures, the numerical worker,
periodic full states, exact own-state resume and checked GGUF export. Real process-kill
and file-size-limit failures preserve the prior committed state. A source audit caught
and fixed recovery-needed state publication after a clock error.

Limits: reduced CPU constructions and selected historical arithmetic characterization;
no full-size retained-state parity, native load, scientific adequacy or hardware result.
The controller fixture's zero vocabulary tests construction and accounting, while a
separate six-layer fixture verifies a nonzero parameter update. Large-index intake is
still pending. Full commit evidence includes the required numerical cases and 125 bad/
control pairs; a green gate grants no execution authority or serving qualification.

Evidence: [execution contract](docs/execution.md), [arithmetic characterization](docs/evidence/arithmetic-reference.json),
[selected source inventory](docs/evidence/source-inventory.json), [decision 007](docs/decisions/007-checks-and-review.md).


## F013

A fresh agent repaired a scoped immutable-window contract using normal docs and checks,
without hints or checker/policy changes. It reproduced failures, corrected two typing
errors in its test expressions, and passed the full gate. Parent probes independently
confirmed the requested type/immutability/position behavior. An extra whitespace-only
probe over-specified the supplied task; the discrepancy was adjudicated before a separate
parent correction made identity handling consistent with neighboring records.

Limits: one shared-host routine task, instruction-based scope isolation, warm caches,
no matched comparison or causal efficiency claim. All unsuccessful check attempts remain
in the receipt. This supplements F011, not a universal qualification of agent behavior.

Evidence: [routine exercise receipt](docs/evidence/agent-window-exercise.json).

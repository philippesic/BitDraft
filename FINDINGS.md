# Findings

Closed questions, each with its answer, scope and evidence. Failures and unfavorable
results stay. Correct an entry in place when it turns out wrong. F001 to F010 report
measurements from the predecessor repository on the RTX 5080 unless stated otherwise;
they were not rerun here. Raw data stays outside Git; pinned object identities are in
the [evidence registry](policy/evidence.toml).

## F001: Which drafter is the incumbent?

The external DFlash Q4KM drafter is the Baseline. HE4 (W1A4, A8 down projection, Q4K
head), historical DSpark W1A8 and the historical DSpark Q4KM baseline are separate
identities.

Scope: naming only. There is no speed ratio across eras, and DSpark results are not
relabeled as Baseline results.

Evidence: [experiments/w1a8-standing-20261102/baseline.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/w1a8-standing-20261102/baseline.json) [docs/CANDIDATE_BENCHMARKING.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/docs/CANDIDATE_BENCHMARKING.md) [experiments/dspark-dflash-catchup-20261104/README.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dspark-dflash-catchup-20261104/README.md)

## F002: Do the retained HE4 and Csmallbroad states beat Baseline?

No. On heldout requests HE4 ran 167.06 tok/s, Csmallbroad 168.95 and Baseline 236.67.
Relative to Baseline, HE4 scored 0.706 [0.690, 0.724] and Csmallbroad 0.714
[0.696, 0.734]. HE4's GPU forward pass improved 14.88% against H0, and its private
vocabulary adds 341.02 MiB. Both are tradeoff points, not deployment winners.

Scope: 24 clusters that were fresh then and are consumed now, 5 process repetitions.
Beating H0 is not beating Baseline.

Evidence: [experiments/component-campaign-20261105/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/component-campaign-20261105/report.md) [experiments/component-campaign-20261105/results.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/component-campaign-20261105/results.json)

## F003: Did continuing HE4 training on the same data help?

No, it made the model worse. Acceptance fell from 22.61% to 17.85%, and complete-request
throughput fell from 170.56 to 160.51 tok/s (paired ratio 0.9411 [0.9338, 0.9470]).
The run added 169,097 updates on only 1,762 unique labels, a mean exposure of 1,439.532
per position. Development loss rose from 7.12 to 8.55 and had already reached 7.53 after
5 minutes, while the run kept reporting healthy because monitoring checked liveness only.

Scope: one fixed-data trajectory, DEV3, 5 process repetitions. The smoke cap of 64 differs
from the formal 256. No general conclusion about capacity or convergence.

Evidence: [experiments/he4-continuation-20261107/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/report.md) [experiments/he4-continuation-20261107/audit.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/audit.json) [experiments/he4-continuation-20261107/trajectory.csv](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/trajectory.csv) [experiments/he4-continuation-20261107/native.csv](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4-continuation-20261107/native.csv)

## F004: Did fresh-source HE4broad training produce a better checkpoint?

It ended at 24.5658% acceptance and 174.113 tok/s, against 185.444 for target-only and
65.3257% and 249.662 tok/s for Baseline. Training ran 226,853 updates over 40 sequences,
3,234 windows and 3,794 unique labels, a mean exposure of 896.889. DEV top-1 peaked at
17.7778% at hour 4 and ended at 13.3333%; final cross-entropy was 8.899076. Formal serving
was scored only at hours 0 and 8, so the best checkpoint was never selected, although
hourly full states were kept.

Scope: DEV3, one seed and trajectory. The hour-4 DEV peak does not establish a formal
serving peak. No HE4 win over Baseline.

Evidence: [experiments/he4broad-20261108/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/report.md) [experiments/he4broad-20261108/audit.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/audit.json) [experiments/he4broad-20261108/results.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/results.json) [experiments/he4broad-20261108/manifest.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/he4broad-20261108/manifest.json)

## F005: How many prompts did the null-topic deduplication bug drop?

A deduplication bug on empty `source_topic` merged distinct prompts. The supplied
partition is 10,000 = 96 cal-fit + 48 cal-val + 9,856 candidates, and 9,856 = 12 DEV +
23 real collisions + 5,505 wrong rejections + 4,316 admitted. With only the genuine
exclusions, 9,821 would have been admitted. The recapture of about 5,497 prompts differs
from 5,505 by eight, and that difference is unexplained. The old fixture had one null
topic and could not catch two nulls being merged.

The 2080 Ti recapture reported 10,150 sequences, 970,121 windows and 1,111,811 labels.
These are gross counts supplied by Philip, not audited and not admitted.

Scope: bug description and counts supplied by Philip, not audited on the 2080 Ti.
No recaptured prompt is admitted.

Evidence: [scripts/capture_nine_model_train_data.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/scripts/capture_nine_model_train_data.py) [tests/test_dspark_full_pool_capture.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/tests/test_dspark_full_pool_capture.py)

## F006: Did the accepted Csmallbroad continuation help?

Not consistently. It ran 9,998 updates in 1,800.073 update seconds. Request throughput
changed by -2.19% on the Q4 target and +1.54% on NVFP4. Proxy cross-entropy worsened
while top-1 rose. Two failed attempts used another 2,100.023 update seconds; in one,
aliased scalar Adam state invalidated the attempt.

Scope: two process repetitions and DEV3 smoke. No consistent gain or ranking. The
regression source was read, not executed.

Evidence: [experiments/dflash2-smoke-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/report.md) [experiments/dflash2-smoke-20261106/parent-review.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/parent-review.md) [experiments/dflash2-smoke-20261106/attempts.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/attempts.md) [tests/test_gpu_resume_optimizer_state.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/tests/test_gpu_resume_optimizer_state.py)

## F007: Did the original DFlash2 Q4 and F16 native paths work?

No, both produced nonfinite logits. The accepted fallback replaced exactly five down
projection matrices with released BF16 values converted to F32 and left the other 76
tensors unchanged. A bounded search found no matching 9B release.

Scope: a mixed Q4 and F32 result, not a fully Q4 DFlash2. The search was bounded and
historical.

Evidence: [experiments/dflash2-smoke-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/report.md) [experiments/dflash2-smoke-20261106/results.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/results.json)

## F008: Can execution speedups compensate for model quality?

Not so far. The learned-small GPU forward pass improved 17.2% [15.5, 18.8] and the full
request 2.51% [1.37, 3.50], yet it ran 176.21 tok/s against Baseline's 244.39. Some
fusions changed packed bytes and were rejected.

Scope: GPU replay, host trace and full HTTP request clocks differ. natural12 is consumed
DEV. Learned-small is not HE4, and the old DSpark head precision differs.

Evidence: [experiments/execution-campaign-20261105/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/execution-campaign-20261105/report.md) [experiments/execution-campaign-20261105/ledger.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/execution-campaign-20261105/ledger.json) [experiments/execution-campaign-20261105/historical-latency-context.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/execution-campaign-20261105/historical-latency-context.md)

## F009: Did any binary method beat the retained W1A8 pipeline?

No. A1 XOR requests ran 137.95 tok/s against 197.45 for A8. A8-trained weights run with
an A1 activation override are not trained W1A1. The tested binary-MMA PTX lowered to
integer MMA.

Scope: Qwen3 F16 with DSpark D5, two repetitions. No modern DFlash inference and no
general hardware conclusion.

Evidence: [experiments/binary-methods-20261104/README.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/binary-methods-20261104/README.md) [experiments/binary-methods-20261104/analysis.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/binary-methods-20261104/analysis.json)

## F010: Which retained training states survived the November 6 cleanup?

34 optimizer archives were deliberately retired, and five protected full states survived.
HE4, Csmallbroad, B0, foundation and the accepted Csmallbroad continuation remain
separate identities. Their compression bases, dictionaries, source and capture closure
must be preserved.

Scope: the retirement receipt overrides older preservation statements for the exact
retired paths. Current metadata is not a fresh multi-GiB hash or an independent backup.

Evidence: [experiments/storage-cleanup-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/storage-cleanup-20261106/report.md) [experiments/storage-cleanup-20261106/summary.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/storage-cleanup-20261106/summary.json) [docs/DECISIONS.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/docs/DECISIONS.md)

## F011: Does the retained modern index reconcile?

Yes. The selected modern index reconciles 15,647 records, 1,568,435 windows and
1,786,275 unique supervised positions; 87 records have no windows. Prior use is
historical training for 9,813 records, unknown for 5,658 and unobserved for 176.
Intake on the real metadata exposed and fixed two port errors: first-anchor and
end-of-generation handling, and the original sampler order.

Scope: metadata and source-chain reconciliation only. No raw-feature hashing, teacher
replay or data adequacy. The F005 recovery difference is still open.

Evidence: [complete intake](evidence/modern-index-complete-intake.json),
[pinned intake](evidence/modern-index-qualified-intake.json),
[earlier refusals](evidence/modern-index-intake.json),
[snapshot refusal](evidence/modern-index-snapshot-intake.json),
[transport](evidence/metadata-snapshot-transport.json),
[initializer header](evidence/he4-initializer-header.json).

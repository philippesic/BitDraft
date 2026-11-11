# Research context and historical sources

The goal is faster complete requests without changing the target model's verification
semantics. This page explains why the current candidate exists and where to find the
supporting work. [FINDINGS](../FINDINGS.md) indexes result pages containing conclusions, scope and evidence.
Current data, model and execution behavior live in [data](data.md), [model](model.md)
and [execution](execution.md). Historical plans grant no present work allocation.

## Why HE4 is the current candidate

The project tested activation precision, training recovery, native execution cost and
memory tradeoffs. A four-bit gate/up, eight-bit down family was useful among the tested
recipes, but those earlier learned-small models are separate from HE4. HE4 then added
a private tied Q4_K vocabulary path that reduced draft latency at additional memory
cost. Neither step established a win over external DFlash Q4_K_M.

Philip chose HE4 as the development priority because lower draft cost leaves room for
an end-to-end win if prediction quality improves. Csmallbroad remains a compact
fallback when vocabulary/cache memory matters. That is a research direction, not a
deployment promotion. See [the recorded decision](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/docs/DECISIONS.md),
[F002](../FINDINGS.md#f002-do-the-retained-he4-and-csmallbroad-states-beat-baseline)
and [the activation-family result](../FINDINGS.md#f012-what-did-the-activation-family-comparison-establish).

Longer training did not automatically help: the fixed-data continuation got worse,
and the earlier broader-data run did not score serving performance at its best DEV
readout. Those are bounded negative results, not proof that HE4 cannot improve. See
[F003](../FINDINGS.md#f003-did-continuing-he4-training-on-the-same-data-help) and
[F004](../FINDINGS.md#f004-did-fresh-source-he4broad-training-produce-a-better-checkpoint).

## Which past work to consult

| Question | Read | How to use it |
| --- | --- | --- |
| Why this model and a compact fallback? | [Priority and deferred precision decision](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/docs/DECISIONS.md) | Preserve the distinction between development priority and the deployment reference |
| Which activation families were actually compared? | [Precision results](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/precision-campaign-20261104/summary.json) | Keep original family identities, finite training budgets and consumed cohorts |
| Did private vocabulary or small caches help? | [Completed component report](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/component-campaign-20261105/report.md) | Use matched controls and separately account for memory and request speed |
| Why are fast binary dots insufficient? | [Binary pipeline profile](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/binary-pipeline-profile-20261104/README.md), [F008](../FINDINGS.md#f008-can-execution-speedups-compensate-for-model-quality) | Keep packing, target verification, acceptance and complete-request cost in view |
| How should a native comparison be interpreted? | [Historical comparison protocol](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/docs/CANDIDATE_BENCHMARKING.md), [current terms](domain.md) | Reuse the rationale; a future benchmark command and workload still need design and qualification |
| What optimizer or objective ideas were proposed? | [Sign optimization audit](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/training-audit-2026-10-01/03-binary-optimization.md), [research synthesis](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/nine-model-research-slate-2026-10-04/synthesis.md) | These contain source/literature arguments and hypotheses, not demonstrated HE4 gains |
| Does an old actor audit describe the current model? | [Older QAT source audit](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/design-readiness-20261104/qat-source-audit.md), [current model](model.md) | Check era, objective, trainable groups and state schema before transferring a claim |

## What stays in the original repository

Campaign reports, failed attempts, raw-run locators and original source revisions stay
as historical evidence. Summaries here preserve the model, target, workload, training
budget, outcome and limitations needed to interpret them. We do not copy command
history, superseded authorizations, former job ownership or old runbooks into the
current setup instructions.

The [November 10 HE4broad snapshot](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/he4broad-20261110/report.md) was still marked running at the
pinned revision. Its interim readouts are not a closed finding here. It uses combined
captures and a different trajectory from the earlier runs; it cannot silently replace
F003 or F004. Any later conclusion needs its own completed evidence and provenance.

A historical report being available in Git does not prove that its large raw artifacts
still exist. Preserve the original evidence and verify artifact availability before a
reproduction. The [documentation audit](../evidence/documentation-audit-20261110.json)
records source selection, local verification and exclusions. Source links are pinned
and hashed locally; remote availability and the old experiments were not reverified.

## Teaching material

The original [from-zero tutorial](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/docs/learning/from-zero.html) explains neurons, attention and
speculative drafting with interactive toy examples. Its invented numbers are teaching
examples, not experimental results. It also contains an older statement that precision
choices are still undecided. Preserve the original; adapting it to the current HE4
model would be a separate tutorial update, not a new scientific finding.

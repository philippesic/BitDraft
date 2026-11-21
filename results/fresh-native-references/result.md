# The full-head leader and trained 4-bit control are within 1% in the same native Linux suites

The two drafters again deliver similar throughput on thinking-on, sampled and agentic DEV
requests. The 4-bit control accepts more on all three suites. It trained for 21,000 updates
while the leader trained for 335,812, so the comparison describes their attained performance
at different training lengths.

## Outcome

Four arms ran in the same suite session on the RTX 5080 with Qwen3.5-4B Q4KM, runtime-v8
and draft depth 3. Speed is relative to Baseline. Request suites use generated tokens
per request second, and agentic uses assigned tasks per primary task second.

| DEV suite | Leader speed | 4-bit control speed | Acceptance, Baseline / leader / control |
| --- | ---: | ---: | ---: |
| Thinking-on, 40 requests | 1.009× | 1.012× | 66.64% / 63.02% / 67.76% |
| Sampled, 40 requests × 3 seeds | 1.010× | 1.003× | 50.00% / 47.70% / 50.45% |
| Agentic, 60 assigned tasks | 1.110× | 1.110× | 63.40% / 81.74% / 85.37% |

Thinking-on used exact target-output checks. Sampled passed the established distribution
qualification and response checks. Every agentic arm, including target-only, completed
37 of 60 assignments and passed the benchmark's consistency checks. The 23 failures per
arm are included in the timing.

## Scope and limitations

These are reused DEV suites on one device with one suite repetition and one arm order.
The sampled distribution qualification does not imply identical sampled responses.
The leader uses its full vocabulary table and the control uses Baseline's file layout.
The protected confirmation reserve and release-file forms were outside this comparison.
Source-group bootstrap intervals describe the suite mix, without estimating session-order
or deployment variance. The scope matches the earlier
[one-bit attribution](../one-bit-attribution/result.md), which measured the two drafters
in separate sessions. The 4-bit control still starts from trained Baseline weights.

## Decision

Keep these observations as reusable native Linux references for the measured workload
and runtime. The same-session comparison supports similar throughput under these settings,
and leaves attribution and generalization to the separate matched-training and confirmation work.

## Evidence and reproduction

The executed benchmark source is
`8180f20857446d16e78f1f04a827dd976267ae3b0c3a20d71d1f0997ed3c4935`.
Specifications, result reports and runtime evidence are under
`artifacts/one-bit-attribution-01/linux-references/runs/{thinking,sampled,agentic}/`.
Use the stored specifications with `benchmarks --frozen compare` for request suites
and `benchmarks --frozen live` for the agentic suite, under the GPU lease.
[Compact evidence](result.json) records the authenticated models, runtime, workloads,
all arm rows and registration receipts. It retains the inputs for
`benchmarks.statistics.paired` with 2,000 draws and a family of two, setting every
row's token credit to 1 for assigned-task rate. The sampled interval in the compact
record is a recomputation from the same sampled rows with generation seeds grouped by source.

## Artifact disposition

Keep the three native suite outputs and their inputs for the registered references.
All twelve arm observations are authenticated in the registry. The registry owns
their retention until the corresponding native protocol/workload is retired.
Prior observations and canonical references remain available. No files were retired.

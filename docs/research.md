# Research context

The objective is faster complete requests with the target's verification semantics
preserved. HE4 is the development candidate; external DFlash Q4_K_M remains the deployment
reference. Csmallbroad is a compact fallback, not a promoted winner. Current model and
execution behavior live in [model](model.md) and [execution](execution.md).

## What prior work tells us

| Question | Relevant knowledge |
| --- | --- |
| Why HE4 and a compact fallback? | [Retained precision, latency and memory tradeoffs](../results/retained-drafter-tradeoffs/result.md) |
| Did longer training help? | [Fixed-data HE4 failure](../results/he4-fixed-data-continuation/result.md), [broader-data trajectory](../results/he4broad-fresh-source-training/result.md), [inconclusive Csmallbroad continuation](../results/csmallbroad-continuation/result.md) |
| Which activation choice has evidence? | [Activation family comparison](../results/activation-family-comparison/result.md), [fixed clipping control](../results/fixed-versus-learned-clipping/result.md) |
| Can lower kernel cost compensate for quality? | [Binary execution screen](../results/binary-method-comparison/result.md), [complete execution costs](../results/execution-speed-and-model-quality/result.md) |
| What did earlier models establish? | [Turing quantization](../results/turing-quantization/result.md), [released block drafters](../results/released-block-drafters/result.md), [equal-budget A8 recovery](../results/a8-equal-budget-recovery/result.md) |
| Which correctness lessons survive? | [Verifier controls](../results/verifier-and-runtime-controls/result.md), [CPU gradient/export contracts](../results/cpu-training-contracts/result.md), [cache and arithmetic parity](../results/native-parity-diagnostics/result.md) |
| What remains a hypothesis? | [Research idea map](../results/research-hypotheses/result.md), [data and objective controls](../results/data-and-objective-controls/result.md) |

The complete entrance is [FINDINGS](../FINDINGS.md). Results preserve model/target eras,
data roles, budgets, failed attempts and limitations. Historical Qwen3 F16 EAGLE/DSpark
rates cannot be put on the same leaderboard as modern Qwen3.5 HE4 rates. Development
repetitions reduce timing noise without creating independent quality examples.

## Using the evidence

Useful reports are local under `results/sources/`; the independent external archive
preserves their supporting history and implementation. See [evidence access](evidence.md).
Old runbooks and proposal records are not current instructions or authorizations.
Some states were deliberately retired; a readable finding does not guarantee that its
experiment can be resumed. The current [support boundary](support.md) describes what this
repository actually implements and qualifies.

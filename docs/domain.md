# Domain terms and metrics

## Names

Each name below is a separate identity. Do not relabel one as another.

| Name | Meaning |
| --- | --- |
| HE4 | Development drafter: one-bit FFN weights, four-bit gate/up inputs, eight-bit down inputs, Q4K head |
| Baseline | Fixed external DFlash Q4KM drafter paired with the selected target |
| Candidate | Specific experimental drafter recipe and checkpoint |
| Incumbent | Best supported experimental candidate; does not replace Baseline |
| Target-only | Selected target running without a drafter |
| W1A8 | One-bit weights/eight-bit activations in declared components; name the model family |
| DSpark Q4KM | Historical DSpark-era baseline |
| Csmallbroad, B0, foundation | Retained historical states with their own identities |

The retained comparisons summarized here do not show HE4 beating Baseline. See [findings](../FINDINGS.md).

## Data roles

Every prompt has one role: train, DEV, calibration or heldout. heldout24 and every
alias of it are evaluation-only, permanently. They never enter training, calibration,
selection, sampling or conversion. Consumed DEV is selection data, not fresh
confirmation. A prompt with unknown prior training use cannot join a new protected
evaluation set. Details: [data](data.md).

## Metrics

- Complete-request throughput: total emitted tokens divided by total full HTTP
  request wall time. Never a mean of per-request rates.
- Acceptance: total accepted tokens divided by total proposed tokens.
- Exposure at a position: the number of window visits that cover it. Mean exposure is
  total position visits divided by unique positions. Window passes are total window
  visits divided by unique windows. These are different denominators.

## Clocks

Host-traced draft wall time, GPU event time, server decode wall time, full HTTP request
wall time and training update time measure different scopes. Report each under its own
name and never pool or compare across them. Diagnostic instrumentation changes timing;
keep it separate from clean timing runs. Repetitions estimate timing variation, not
generalization to unseen prompts.

## Comparisons

Two results are comparable only when they match on machine and software, era, target,
tokenizer and chat template, workload with prompt IDs, roles and prior consumption,
output cap, complete process blocks, binary and runtime pins, instrumentation, draft
depth, KV cache and generation controls. Declared candidate differences keep their
exact identities. Never waive a clock, workload or correctness mismatch, keep only
favorable retries, invent a missing field or add a force option. Keep raw token outputs
and activation evidence so aggregates can be recomputed. Results from different machines
stay separate. Historical DSpark and modern DFlash results get no speed ratio.

## Selection

A checkpoint is selected by a valid, complete, matched serving score attached to a
durable full state. Higher score wins; an exact tie keeps the earlier state. DEV loss,
cross-entropy and acceptance are diagnostics, not selection scores. A final decline,
an interrupted panel, invalid output, a nonfinite score or a state that is not durable
cannot replace the current best. A DEV peak does not establish a serving peak.
The selection mechanism itself is not implemented yet.

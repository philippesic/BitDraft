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

Match target precision, tokenizer/template, exact prompts and output caps, generation
controls, runtime, draft depth, KV policy and data role. Saved references bind those
identities and raw evidence. Never pool incompatible protocols. Candidate-only checks
verify output/stopping against saved target outputs and compare acceptance with the
fixed baseline and previous leader. Historical timings are descriptive, even on the
same GPU; they do not supply fresh paired speed evidence. A hardware/driver change
requires new controls. Keep failures and unfavorable observations.

## Selection

A valid complete60-request screen can select a provisional acceptance leader attached
to a durable state/export. Higher accepted/proposed wins; exact ties retain the earlier
state. It does not establish better throughput or application performance. Full-request
selection, live outcomes and the latest recovery checkpoint remain separate. Invalid,
incomplete or nonfinite evaluations cannot select. DEV loss/top1 remain diagnostics;
repeated DEV selection is not unseen confirmation.

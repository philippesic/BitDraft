# Using the data and cost campaign in joint design

This campaign keeps the existing row-W1A8 actor fixed. It does not rank unknown
precisions or architectures, select a new feature recipe, or authorize long QAT.
Use the [backlog readiness brief](../design-readiness-20261104/readiness.md) for
feature evidence and the [contender workflow](../../scripts/contender_pipeline/README.md)
for staged execution. The campaign report supplies measured costs and data evidence.

## Decisions we can make with the evidence

- Compare **whole draft cost**, including generation, head and catch-up. A fast
  isolated projection or draft-generation timer can coexist with slower serving.
- Compare accepted tokens and total target verification rounds on the same
  requests. Acceptance percentages alone hide proposal-depth differences.
- Price data preparation, startup, export and native checks separately from
  update time. Equal updates need not mean equal GPU occupancy or equal labels.
- Use the larger admitted TRAIN bank when its target/taps/history contract fits.
  Reuse the source roster and exclusion ancestry if a new design needs recapture.
- Carry forward evidence-backed backlog components with compatible contracts;
  treat architecture-dependent features as conditional options. Existing trainable
  row scales are part of this reference, not a new group-scale comparison.

## What still needs design-specific testing

For each shortlisted precision/architecture, record target and runtime, feature
taps, activation/weight precision, scale granularity, head mapping, draft depth,
attention/cache semantics, and which captured assets remain compatible. Qualify
its actual export and native loader before buying training. If an adapter is new,
exercise one optimizer update, checkpoint/resume and native output correctness.
Existing infra cannot certify an unimplemented model or kernel in advance.

Give viable candidates the same predeclared initial update budget and preserve
checkpoints. Compare whole-request speed and draft/round counters against frozen
DFlash with uncertainty. Promote clear gains; park unresolved candidates when
further evidence costs more than its decision value. Do not permanently reject
an idea on noisy early training loss or a short native panel: a universal cheap
predictor has not been established. Timing repetitions reduce measurement noise;
they do not create additional independent evaluation prompts.

Reserve untouched confirmation for the small final shortlist. Apply this
campaign's TRAIN exposure index before selecting any future held-out groups.
Long-QAT is a separate decision after actual native viability, cost and training
transfer are demonstrated for the selected representation.

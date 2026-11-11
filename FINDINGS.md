# Findings

Current conclusions and their result pages. Historical measurements were not rerun
here. Result pages distinguish reported evidence from fresh verification.
See [closeout specification](docs/results.md).

## F001: Which drafter is the incumbent?

The external DFlash Q4KM drafter is the Baseline. [Result](results/baseline-identity/result.md).

## F002: Do the retained HE4 and Csmallbroad states beat Baseline?

Neither beats the external baseline; retain the latency and memory tradeoffs. [Result](results/retained-drafter-tradeoffs/result.md).

## F003: Did continuing HE4 training on the same data help?

The fixed-data continuation worsened acceptance and request throughput. [Result](results/he4-fixed-data-continuation/result.md).

## F004: Did fresh-source HE4broad training produce a better checkpoint?

It ended at 24.5658% acceptance and 174.113 tok/s, against 185.444 for target-only and 65.3257% and 249.662 tok/s for Baseline. [Result](results/he4broad-fresh-source-training/result.md).

## F005: How many prompts did the null-topic deduplication bug drop?

The missing-topic bug rejected distinct prompts; an eight-prompt discrepancy remains unresolved. [Result](results/null-topic-deduplication/result.md).

## F006: Did the accepted Csmallbroad continuation help?

The continuation produced no consistent throughput improvement. [Result](results/csmallbroad-continuation/result.md).

## F007: Did the original DFlash2 Q4 and F16 native paths work?

No, both produced nonfinite logits. [Result](results/dflash2-native-logit-failure/result.md).

## F008: Can execution speedups compensate for model quality?

Not so far. [Result](results/execution-speed-and-model-quality/result.md).

## F009: Did any binary method beat the retained W1A8 pipeline?

No. [Result](results/binary-method-comparison/result.md).

## F010: Which retained training states survived the November 6 cleanup?

34 optimizer archives were deliberately retired, and five protected full states survived. [Result](results/retained-training-states/result.md).

## F011: Does the retained modern index reconcile?

Yes. [Result](results/modern-index-reconciliation/result.md).

## F012: What did the activation-family comparison establish?

In the November 4 modern 4B/RTX 5080 campaign, learned-small G4/D8 was faster on complete heldout requests than the other retained trained activation-family endpoints, but remained slower than external DFlash Q4_K_M. [Result](results/activation-family-comparison/result.md).

## F013: Did the historical binary pipeline profile establish binary acceleration?

No general binary-acceleration claim followed. [Result](results/binary-pipeline-profile/result.md).

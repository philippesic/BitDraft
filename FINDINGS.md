# Findings

Current conclusions and their result pages. Historical measurements were not rerun
here. Result pages distinguish reported evidence from fresh verification.
See [closeout specification](docs/closeout.md).

## F001: Which drafter is the incumbent?

The external DFlash Q4KM drafter is the Baseline. [Result](docs/findings/f001.md).

## F002: Do the retained HE4 and Csmallbroad states beat Baseline?

Neither beats the external baseline; retain the latency and memory tradeoffs. [Result](docs/findings/f002.md).

## F003: Did continuing HE4 training on the same data help?

The fixed-data continuation worsened acceptance and request throughput. [Result](docs/findings/f003.md).

## F004: Did fresh-source HE4broad training produce a better checkpoint?

It ended at 24.5658% acceptance and 174.113 tok/s, against 185.444 for target-only and 65.3257% and 249.662 tok/s for Baseline. [Result](docs/findings/f004.md).

## F005: How many prompts did the null-topic deduplication bug drop?

The missing-topic bug rejected distinct prompts; an eight-prompt discrepancy remains unresolved. [Result](docs/findings/f005.md).

## F006: Did the accepted Csmallbroad continuation help?

The continuation produced no consistent throughput improvement. [Result](docs/findings/f006.md).

## F007: Did the original DFlash2 Q4 and F16 native paths work?

No, both produced nonfinite logits. [Result](docs/findings/f007.md).

## F008: Can execution speedups compensate for model quality?

Not so far. [Result](docs/findings/f008.md).

## F009: Did any binary method beat the retained W1A8 pipeline?

No. [Result](docs/findings/f009.md).

## F010: Which retained training states survived the November 6 cleanup?

34 optimizer archives were deliberately retired, and five protected full states survived. [Result](docs/findings/f010.md).

## F011: Does the retained modern index reconcile?

Yes. [Result](docs/findings/f011.md).

## F012: What did the activation-family comparison establish?

In the November 4 modern 4B/RTX 5080 campaign, learned-small G4/D8 was faster on complete heldout requests than the other retained trained activation-family endpoints, but remained slower than external DFlash Q4_K_M. [Result](docs/findings/f012.md).

## F013: Did the historical binary pipeline profile establish binary acceleration?

No general binary-acceleration claim followed. [Result](docs/findings/f013.md).

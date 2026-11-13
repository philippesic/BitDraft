# Findings

- FP16 feature storage passed the matched 1,024-update HE4/Q4KM qualification,
  including separate DEV loss and 120 native requests; use it for scoped supplemental
  capture. [Result](results/fp16-storage-adoption/result.md).

Established conclusions and scoped negative/inconclusive results. Historical
measurements were imported, not rerun. Read each result before reusing its claims.
See [closeout specification](docs/results.md).

## Current candidate and retained controls

### F001: Which drafter is the incumbent?

The external DFlash Q4KM drafter is the Baseline. [Result](results/baseline-identity/result.md).

### F002: Do the retained HE4 and Csmallbroad states beat Baseline?

Neither beats the external baseline; retain the latency and memory tradeoffs. [Result](results/retained-drafter-tradeoffs/result.md).

### F003: Did continuing HE4 training on the same data help?

The fixed-data continuation worsened acceptance and request throughput. [Result](results/he4-fixed-data-continuation/result.md).

### F004: Did the November 8 forty-sequence HE4broad run beat Baseline?

It ended at 24.5658% acceptance and 174.113 tok/s, against 185.444 for target-only and 65.3257% and 249.662 tok/s for Baseline. [Result](results/he4broad-fresh-source-training/result.md).
2026-11-12: this is a historical run, not HE4's current standing; see F014.

### F005: How many prompts did the null-topic deduplication bug drop?

The missing-topic bug rejected distinct prompts; an eight-prompt discrepancy remains unresolved. [Result](results/null-topic-deduplication/result.md).

### F006: Did the accepted Csmallbroad continuation help?

The continuation produced no consistent throughput improvement. [Result](results/csmallbroad-continuation/result.md).

### F007: Did the original DFlash2 Q4 and F16 native paths work?

No, both produced nonfinite logits. [Result](results/dflash2-native-logit-failure/result.md).

### F008: Can execution speedups compensate for model quality?

Not so far. [Result](results/execution-speed-and-model-quality/result.md).

### F009: Did any binary method beat the retained W1A8 pipeline?

No. [Result](results/binary-method-comparison/result.md).

### F010: Which retained training states survived the November 6 cleanup?

34 archives were retired and five survived November 6; those five obsolete states were
retired November 12 after compact evidence preservation. [Result](results/retained-training-states/result.md).

### F011: Does the retained modern index reconcile?

Yes. [Result](results/modern-index-reconciliation/result.md).

### F012: What did the activation-family comparison establish?

In the November 4 modern 4B/RTX 5080 campaign, learned-small G4/D8 was faster on complete heldout requests than the other retained trained activation-family endpoints, but remained slower than external DFlash Q4_K_M. [Result](results/activation-family-comparison/result.md).

### F013: Did the historical binary pipeline profile establish binary acceleration?

No general binary-acceleration claim followed. [Result](results/binary-pipeline-profile/result.md).

### F014: Where does the current 220,095-update HE4broad state stand?

On 600 Q4KM DEV prompts it reached 41.85% acceptance (0.84 of 2 drafts per round) and
180.9 tok/s: 1.07× target-only, 0.85× Baseline (65.77%, 213.5 tok/s). It does not beat
Baseline. [Result](results/he4broad-final-standing/result.md).

### F015: Does any drafter speed up the live tool tasks?

Not measurably. Over 36 task runs target-only took 7.40 s per assigned task, Baseline 7.49
and HE4 7.99; each drafter doubled prompt processing, cancelling its decoding gain.
Twelve tasks from one project.
[Result](results/he4broad-final-standing/result.md).


## Earlier experiments and durable lessons

### Packed low-bit paths did not beat ordinary EAGLE in the Turing suite

[Result](results/turing-quantization/result.md).

### Changing activation precision alone did not close the W1Ax serving gap

[Result](results/turing-activation-policy/result.md).

### Scale fitting and selective precision rescue improved acceptance without earning promotion

[Result](results/scale-and-head-recovery/result.md).

### The combined A8 recipe lost to its reference and both trailed Q4_0

[Result](results/a8-equal-budget-recovery/result.md).

### Released seven-token DSpark and DFlash beat Q4_0 EAGLE on Turing

[Result](results/released-block-drafters/result.md).

### A large catch-up penalty was not inherent to DSpark

[Result](results/historical-dspark-cost/result.md).

### Better fusion reconstruction did not establish better native acceptance

[Result](results/fusion-fitting/result.md).

### Matching forward values did not guarantee matching learned-quantizer gradients

[Result](results/cpu-training-contracts/result.md).

### Exact bounded CPU replay required matching native arithmetic and cache history

[Result](results/native-parity-diagnostics/result.md).

### Verifier parity and runtime speed depend on the exact arithmetic contract

[Result](results/verifier-and-runtime-controls/result.md).

### Broader data and simple objective additions did not establish serving gains

[Result](results/data-and-objective-controls/result.md).

### Fixed clipping beat learned clipping in one matched held-out comparison

[Result](results/fixed-versus-learned-clipping/result.md).

### Remote input materialization dominated the audited HE4broad training clock

[Result](results/input-locality-and-clock/result.md).

### Historical results survived cleanup but many exact model states did not

[Result](results/artifact-retirement/result.md).

### Historical research proposals remain hypotheses rather than measured HE4 improvements

[Result](results/research-hypotheses/result.md).

### Skipping unused catch-up head work gave a small matched throughput gain

[Result](results/unused-head-pruning/result.md).

### The packed W1A1 head was slower than ordinary EAGLE on RTX 5080

[Result](results/packed-head-native-comparison/result.md).

## Evidence preservation

[Historical reports and committed source are independently recoverable](results/archive-preservation/result.md); raw-artifact limits remain explicit.

[Historical raw outputs are replaced by compact evidence](results/historical-output-retirement/result.md):
Windows archives and closed qualification payloads are retired after report/source
preservation, member authentication and comparison-total verification; current development
inputs and saved-control dependencies remain.

## Development benchmarks

[Saved controls and resident checks](results/reusable-acceptance-checks/result.md):
complete 60-request training pauses took 77–81 seconds including cold data materialization.
Candidate-only 120-request/18-prefix/12-live comparisons took 6–7 minutes per precision;
failure recovery preserved exact state and reused historical controls.

## Capture storage

[The frozen 6K TRAIN bank is fully captured](results/frozen-train-bank-capture/result.md):
all 6,000 records passed product validation and metadata admission; FP16 arrays use about 230 GiB.

[Early rounding screen](results/fp16-feature-screen/result.md), followed by
[matched training/native adoption](results/fp16-storage-adoption/result.md).

- [HE4broad checkpoint migration](results/he4-checkpoint-migration/result.md): the final original checkpoint reproduces its next update exactly after correcting cache gradients.

- [Earlier bounded capture qualification](results/fp16-capture-qualification/result.md):
  superseded for adoption by the matched 1,024-update study above.

[Legacy artifact closeout](results/legacy-artifact-closeout/result.md): compact evidence
and independent source/WIP recovery replace obsolete large raw run trees. Current corpora
and the original HE4broad continuation parent remain intact. The old 5080 and 2080 Ti
checkouts and obsolete external artifacts have been removed after preservation checks.

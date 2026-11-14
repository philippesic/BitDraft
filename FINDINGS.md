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
2026-11-13: measured under depth 2 and runtime-v5; F023 gives the protocol-v3 standing.

### F015: Does any drafter speed up the live tool tasks?

Not measurably. Over 36 task runs target-only took 7.40 s per assigned task, Baseline 7.49
and HE4 7.99; each drafter doubled prompt processing, cancelling its decoding gain.
Twelve tasks from one project.
[Result](results/he4broad-final-standing/result.md).
2026-11-13: under protocol v3 Baseline is 1.48× faster than target-only on these tasks (F023).


### F016: Which step-size recipe should fresh grouped HE4 training use?

Over 7,800 fresh updates (the old 8-hour run's 3.3M labels), decaying the step size raised
60-request screen acceptance from 51.4% to 53.3%. A bounded clamped row latent scale needed
a 4× sign step to match it (53.3%; 50.4% at the old step). All arms beat the old state's
46.4% on that screen; Baseline scores 69.6%. Arm D's state is kept for continuation. One seed.
[Result](results/step-size-schedule-latent-scale/result.md).

### F017: What does exact speculative verification cost, and is it causing problems?

`GGML_CUDA_DECODE_CONSISTENT=32` costs every arm about 4–6% of decoding (19% for target-only
at 8K–16K inputs). It makes rounds jump 28% beyond depth 3, so exact Baseline decodes fastest
at depth 3: +6.6% over depth 2 on the 120-request panel (217.7 vs 204.3 tok/s), within 3% of
the best inexact setting. Without it, outputs differ only through near-tie flips (logit margin
≤0.40, always the target's second choice). Speculative runs differ from target-only in 55% of
panel requests at depth 4, and target-only itself changes in 62%. No wrongly accepted draft
occurred in 11,055 traced rounds. Keep exactness.
[Result](results/cuda-decode-consistency-cost/result.md).

### F018: Why does any drafter double prompt processing?

Runtime overhead around the drafter, the same for Baseline and HE4. About 0.057 ms per
prompt token is the GPU idling while target features are copied through host memory.
Target kernels are unchanged and drafter kernels are about 2%. The drafters' 4,096-token
sliding window also grows each context checkpoint from 50 to 130 MiB, about 100–140 ms
per short cached call. Fixes that keep target outputs exact are proposed, not built.
[Result](results/speculative-prefill-overhead/result.md). Correction 2026-11-13: built
and measured in F020.

### F020: Do the exact prompt-overhead fixes work?

Yes. Runtime-v6 keeps target features on the GPU and copies checkpoints into reused,
pre-faulted host memory. Every output, token ID and proposed/accepted count matched
runtime-v5 on synthetic prompts, 60 DEV requests and 12 live DEV tasks. Drafter prompt
overhead fell 64–91%: at 16,000 tokens from 1,027 to 119 ms, and a 32-token cached call
from 210 to 55 ms. On live tasks Baseline took 70.1 s against target-only's 83.8 s
(v5: 84.6 against 89.5), and HE4 matched target-only.
[Result](results/drafter-prompt-overhead-fixes/result.md). 2026-11-13: adopted as the
benchmark runtime in protocol v3 (F023).

### F019: Does the fixed depth-2 exact protocol treat HE4 and Baseline fairly?

No. Depth 2 caps Baseline below its best exact depth (3) while HE4 loses nothing, so on the
120-request Q4KM panel HE4's speed ratio to Baseline is 0.87 at depth 2 against 0.80 with each
drafter at its best exact depth (arm D: 0.88 vs 0.82), about 7–10% too favourable. Fixed depth
3 matches the best-depth ratios within 1%. Exact mode leaves acceptance on identical text
unchanged and does not systematically favour HE4. Depth 3 is proposed as the shared default.
[Result](results/draft-depth-fairness/result.md). 2026-11-13: adopted in protocol v3 (F023).

### F021: Can a drafter's serving depth be chosen per deployment without a panel sweep?

Yes, on the measured Q4KM/runtime-v5 deployment. `benchmarks calibrate-depth` takes 2–4 GPU
minutes. It combines one depth-15 survival run with short cost sessions at depths 0–15 and
recommended each drafter's measured best panel depth: Baseline 3, HE4 220,095 2 (3 within
2%), arm D 3 (2 within 2%). Predicted depth-3/depth-2 ratios were within 1.5% on the
complete-request clock and 2.5% on decode; screen decode ratios at depths 2–15 within 1–3%. Absolute speeds vary about ±5% between
sessions. Summing per-position acceptance instead of splitting deep rounds undercounts
shallow depths by up to 14%. [Result](results/draft-depth-calibration/result.md).

### F022: Should HE4 train on the block size it is served with?

Yes. At about equal optimizer time from arm D's start, training every update on 4-slot
blocks raised screen acceptance over 16-slot training by 6.6, 6.0 and 5.1 points at depths
2, 3 and 4 (60.0%, 49.1%, 40.7%). Random 3–16-slot blocks with reach-weighted positions
gained 4.1–5.0 points but trailed 4-slot blocks by 1.0–1.6 at every tested depth; they
keep deeper positions better on DEV, untested natively beyond depth 4. Baseline still
leads by about 10 points. The 4-slot state is kept. One seed.
[Result](results/block-size-and-reach-weights/result.md).

### F023: How do Baseline and S4 stand under benchmark protocol v3?

Protocol v3 (runtime-v6, exact depth 3, no RAM prompt cache) is the default; target
outputs and depth-3 draft counts match runtime-v5 exactly. Baseline runs 1.36–1.43× target-only
on the 60- and 120-request cohorts, 1.70–1.72× on the prefix harness and 1.34–1.48× on live
tool tasks, on both precisions. S4 reaches 0.88–0.93 of Baseline's request speed, 11–13
acceptance points lower; on live tasks with matching trajectories it ties Baseline
(Q4KM 47.3 vs 47.6 s, NVFP4 30.5 vs 30.0 s). The 220,095 export reaches 0.79 of Baseline on
the panel. S4's live trajectory diverged on 1–2 of 12 tasks through cached-prefix reuse after
a drafted end-of-turn token, so it has no live reference. Depth-2/runtime-v5 ratios are not
comparable. [Result](results/benchmark-protocol-v3/result.md).

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

## Training throughput

[Grouped updates](results/grouped-updates/result.md): training one group of windows per
update raised RTX 5080 update throughput from about 90 to about 2,100 labels/s at typical
context within a 14 GiB device cap; before prefetching, disk loading limited end-to-end
training to about 970 labels/s, and step sizes still need retuning.

[Sequence prefetch](results/sequence-prefetch/result.md): loading the next group's sequence
in the background, hashing feature bytes in memory and converting FP16 to FP32 on the GPU
raised end-to-end throughput from 810 to 1,685 labels/s on real weights, within 9% of
update-only throughput, with bit-identical states.

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

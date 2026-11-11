# Architecture and precision design readiness

**We are ready for the joint design discussion.** The known backlog is reconciled,
the current candidate provides an exercised training/evaluation reference, and
feature dependencies and implementation costs are explicit. No final architecture,
precision, new feature recipe or major long-QAT allocation has been selected.
This is readiness to make an informed choice; a changed design still needs its
specific implementation and qualification work.

## What the backlog now tells us

[The ledger](backlog-inventory.json) preserves all **98 original training segments**
across **34 recipe families**, plus all **F01–F32 feature IDs**. Those are overlapping
ledgers, not130 independent ideas. All51 retained compact source receipts rehash
correctly and every original segment record matches the canonical catalog.

Many historical tests were very short:80/98 used under300 measured TRAIN seconds,
and32 under60. The ledger distinguishes full-request results from native-prefix
scores, reconstruction proxies, implementation-only preparation and missing
historical replay assets. It does not permanently reject a family. Optional
exposure metadata is still unresolved; some raw captures/intermediate states
were removed by the previously authorized cleanup. Preserved reports are not
substitutes for those exact bytes.

[The feature map](decision-feature-map.md) connects each family and feature to
its mechanism, evidence, dependencies, implementation state and cheapest useful
next test. [The design agenda](design-questions.md) orders the decisions we need
to make together.

## What we measured on the fixed candidate

Current configuration: Qwen3.5 4B Q4_K_M target, RTX5080, frozen mode32 runtime,
row W1A8, depth2 and fixed binary signs. It trains **76,880 per-output-row scales
in nine tensors**. These are not nine scalar values or within-row group scales.

We tested root-heavy depth weighting, label smoothing and their combination,
with the same parent/data/optimizer/replay and960 additional updates each. The
control and apparent best/worst treatments then received another960 updates.
Both native panels used12 already-consumed DEV prompts, five alternating
repetitions, matched target/DFlash/initializer controls and exact native counters.

| Feature versus its same-budget control | First960 updates | After1920 updates |
| --- | --- | --- |
| Depth weights1.5/.5 | +1.25%;95% interval−0.32% to+3.45% | −0.67%;−2.63% to+1.07% |
| Smoothing.05 | −0.84%;−3.99% to+2.26% | −1.48%;−4.11% to+0.69% |
| Combination | −0.07%;−2.35% to+2.52% | Not extended under the frozen rule |

**No extra feature established a throughput benefit.** These exploratory,
unadjusted intervals and one training trajectory per treatment do not establish
universal failures, cross-design transfer or an early-pruning rule. The previously
selected replay5760 remains the recorded candidate. The prior fresh24 transfer
result remains valid; this study does not claim to have fixed generalization
causally or produced a new untouched confirmation.

All **1,246 panel/qualification responses** match target tokens, text and stopping;
960 are measured timing requests. Two additional known-FIT recaptures also match.
The raw audit reconciles metrics against every request and verifies model,
runtime/source pins and released groups. See [first-stage results](objective-stage1.json),
[longer results](objective-stage2.json) and the linked raw audits.

The paid candidate needed2785 counted speculative rounds versus DFlash1900 for
the same4685 output tokens in the first panel. Aggregate generation time divided
by those counts was8.876 versus8.646ms. This includes boundary/catch-up work;
it is not an isolated verifier or GPU-kernel timer. A separate single verbose
pass observed a cheaper narrow drafting-method span for W1A8,1.498 versus1.777ms,
but that method omits catch-up. This supports focusing on accepted continuations
and complete request cost rather than a kernel or acceptance-percentage shortcut.
Do not mix these DEV80-token rates with the previous natural256-token panel.

## Features and decisions worth carrying forward

| Area | What is ready or established | What remains conditional |
| --- | --- | --- |
| Native histories, hard labels, splits and replay | Actual current data/recapture path works; replay has preservation evidence | Replay's extra speed/generalization benefit is unestablished; new designs need compatible histories/masks/margins |
| Loss weighting and smoothing | Current implementation/gradients/resume/native tests work | No demonstrated speed win here; do not add these by default or reject them universally |
| Optimization and curricula |19 retained CPU loss/optimizer/state tests pass | Some helpers hard-code five positions or use different normalization; adapt explicitly |
| Probability distillation |404 eligible retained Q4 teacher rows join exact FIT token histories | Conditional-head and full-target objectives differ; teacher/runtime/gradient integration is not admitted |
| Head and vocabulary | Current head membership and teacher probability mass are quantified | Head changes, shortlists, factors and full-target support belong in the joint design |
| Trainable signs | Separate selected-O branch has meaningful deployed flips and resume/native evidence | Recent fresh24 success is scale-only; all-nine-sign QAT is not qualified |
| Group scales, planes, rotations, corrections | Costs, historical evidence and state/export/kernel dependencies mapped | These are representation/architecture choices; no new variant tested |
| Transfer/packing/catch-up | Current traced decoder already returns compact candidate IDs/logits | F26 is already present in observed paths; remaining catch-up/head GPU work needs its own measured case |

A16-sample FIT gradient diagnostic found CE drove most sampled updates; replay
was usually inactive, with occasional large corrections. Its observer leaves
control state byte-identical. This does not support a blanket claim that replay
always overwhelms the learning signal. [Diagnostic](gradient-diagnostic.json).

## Concrete inputs for our design discussion

- Hypothetical group128 scales, holding today's matrix shapes/signs fixed, add
  about6.65MiB of F32 scale payload and require scaled partial sums. The current
  loader/kernel does not support that representation. The old grouped128+R512
  experiments bundled rotation and grouping and cannot isolate either effect here.
- The current50k head covers97.37% of returned tokens in the consumed natural24
  panel, and94.02% in the separate DEV12 panel. These are membership fractions,
  not attainable acceptance or speed bounds. [Head audit](head-coverage.json).
- On404 eligible FIT teacher rows, mean probability mass inside that head is
  95.60% atT1 and81.60% atT2. A conditional-head KL/CE objective is not the same
  as a full-target objective. The known excluded conditional row stays excluded.
  No hidden-MSE substitute or KD speed benefit is claimed. [Teacher audit](teacher-coverage.json).
- Synthetic F32 masters+two Adam moments at today's full-sign geometry serialize
  to about2.8GB. Actual save+fsync was11.12s; load9.75s; hashes restore exactly.
  This measures CPU/storage behavior, not all-sign GPU/backward/model-state fit.
  Keeping every such checkpoint would require a retention plan. [I/O probe](checkpoint-io.json).

## Preparation that has actually been exercised

The new objective path matches the original trainer after48 updates, byte for
byte in parameters/Adam/RNG. Continuous training and a480-update split-resume
match in contract, parameters, Adam, all RNG and exposure. Every treatment changes
real parameters, exports a real GGUF and passes native loader/output checks.
Two FITs recapture byte-identical training arrays and token-aligned diagnostic
logits, then materialize114windows with no missing embeddings.

Reusable tools are under `scripts/design_readiness/`: fixed current-candidate
loss ablations, exact checkpoint comparison, native metrics/evaluation auditing,
coverage/teacher audits and resource probes. Completed commands, source freezes
and recovery evidence are under `runs/design-readiness-20261104-01/`.
The ablation trainer intentionally caps its cursor at7680; it is not an
unallocated long-run trainer. The established original branch retains its19200
horizon and paid continuations.

The process-reuse follow-up preserves exact parent reset, optimizer/RNG state
and exported bytes, but255.85s then251.19s does not establish a material workflow
speedup. It remains optional. One profiled control identifies the dominant cost:
data-union authentication164.42s inclusive, with repeated filesystem metadata,
opens and row reads. These nested times overlap. No provenance checks were
weakened to make the preparation look faster. [Profile](preparation-profile.json).

## Known work after choosing a design

1. Bind the chosen architecture, precision, scale granularity, trainable scope,
   head and target-feature contract. This determines which feature adapters apply.
2. Qualify any changed state/export/native graph, including meaningful updates,
   exact resume and actual memory measurements. Grouped kernels, all-sign
   state/export and unknown architectures are explicitly not already qualified.
3. Prepare adequate TRAIN breadth for that choice. Three source files and two
   manifests/all25 opaque index entries reauthenticate, without opening dedicated
   DEV/SEALED payloads. Current48FIT is a working small dataset, not proof of
   sufficient long-run diversity. Current1GiB resident-cache and bounded audit
   limits require an explicit sharding/streaming/resource plan for larger data.
4. Declare long-QAT allocation, checkpoint retention and native checkpoint tests.
   Use actual selected-design resources and modest early comparisons; preserve
   independent confirmation inputs and do not permanently prune from short losses.

Those are visible design-dependent tasks, rather than a promise that every
possible design is ready to train without implementation. [Data readiness](data-readiness.json)
and [QAT source/readiness details](qat-source-audit.md) retain the precise limits.

## Cost and release

Direct execution used7824 physical updates, including duplicate correctness and
workflow checks;6720 updates contribute unique treatment trajectories. Measured
update wall totals1054.29s (17.57min); nonoverlapping supervised experiment walls
sum4929.23s (82.15min), including preparation/evaluation. Do not add these clocks
or call them pure GPU time. Earlier saved noop preparation and overlapping CPU
research/audits are separate. [Cost ledger](cost-ledger.json).

All task processes are terminal;91 native groups have release receipts and no
owned group remains. Paid checkpoints and useful raw evidence are retained.
Five duplicate exports now share verified hardlinks, preserving all paths/bytes;
about204MB of duplicate payload was removed, with C: measured free bytes rising
203.43MB during cleanup. The synthetic2.8GB scratch was removed after its probe.
Task-local bytecode was removed; shared useful caches and unrelated work remain.
[Release and inventory](../../runs/design-readiness-20261104-01/cleanup-final.json).
No push occurred.

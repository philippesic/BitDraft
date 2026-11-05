# DFlash candidate stages: essential features, precision, then experiments

## Direction and authority

User correction, November 4, 2026: build a strong common feature foundation
FIRST, including high-impact compensation for binary weight quantization. Then
compare precision setups on that foundation. Only afterward expand into more
experimental feature tiers. “Essential” is the design scope selected for this
foundation, not a claim that each feature has already shown a local speed win.

The existing delegated vanilla W1A8 run is a **minimal control**, not the proposed
essential-feature baseline. Preserve that run, its allocation and its frozen
artifacts. Do not retrofit new features into it or relabel its results. A separate
foundation candidate must be implemented, trained and evaluated before the
precision comparison. This document records feature scope; it does not launch
new GPU runs or grant new training allocations.

Read [CANDIDATE_BENCHMARKING.md](CANDIDATE_BENCHMARKING.md) for matched comparisons,
[the vanilla registry](../experiments/dflash-w1a8-baseline-20261104/manifest.json)
for actual results, and the complete
[feature inventory](../experiments/dflash-feature-survey-20261104/inventory.json)
for evidence and Q/I identifiers. Current research phase: establish the essential
feature baseline that will support precision selection. No precision winner yet.

## Stage 0: essential-feature baseline, before precision selection

### Included common model features

- **Trainable binary signs and learned output-row scales (Q01).** Initially all
  18 DFlash FFN gate/up/down matrices; explicitly record any coverage exception.
- **Learned binary midpoint per output row (Q04 center portion).** Use
  `w = midpoint + row_scale * sign`, with zero midpoint as a reversible starting
  point. This admits two unequal levels around zero. Its computation includes
  a midpoint times the sum of the gained, quantized inputs; it is NOT an
  ordinary output bias. Native/export and training arithmetic must implement
  the same definition. A nonzero calibrated
  midpoint initializer is a separate initialization test, not an unnoticed change.
- **Learned positive input-channel gains (Q02)** in addition to output-row scales, as the
  proposed common channel-compensation mechanism. Initialize gains to one and
  apply them before the activation quantizer. Use one shared gain vector for
  gate/up per layer and a separate down-input gain vector. This preserves the
  possibility of shared gate/up activation quantization; it is an initial design
  constraint, not a measured optimum. Declare the positivity/scale convention.
  Qualify runtime cost; do not silently fold gains away if that changes binary
  weights or quantization.
- **Adapt the existing normalization gains (existing Q05)** without adding new
  normalization layers or residual branches.
- **Protected operations (I01):** preserve fusion, attention, shared embedding/
  head and KV at declared working precision initially. These exceptions are an
  explicit part of the foundation; W1 does not describe the entire model.

For the combined affine binary projection, define:

`u = Q_A(g * x)`

`y_o = s_o * sum_j(b_oj * u_j) + m_o * sum_j(u_j)`

Here `Q_A` returns dequantized quantized activations, `g` is the positive input
channel gain, `s_o` the nonnegative row scale, `b_oj` the binary sign and `m_o`
the row midpoint. Both terms consume the SAME `u`; using raw input for the
midpoint term is a different model. Include the shared input-sum computation
in native cost. Midpoint zero and gain one recover the row-scaled reference.

### Included training and evaluation essentials

- Effective sign updates, explicit latent magnitude/initialization and STE,
  separate sign/scale/midpoint/gain/norm learning rates, scale constraints and
  complete optimizer/RNG/data-cursor resume (Q16). Measure persistent sign changes,
  gradient health and validation quality. Do not equate many flips with learning.
- Choose/qualify the sign-update mechanism here, before fixing the foundation:
  latent-gradient updates versus a direct accumulated-gradient sign-flip rule
  (Q24 optimizer portion). Current vanilla AdamW already changes signs; a Bop-like
  rule is a candidate, not automatically necessary or already validated. Use a
  bounded from-zero comparison if evidence warrants changing the optimizer.
- Native target features/labels, correct masks/positions, representative TRAIN
  coverage and retained DEV separation (foundation Q15). No mislabeled old taps.
- Preserve the pinned, source-compatible DFlash objective, its supervision and
  original position weighting; audit these rather than assuming every released
  DFlash recipe uses the same targets. Original fusion,
  taps, topology and shared head remain base components, not added features.
- Data-aware scale initialization, baseline sign initialization and basic input
  range/outlier diagnostics (basic I04/I20), without an initialization sweep.
- Fix the extensible activation quantizer contract (I02/I03/Q06): signedness,
  rounding, saturation, scale granularity, zero-point handling, STE and export
  version. Qualify fixed A8 here. Stage 1 qualifies learned clipping/step size
  (gradients, resume, export and native arithmetic) before precision selection.
  Its implementation need not delay freezing the A8 foundation, and active
  learned clipping is not mandatory in the A8 default.

### Foundation acceptance and controls

Freeze a separate essential-feature reference after bounded QAT and native
qualification. Compare the bundle with the vanilla control and keep removal
ablations for midpoint and input gains; they are controls on the chosen scope,
not a reason to postpone all binary compensation until after precision selection.
Qualification requires valid execution, healthy training and acceptable bounded
resource cost; a noisy short pilot need not beat vanilla before this package can
serve as the precision reference. If a feature demonstrably harms correctness,
quality or native throughput, record a justified
revision of foundation scope rather than silently dropping it.

Midpoints and input gains are planned scope, not claims about implemented modern
DFlash support. Avoid a large feature search here: qualify one concrete package,
its update mechanism and its essential controls. Fix the package before comparing
precision, except for explicitly coupled quantizer choices below.

## Stage 1: precision setups on the common essential baseline

The agreed adaptive search is specified in [PRECISION_SEARCH.md](PRECISION_SEARCH.md).
It replaces the earlier A8/A4-only trio with four gate/up starting families:

| Arm | FFN gate/up input activations | Initial FFN down inputs | Common features |
| --- | --- | --- | --- |
| G8/D8 | A8 | A8 | Stage 0 package |
| G4/D8 | A4 | A8 | Same package |
| G2/D8 | A2 | A8 | Same package |
| G1/D8 | A1 | A8 | Same package; qualified two-level quantizer |

All listed FFN weights start binary. A widths describe projection inputs, not
residuals, accumulation, scales, softmax, head or KV. Down precision search includes
A8/A4/A2/A1, with an explicit G1/D1 candidate. Protected components remain declared;
G1/D1 is not a whole-model one-bit claim. Start other precisions identically, then
probe head/down reductions early and prioritize component adaptation by measured
cost, damage, recoverable memory and implementation readiness. Fusion, attention
and embedding weight precision join the finite queue; target precision never changes.

Precision-stage features and decisions:

- Per-operation activation widths and activation-scale granularity (I02).
- Fixed/calibrated clipping, scale and code-grid choices (I03).
- Learned clipping/step sizes (Q06), evaluated as part of the precision/quantizer
  combination, not withheld until after judging A4. Compare matched fixed/learned
  controls as needed; do not confuse quantizer capability with proven superiority.
- Sensitivity-guided A8 restoration for individual operations/layers (I01), and
  only then separately labeled higher-precision weight exceptions if necessary.
- Activation-precision adaptation/warmup schedule (basic Q18): allow matched QAT
  adaptation of each precision. A zero-training swap is diagnostic only. More
  elaborate curricula stay in Stage 3.

Select precision, quantizer and protected-operation map together. Use matched
unchanged continuations and preserve training history. Retain the bounded
multi-family frontier in PRECISION_SEARCH.md, separately from the deployment
incumbent; a useful small candidate need not win throughput. Confirm final
precision-attribution comparisons from the same frozen foundation and matched
training policy/exposure. Revisit interactions when later features earn inclusion.
Nominal activation width alone establishes neither smaller storage nor faster execution.

The November 5 follow-on [component plan](REMAINING_PRECISION.md) applies the
precision stage to optimized G4/D8's output head, embeddings, drafter KV and
scalar/norm-gain storage, with matched recovery. This is the user's selected next
precision scope, not a feature/topology change. Existing floating reductions and
target caches stay protected; the older blanket drafter-KV deferral is superseded
for this plan only.

## Stage 2: incremental parameterization and conditioning experiments

These are not required for the first common package:

- Learned within-row weight group scales (Q03); fixed group scales/affine
  calibration (relevant I05 portions) as controls. Row scales and row midpoints
  already belong to Stage 0; finer granularity is the experiment.
- Separate gate/up input gains (remaining Q02), relaxing the initial shared-gain
  constraint, with the cost of separate quantization measured.
- Ordinary additive projection bias (remaining Q04), distinct from a midpoint.
- Extra normalization and learned residual-branch gains (remaining Q05).
- Fixed rotations/channel equalization (I06) and training-integrated transforms
  (Q07), with explicit binary-representation and native execution checks.
- Advanced data-aware sign/center initialization or row-orientation rescue
  (remaining I04), including from-zero comparisons.

Test additions separately against the unchanged foundation/precision incumbent,
then combine supported winners. A diagnosed failure of the initial foundation
can justify pulling a specific remedy forward, with evidence and an explicit
scope revision; do not move whole speculative bundles into “essentials.”

## Stage 3: experimental training and supervision

- Broader STE, sign-latent initialization and parameter-group LR sweeps (Q16),
  beyond the healthy update mechanism fixed in Stage 0.
- Oscillation damping, selective freezing and stability schedules (Q17).
- Extended precision/coverage curricula (remaining Q18).
- Additional target-distribution distillation (Q12), beyond the source-compatible
  supervision preserved in Stage 0. It is not presumed necessary for the initial
  weight-compensation package; inadequate recovery may justify a specifically
  documented earlier remedy.
- Output/block reconstruction and intermediate-feature auxiliary losses (Q13).
- Verification-aware, prefix-survival and expected-progress losses (Q14), beyond
  original DFlash position weighting already included in Stage 0.
- Hard-prefix/margin retention, candidate-dependent replay and advanced data
  balancing (remaining Q15), beyond basic representative native replay.
- Reduced-state optimizers (remaining Q24), with numerical and resume checks.

Optimizer/initialization/curriculum hypotheses need from-zero arms when a trained
parent could conceal their effects. No local or literature result is presumed
transferable to this binary DFlash merely because it helped another model.

## Stage 4: additional capacity and trained computation

Only after cheaper changes are measured, consider:

- Task-trained low-rank residual correction (Q08).
- Task-trained sparse outlier correction (Q09).
- Additional binary residual planes with learned amplitudes (Q10); this is
  additional capacity, not a strict one-plane W1 model. Fixed-plane I05 and
  post-training low-rank/sparse I07 variants are separate controls.
- Activation sparsification learned in the forward pass (Q11).

Account for packed bytes, metadata, extra launches and full-request cost. A
reconstruction-error improvement without native throughput benefit is insufficient.

## Stage 5: deferred architecture and proposal-system changes

Do not include these in the first precision/feature campaign:

- Learned acceptance/confidence prediction (Q19).
- Markov or other token-conditioned correction/candidate scoring (Q20).
- Learned head compression or vocabulary-candidate selection (Q21).
- Alternative feature fusion, target-tap choice, draft width or depth (Q22).
- Binary factorization, experts or learned binary codecs (Q23).
- Post-training factorization/capacity allocation (I08), structural pruning,
  slicing or frozen-head factorization (I19).
- Heuristic vocabulary shortlists (I16), tree/beam proposals, retries and suffix
  reuse (I17). Drafter KV quantization (I18) is now in the November 5 follow-on
  precision plan; target KV and attention-arithmetic changes remain outside it.
- Parallel drafting, trees, retries and other proposal-system/architecture changes
  remain deferred. W1A1 precision itself is now in Stage 1, including G1/D1.
  Historical W1A1 acceptance/latency recollections do not validate modern DFlash.

These can change the model or verification workload enough to require their own
explicit research decision and matched baseline comparison.

## Independent engineering lane and invariants

Correctness-preserving engineering can be tested when useful, without waiting
for a feature's QAT run: shape-specific kernel dispatch (I09), prepacking/shared
quantization/fusion (I10), CUDA graphs/buffers/synchronization (I11), equivalent
output/head extraction (I12), correct KV reuse and minimal target-feature catch-up
(I13), recipe-preserving training efficiency (I14), and stage/position diagnostics
(I20). Static proposal depth or confidence-gap policy sweeps (I15) are also
training-independent experiments, but alter the workload: freeze them within
precision/feature comparisons and report any later retuning separately.

“Training-independent” means a particular implementation needs no trained
parameters; it does not mean accuracy-independent or free to change mid-test.
Fixed transforms, clipping, corrections, shortlists and cache quantization can
change outputs and must be evaluated. Some need matched QAT adaptation to assess
fairly. Representation/execution optimizations require correctness checks.

For continuation comparisons, branch a candidate and an unchanged continuation
from the same parent with matched data/exposure and training budget. Optimizer,
initialization and curriculum comparisons also permit matched from-zero arms,
where starting from an adapted parent would conceal the hypothesis. Promote a separate moving
incumbent only with native evidence. Preserve frozen zero/initial references and
external DFlash Q4_K_M. No short-run result proves long-run convergence, and no
stage assignment means a feature is already validated or selected for deployment.

## Scope debate and agreement

Root and an independent Astra agent debated this revision on November 4.
The agreed package includes midpoint and gains BEFORE precision selection,
with bounded removal controls and explicit scope revision if evidence warrants.
The debate resolved four points:

1. Input gains are included in the first candidate, not an unconditional promise
   of deployment benefit. Shared gate/up and separate down gains are the concrete
   starting choice; removal controls distinguish contribution from bundle effects.
2. Midpoints consume the same gained, quantized stream as the signed dot product.
3. Learned clipping belongs to precision selection. Root challenged requiring
   its implementation before freezing an A8 foundation; Astra agreed that Stage 0
   needs the extension contract/fixed-A8 qualification, and Stage 1 needs the
   learned-quantizer implementation before selection. Unavailable implementation
   leaves that arm untested; it does not show A4 is inferior.
4. Group scales/rotations remain later conditioning experiments; additional
   distillation remains later supervision work. Healthy sign updates are essential
   now, but Bop itself is not mandatory. Neither agent claims empirical validation
   from this design agreement.

No remaining scope disagreement. Implementation/results must still be established
by the benchmark protocol. This supersedes the earlier midpoint-after-precision
roadmap, which misread the user's intended feature scope.

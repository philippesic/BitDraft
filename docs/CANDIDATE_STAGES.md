# DFlash candidate stages: precision first, then feature tiers

## Current direction and authority

User decision, November 4, 2026: establish one useful binary-weight foundation,
validate activation/mixed-precision setups on that common foundation, then work
incrementally through tiers of additional features. Do not build a large feature
bundle before selecting precision. The detailed tier ordering below is the
coordinator's recommended experiment queue, not evidence that these features
work or permission to launch every experiment automatically.

**Current research phase: precision selection.** The delegated vanilla W1A8 run
is its reference-building prerequisite (Stage 0), not a completed precision
comparison. No A4 winner or final mixed-precision recipe has been established.
Keep baseline training/supervision/evaluation delegated as the user requested;
keep the main conversation available for architecture discussion. Preserve its
existing run and budget. This document does not authorize new training budgets,
feature sweeps, architecture changes, or a replacement active goal.

Read [CANDIDATE_BENCHMARKING.md](CANDIDATE_BENCHMARKING.md) for comparison rules and
[the baseline registry](../experiments/dflash-w1a8-baseline-20261104/manifest.json)
for actual implementation/results. IDs below refer to the complete
[feature inventory](../experiments/dflash-feature-survey-20261104/inventory.json).
That inventory remains the evidence catalog; this file defines experiment order.

## Stage 0: common foundation and frozen references

Include these in the starting candidate, before comparing precision:

- Released modern DFlash topology, existing feature fusion, target taps, shared
  embedding/head and original within-block position-weighted token objective.
  These are base-model components, not new features.
- Trainable binary FFN signs and one learned nonnegative output-row scale per
  binary matrix row (Q01). Initial coverage: all 18 gate/up/down matrices.
- Train the existing drafter normalization gains (existing-gain portion of Q05).
  No extra normalization layers or residual gains yet.
- Matching native target-feature/label replay and correct masks, positions and
  data splits (foundation portion of Q15).
- Explicit latent-sign initialization, STE gradient rule, separate sign versus
  scale/norm learning rates and a resumable optimizer (foundation Q16).
  The current run uses latent weights with AdamW; direct sign flipping is a
  later comparison, not an already implemented baseline optimizer.
- A8 binary-matrix inputs with a fixed symmetric per-token absmax quantizer;
  preserve fusion, attention, embedding/head and KV at the declared working
  precisions (initial I01/I02). Record training simulation versus native execution.

Freeze the zero-update export and short-trained W1A8 checkpoint. Learned
midpoints, input-channel gains, group scales, learned clipping, rotations,
correction branches and new losses stay OFF in this reference. Do not restart
its existing run to add them.

## Stage 1: activation precision and mixed precision

Use the same foundation above for every arm. Compare these first:

| Arm | FFN gate/up input activations | FFN down input activations | Other operations |
| --- | --- | --- | --- |
| P0 reference | A8 | A8 | Same protected precision as Stage 0 |
| P1 mixed | A4 | A8 | Same as P0 |
| P2 low activation precision | A4 | A4 | Same as P0 |

Weights remain binary on the same 18 FFN matrices in P0/P1/P2. A8/A4 label
matrix inputs, not residual streams, attention softmax, accumulation, scales,
KV, embeddings or the output head. Document each of those separately.

Include in this stage because they define or interact directly with precision:

- Per-operation A8/A4 and activation-scale granularity (I02).
- Fixed/calibrated clipping thresholds, symmetric/asymmetric code grids and
  activation scaling (I03), with exact export/native arithmetic recorded.
- Learned activation clipping or step size (Q06), as a separately identified
  quantizer arm if plain A4 underperforms. Do not silently give only one precision
  more training or extra features. Compare an unchanged control with each arm.
- Sensitivity-guided protected operations (I01): first try restoring A8 to a
  failing operation/layer in an A4 candidate. If that is insufficient, separately
  test leaving that FFN projection at its original weight precision. This changes
  W1 coverage and must be labeled explicitly; it is not the same W1A4 arm.
- Fixed rotations/channel equalization (I06) or training-integrated transforms
  (Q07) are conditional precision-rescue arms, not defaults. Try only when a
  diagnosed outlier/quantization issue justifies their cost; do not fold them into
  every candidate. Verify that any transform preserves the intended binary form.

Select **precision plus quantizer plus protected-operation map** together.
Give each changed forward path matched short QAT adaptation and compare it with
an equally trained A8 continuation. A no-training A8-to-A4 swap is a diagnostic,
not a fair final rejection. Measure actual native request throughput as well as
acceptance and draft latency; narrower activations do not guarantee a faster
available kernel. Keep a runner-up, and treat the selection as provisional.

## Stage 2: low-cost parameterization additions

Hold the provisional precision/quantizer map fixed. Test separately, in this
suggested order; retain only demonstrated improvements:

1. Learned input-channel gains alongside existing row scales (Q02).
2. Learned binary midpoint/center per output row (Q04):
   `w = midpoint + scale * sign`. Initialize midpoint to zero. A learned additive
   projection bias is a separate Q04 arm, not equivalent to a binary midpoint.
3. Finer within-row weight group scales (Q03), including explicit metadata and
   execution cost. Fixed/calibrated group scales or affine corrections (relevant
   I05 portions) are diagnostic controls for these learned variants.
4. Extra normalization or learned residual-branch gains (remaining Q05), separate
   from adaptation of the original norms already present in Stage 0.

Do not launch all four as one bundle. After a clear winner, test its interaction
with the retained precision runner-up before treating precision as permanently
settled. This feedback check prevents the staged process from hiding interactions.

## Stage 3: optimization, supervision and training robustness

These change how the selected candidate learns, rather than adding large
inference components. Use the feature incumbent and matched controls:

- Latent-sign magnitude/initialization, STE variants and parameter-group learning
  rates (remaining Q16); data-aware initialization/row-orientation rescue (I04).
- Direct sign-flip optimizer, such as a Bop-style accumulated-gradient threshold
  rule, versus the existing latent-weight optimizer (Q24 optimizer portion).
  Measure flips, persistence and validation gains; flip count alone is not success.
- Oscillation damping, selective freezing and stability schedules (Q17).
- A8-to-A4 or protected-to-binary precision/coverage curricula (Q18).
- Target-distribution distillation (Q12), independently of the base hard-label
  token objective.
- Output/block reconstruction or intermediate-feature auxiliary losses (Q13).
- Verification-aware, prefix-survival or expected-progress objectives (Q14),
  independently of original DFlash position weighting already in Stage 0.
- Replay balancing, native serving-state coverage, hard-prefix/margin retention
  and candidate-dependent replay refresh (additional Q15 variants).

Initialization, optimizer and curriculum comparisons also need from-zero arms:
branching only from an adapted incumbent can conceal their initial-conversion
behavior. These are not blockers to the initial precision screen unless updates
are unhealthy. Reduced-state training optimizers (remaining Q24) must demonstrate
that changed numerics do not invalidate the comparison.

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
  reuse (I17), and drafter KV quantization (I18).
- W1A1 parallel drafting/tree/retry revisit: retained research lead, separate from
  this W1Ax precision campaign; historical acceptance/latency recollections are
  not a selected or validated modern configuration.

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

At every tier, branch a candidate and an unchanged continuation from the same
parent with matched data/exposure and training budget. Promote a separate moving
incumbent only with native evidence. Preserve frozen zero/initial references and
external DFlash Q4_K_M. No short-run result proves long-run convergence, and no
stage assignment means a feature is already validated or selected for deployment.

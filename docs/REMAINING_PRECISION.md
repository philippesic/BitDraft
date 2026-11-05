# Remaining component precision and recovery plan

## Objective and starting reference

November 5, 2026: the user requests an incremental plan for output head,
embeddings, drafter KV and scalar metadata/normalization gains. Extend
[PRECISION_SEARCH.md](PRECISION_SEARCH.md) on the qualified optimized G4/D8
candidate; do not restart the activation-family campaign. This is planning only,
not a new GPU/training allocation. The user has selected head reduction as the
next experiment; no experiment is reported as run by this document.

Freeze B0: exact learned-small2048 checkpoint, optimized qualified execution
runtime, W1 gate/up/down weights, A4 gate/up inputs, A8 down inputs, Q4_0
fusion/attention weights, shared Q6_K head/embedding, F16 draft KV and F32 scalar
metadata/norm gains. Preserve the original runtime and external DFlash Q4_K_M
control. Target head/embedding/KV, target precision and verifier stay immutable.
Use [CANDIDATE_BENCHMARKING.md](CANDIDATE_BENCHMARKING.md) for final comparisons.

The order is head, embedding/sharing, KV, scalar storage, then combinations.
Each component first gets an isolated comparison against B0. Probe a few exact
formats, not a Cartesian product. Unsupported native execution or missing QAT
simulation is an implementation blocker, never a quality failure.

## Common experiment and recovery rules

For every arm, pin the source, initializer, parent, quantizer, tensor coverage,
native format, runtime, data and trainable parameter groups before testing.

1. Qualify actual native admission, arithmetic and export/resume first. Declare
   the new precision reference: changed codes versus B0 may be legitimate here;
   native execution must match the declared NEW quantizer/storage semantics.
2. Run a zero-adaptation DEV screen for acceptance/prefix survival, native
   latency and storage/peak memory. Large initial loss does not reject an arm.
3. Train matched additional exposure from the same parent, with an unchanged
   continuation. Proposed readouts: 0, 256, 1,024 and 4,096 updates. Calibrate step
   cost, data coverage and total allocation in a pilot before launching. Update
   counts are initial screening choices, not equivalent convergence horizons.
4. Use the same examples/order and supervised exposures in paired arms, recording
   compute separately. Reuse a continuation only if all those conditions match.
   Adaptation resets and exact resume remain explicit different operations.
5. Track the residual gap to the equally trained control, not improvement over
   the untrained parent alone. Dense DEV loss, target-token ranks/margins and
   agreement diagnose recovery; periodic native prefix/latency/throughput tests
   decide usefulness. Inference loss must not be silently replaced by a proxy.
6. Give correctly implemented low-precision arms with credible latency OR memory
   potential at least the initial recovery window, even after severe initial
   damage. Reserve extension budget for at least two slow starters before ranking
   candidates. Extend on supported recovery, noisy acceptance with improving
   independent DEV loss, or a targeted recipe remedy; prioritize the strongest
   tradeoffs. A flat short run is "stalled under this recipe/budget," not proof
   of irrecoverability. Further large allocations remain explicit decisions.
7. Cheap readouts can use a small fixed consumed DEV cohort and be labeled smoke.
   Finalists use broader DEV and the standard alternating native protocol. Keep
   TRAIN separate; use representative/expanded native TRAIN coverage rather than
   count repeated overlapping labels as new data. Lock recipes before any new
   held-out confirmation. Reused prior held-out cohorts are now consumed evidence.

No universal acceptable percentage loss is imposed. A cheaper candidate can be
useful with some residual acceptance loss; judge actual end-to-end throughput
and separately retain memory-oriented alternatives. Use measured progress per
verification and complete round cost; do not estimate a speed ceiling by
subtracting asynchronous host spans. Timing and training uncertainty differ.

## Training eligibility: two explicitly different recovery tiers

**R1, compensation:** train the existing FFN signs/scales/midpoints/gains/norm
groups and the already qualified activation-range parameters. Keep quantized
head/embedding masters frozen initially. The reduced head/embedding/cache/scalar
behavior MUST participate in the training forward pass. This tests upstream
compensation, not full training of the new head or embedding itself.

**R2, targeted QAT:** if R1 stalls on a potentially useful arm, test a bounded,
separately labeled targeted recipe: quantizer/calibration parameters supported by
the exact deployed format, or trainable high-precision head/embedding masters with
that format simulated in the forward pass. Qualify gradients/export and memory
before choosing this option. The large vocabulary matrices make full master,
gradient and optimizer state expensive; do not assume they fit on the 5080.
If necessary a budgeted offload/chunked update recipe is a distinct training
cost, not free compensation. Unavailable R2 support leaves the arm open.

Preserve tying and trainable capacity across matched arms. Extra correction
layers, new distillation objectives and other architecture/features are not
silently added to rescue a precision arm. Record any later explicit recipe change.
F32 training masters/optimizer can remain even when deployed storage is lower;
they are not inference bytes. Simulate actual storage rounding during training.

## 1. Output head: first priority

Initial three-arm isolation:

| Arm | Head | Purpose |
| --- | --- | --- |
| H0 | Existing shared Q6_K | B0 reference |
| H6 | Private exact copy of original Q6_K codes | Cost of separating head from target |
| H4 | Private Q4_K from the same declared source matrix | Precision effect versus H6; deployed effect versus H0 |

H6 copies the original packed codes and scales without a dequantize/requantize
round trip. H4's source is initially the pinned dequantized Q6_K reference; use of
an original BF16 source is a separate initializer comparison if later available.
Preserve vocabulary/token IDs and original logits/sampling semantics. Confirm
Q4_K dispatch, rather than extrapolate from the earlier Q4_0 private-head test.

Measure the head region, clean forward, complete native draft/request cost and
private allocation. A private Q4 head adds storage because target Q6_K remains.
Do not label that as reduced combined deployment memory. Compare H4 and H6 under
matched recovery, retaining H0 as the actual sharing/performance control.

After the first H4 recovery readouts, use Q5_K as a bridge if needed and Q3_K as
the next compact probe. A supported Q2_K extreme probe is a reserve only when
lower precision offers a meaningful additional benefit. No obligation to finish
every head format before moving to the next component. Each format is a real
native path to qualify, not a claim of faster arithmetic from nominal bits.

## 2. Embeddings: qualify private storage and test tying

First implement/qualify a draft-only embedding path; target embeddings never
change. The private Q6_K embedding control must exactly reproduce B0's lookup,
including the mask and special tokens. Then compare private Q4_K against that
control and shared B0, keeping head unchanged for the isolated effect.

Evaluate common and rare tokens, special/mask tokens, varied domains and prompt
lengths. Do not silently make TRAIN from DEV token examples. Run R1 and reserve
R2 for trainable embeddings if needed; document sparse token coverage and tying.
Lookups, transfer and end-to-end timing all matter: a smaller matrix need not
make this low-work operation faster. Private storage may dominate any savings.

Next combine H4 with E4. Because the original head/embedding weights are tied,
qualify whether both draft operations can use ONE identically quantized matrix
and ONE device allocation. Report physical sharing versus duplicated GGUF/file
bytes. If training either matrix, preserve tying explicitly; do not introduce
two independently trainable copies unnoticed. Unsupported sharing stays blocked.

Only after useful Q4 recovery, or as a reserved slow-starter test, probe Q3_K.
Keep Q5_K available as a bridge. The initial target is a functioning, measured
Q4 lookup/sharing configuration, not an embedding quantizer sweep.

## 3. Drafter KV: isolate K and V, then lower together

Preserve F16 target caches. First qualify that the DFlash draft path actually
uses the requested cache format and retains its reuse/position/mask semantics;
CLI acceptance alone is insufficient. Charge cache scales, residual regions,
conversion and dequantization work; report actual allocated buffers.

| Round | Draft K / V arms |
| --- | --- |
| Reference | F16 / F16 |
| Q8 isolation | Q8_0 / F16; F16 / Q8_0; Q8_0 / Q8_0 |
| Lower precision | Q4_0 / Q8_0; Q8_0 / Q4_0; Q4_0 / Q4_0 |
| Conditional bridge | Qualified Q5_0 or Q5_1 on the sensitive side |
| Numerical control only | BF16 / BF16, if range/stability evidence motivates it |

These are a finite proposed queue; native/backend support must be verified. The
current cache-type list includes Q8/Q5/Q4/BF16 but not Q6_K; do not promise a
Q6 cache arm based on weight-format support. BF16 is still sixteen bits and
offers no storage reduction from F16.

Measure short, medium and near-limit contexts within the fixed 4,096-token
configuration and output reserve. Include sequential cache growth, reuse, draft
rollback/rejection and repeated verification, not only one static attention call.
Longer-context conclusions require compatible TRAIN/DEV coverage.

Recovery training must reproduce the deployed draft-cache quantization layout,
grouping, write/read lifecycle and history. Quantizing a convenient activation
tensor or unrelated teacher cache is not a substitute. If this simulation is
missing, mark the result inference-only with recovery unknown. Start R1 with a
qualified forward simulation; learned clipping/scales are R2 only if the native
format/export supports the same behavior. Keep sensible higher precision on one
of K/V or selected layers if that recovers most benefit; test restoration as a
separate configuration rather than declare all low-bit KV unsuitable.

## 4. Scalar storage and normalization gains: small, controlled tests

Distinguish stored scalar values from computation precision and from training
masters. Start with F16 storage, conversion to F32 after loading, and F32 math/reductions.
Test BF16 storage separately when F16 range/underflow is a concrete issue.
Do not lower normalization reduction, softmax, accumulation or residual precision
in this pass. An F16 norm gain does not imply F16 RMS computation.

Isolated groups, each against its F32 counterpart:

1. FFN output-row scales.
2. Row midpoints, independently from scales.
3. Positive channel gains and activation clipping constants, independently first.
4. Existing normalization gains, after the projection metadata tests.
5. The supported low-storage combination, with selective F32 restoration if needed.

Model the exported quantity: for example round positive `exp(log_gain)` and
`exp(log_clip)` to the deployment dtype, not the latent logarithm and then take
its exponential. Preserve sign/positivity and document zero/underflow/saturation.
Gradient simulation, export/import, actual native loads and storage must agree.
If the loader expands values to permanent F32 buffers, report that honestly as
file-size-only saving, not smaller runtime storage or bandwidth.

Give damaged arms a short matched R1 recovery check, but prioritize training
budget proportionally to the small potential benefit. Gains/midpoints can still
matter to quantized decisions even when their storage is small. Existing norm
parameters occupy about146KiB F32 and projection metadata about1.2MiB: no large
model-size claim follows. Do not conflate rounding-sensitive tiny midpoints or
gains with harmless norm-weight compression.

## 5. Combine, retrain fairly and preserve several useful candidates

After isolated component screens, construct the incremental chain:

`B0 -> head -> head+embedding -> +KV -> +scalar storage`

Also retain useful head+KV configurations if embedding storage/sharing offers no
benefit. At each addition compare an unchanged-parent continuation at the same
exposure, and backtrack individual layers/groups if required. Independent wins
do not establish a combined win; cache rounding, gain rounding and head changes
can interact. At most two deliberate extra interaction/restoration arms in the
initial combination round, with extension budget reserved before running.

Lock a small shortlist: a fastest supported candidate, a smaller balanced one,
and an extreme recovery lead if useful. Final precision-attribution comparisons
instantiate locked configurations from B0 with matched data/order/exposure and
trainable policy. Incrementally adapted histories are exploratory recipes until
matched confirmation; equal total tokens do not erase curriculum differences.

For every row publish the full precision/tie/parameter map, training history,
zero/intermediate/final quality, prefix survival, clean GPU forward, full draft
and request latency/throughput, actual storage/private/shared/cache bytes, peak
device and training memory, and exact native execution support. Use paired
uncertainty and retain unresolved near-ties. Final plots show recovery versus
exposure alongside acceptance/latency and total deployment memory tradeoffs.

Use statuses: unadapted diagnostic, recovering, viable at tested budget, stalled
under tested recipe, or implementation blocked. None means permanently
unrecoverable. Reference/control data, target and verifier stay fixed; final
held-out evidence is opened only after shortlist/recipe lock.

## Pilot allocation before execution

Define actual resource ceilings, representative data coverage, standard exposure
and total training budget in the first pilot. Count private-copy controls,
unchanged continuations, exports/evaluations, the reserved slow starters, R2
qualification and final combination confirmation. Do not consume the whole budget
screening heads before embeddings/KV receive their qualified recovery opportunity.
No large allocation or new GPU job is launched by this plan.

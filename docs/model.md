# HE4 model and training

HE4 is the current development candidate, a small drafter trained to predict the
next tokens of a fixed target model. Its purpose and historical results are in
[research context](research.md). Data and label identities are in [data](data.md);
state and execution behavior are in [execution](execution.md).

## What the CPU model computes

The implementation is `w1a1_eagle/model.py`. Its default geometry has six layers,
eight target-feature taps and at most sixteen draft slots. Geometry can be reduced for
CPU tests; a passing tiny test does not establish full-size behavior.

1. Captured target features are fused into the drafter's context. The feature
   contract and anchor determine which preceding positions may be attended to. Inputs
   declared fused already hold the fusion table's output and skip that product
   ([fused storage](data.md#fused-storage)).
2. One anchor token and k−1 mask tokens initialize a block of k draft slots (2 ≤ k ≤ 16).
3. Each layer applies grouped attention, rotary position encoding and a gated
   feed-forward block. Cached keys and values explicitly round through F16. Slots attend
   to every slot of their own block in both directions, so a slot's output depends on
   the block size.
4. The final normalized state is projected through the frozen vocabulary matrix.
   The worker trains on the k−1 prediction slots after the anchor.

The native runtime uses the same anchor-plus-mask layout and non-causal attention and
builds depth+1 slots per round (`artifacts/native/runtime-v7/source/common/speculative.cpp`):
at the benchmark depth 3 it proposes the first three predictions from a four-slot block. A block of k
slots in training is exactly that layout (tested against a sixteen-slot pass whose later
slots are hidden). The recipe chooses the training block sizes; the default is sixteen
slots for every update ([execution](execution.md#block-sizes-and-position-weights)). The
export keeps a block size of sixteen, the most a server may request.

The CPU implementation simulates quantized arithmetic with floating-point tensors.
It does not demonstrate binary hardware execution or native inference speed.

## What changes during training

The eighteen feed-forward projections have learned sign latents, nonnegative row
scales and row midpoints. Gate/up projections share learned input gains and clipping;
the down projection has its own gain and clipping. Normalization weights also train.
Attention and fusion operands and the vocabulary matrix remain frozen, unless the recipe
sets `trained_tables`; then the attention and fusion tables train in full precision and only
the vocabulary stays frozen ([trained tables](execution.md#trained-tables)).
The recipe can fix row scales or clipping bounds at their starting values and switch gains
(fixed at 1) or midpoints (fixed at 0) off; such parameters leave the optimizer
([feature switches](execution.md#feature-switches)).

The forward path uses hard binary signs and quantized activations. Backward uses a
surrogate derivative so the continuous latent parameters can receive gradients.
Sign zero is positive; the sign surrogate is active at both endpoints of [-1, 1].
Changing these boundaries changes the training recipe and requires explicit review.
A latent starts as the source weight (`raw`) or, with the bounded row scale, as
clamp(weight / (multiple × mean |weight| of its row), -1, 1), and is then clamped back
into [-1, 1] after every optimizer step. Row scales and midpoints start the same way under
both, so the initial signs and model outputs are identical; only the distance each sign
must travel before it flips changes. Latents outside [-1, 1] receive no gradient, so
without the clamp a latent that drifts past either end stops learning.
A custom autograd function computes the sign and saves only the latent itself, so the
surrogate mask is not stored for backward.

With [gradual precision](execution.md#gradual-precision) the first updates use soft weights
instead of hard signs: the latent holds the source weight W and a row's table is
S·tanh(t·W/S)/tanh(t) (Tu et al., arXiv 2508.06974), with S the row scale and t rising with the
update count. At small t the table equals the source table; as t grows it approaches S·sign(W).
Forward and backward follow this function exactly (no straight-through for the weights); inputs
stay rounded. A custom autograd function recomputes tanh in backward, so it stores no more than
the hard path. At the declared switch the latents take the hard recipe's form, signs preserved,
and train as above.

## One optimizer update

An update trains on a group of windows from one sequence, not on one window.
`HE4.group_hidden` fuses the context once and computes each layer's context keys and
values once. Each window attends only to context before its own anchor (with the same
local-window rule) and to its own block's slots, never another window's slots. All
windows of one update use the same block size.
`HE4.forward` remains the single-window path used by diagnostics and evaluation.

The loss is the mean over the group's windows of each window's position-weighted hard-label
cross-entropy over its first k−1 labels, including shorter terminal `up-to-15` windows.
Each window's weights are normalized to sum to one; they are either the fixed decay
exp(−i/7) for zero-based position i or the reach weights described in
[execution](execution.md#block-sizes-and-position-weights). Vocabulary logits are
computed and backpropagated in 128-row chunks, then the accumulated gradient passes
through the body once. Each window's logits match the single-window path to within the
model's rounding sensitivity: the activation quantizers turn last-bit differences between
equivalent kernels into occasional changed codes. On full-size weights that difference is
the same size as running the single-window path on CPU instead of GPU; see
[grouped updates](../results/grouped-updates/result.md).

The worker uses separate fused AdamW groups for sign, scale, midpoint, gain, clipping and
normalization parameters, plus a table group when tables train, without weight decay. Their step sizes, a step-size schedule and
the latent scale are recipe settings ([execution](execution.md#optimization)). The defaults,
1e-3 for signs and 1e-5 for the rest at a constant rate with raw latents, were chosen for
one window per update. A [four-arm test](../results/step-size-schedule-latent-scale/result.md)
favours cosine decay, and a 4e-3 sign step with the bounded latent scale. A
[block-size test](../results/block-size-and-reach-weights/result.md) favours 4-slot blocks
for depths 2–4 over sixteen slots and over random 3–16-slot blocks with reach weights. A nonfinite loss or
gradient fails the update immediately. The all-parameters-finite scan runs every 32
updates and before every checkpoint and export. Frozen operands are fingerprinted and
checked before publication. Trained tables are exported as Q4_0, so the DEV diagnostic
also measures the model with the exported table values.

## What has been established

CPU tests cover scalar projection and sign witnesses, a retained arithmetic
characterization, small six-layer forward/update behavior, exact own-state resume,
branch behavior and export/readback. [Arithmetic evidence](../evidence/arithmetic-reference.json)
and [source inventory](../evidence/source-inventory.json) record their ancestry.

Actual retained-artifact equivalence, native rounding/loading, useful learning and
end-to-end serving performance require separate evidence. An old campaign's GPU
qualification does not qualify this implementation.

Attention-cache values round through FP16 with a straight-through gradient. A direct
FP16 cast has a different derivative and failed the original-checkpoint next-update
comparison; see [migration evidence](../results/he4-checkpoint-migration/result.md).

## Frozen table storage

The optional intent setting `frozen_format` defaults to `Q4_0`. Omission and explicit
`Q4_0` preserve existing recipe identities and frozen operands. `baseline-k` requires
`frozen_baseline`, an authenticated artifact reference (`path`, `sha256`, `bytes`).
It copies attention and fusion table bytes from that file, preserving each Q4_K or
Q6_K type, and uses their dequantized values in the model. The pinned GGUF package
cannot quantize Q4_K; authenticated copying needs no additional dependency.

New candidate recipes set `baseline-k` with the Baseline artifact: it costs no measured
acceptance and removes the per-slot table reads of `Q4_0`
([test list](../tasks/speedup-test-list.md), item 8). `Q4_0` remains the meaning of an
omitted setting so that existing states resume unchanged.

The Baseline identity enters the new recipe. Export preserves its codes and checks
that they decode exactly to the frozen operands, then verifies GGUF payload readback.
A changed frozen fingerprint is accepted only in branch mode and only when the parent
fingerprint matches the independently rebuilt Q4_0 model from the same inputs. The
checkpoint records parent state, old/new fingerprints, format and Baseline identity.
Branch resets the optimizer; exact resume continues to require matching operands.

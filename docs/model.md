# HE4 model and training

HE4 is the current development candidate, a small drafter trained to predict the
next tokens of a fixed target model. Its purpose and historical results are in
[research context](research.md). Data and label identities are in [data](data.md);
state and execution behavior are in [execution](execution.md).

## What the CPU model computes

The implementation is `w1a1_eagle/model.py`. Its default geometry has six layers,
eight target-feature taps and sixteen draft slots. Geometry can be reduced for
CPU tests; a passing tiny test does not establish full-size behavior.

1. Captured target features are fused into the drafter's context. The feature
   contract and anchor determine which preceding positions may be attended to.
2. One anchor token and fifteen mask tokens initialize the sixteen draft slots.
3. Each layer applies grouped attention, rotary position encoding and a gated
   feed-forward block. Cached keys and values explicitly round through F16.
4. The final normalized state is projected through the frozen vocabulary matrix.
   The worker trains on the fifteen prediction slots after the anchor.

The CPU implementation simulates quantized arithmetic with floating-point tensors.
It does not demonstrate binary hardware execution or native inference speed.

## What changes during training

The eighteen feed-forward projections have learned sign latents, nonnegative row
scales and row midpoints. Gate/up projections share learned input gains and clipping;
the down projection has its own gain and clipping. Normalization weights also train.
Attention and fusion operands and the vocabulary matrix remain frozen.

The forward path uses hard binary signs and quantized activations. Backward uses a
surrogate derivative so the continuous latent parameters can receive gradients.
Sign zero is positive; the sign surrogate is active at both endpoints of [-1, 1].
Changing these boundaries changes the training recipe and requires explicit review.

The worker uses separate AdamW groups for sign, scale, midpoint, gain, clipping and
normalization parameters. It computes hard-label cross-entropy with decreasing weights
across the fifteen positions. Current grouping and rates live in
`w1a1_eagle/numerical/worker.py`; they are implementation settings, not evidence of an
optimal recipe. Frozen operands are fingerprinted and checked before publication.

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

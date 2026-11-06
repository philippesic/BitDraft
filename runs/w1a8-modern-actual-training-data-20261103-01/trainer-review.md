# Independent bounded trainer review

Reviewed the new trainer and its immediate data/actor/adapter APIs, without
opening weights, loading a model, using CUDA, or changing source. No blocking
objective, tensor-lineage, fit/development split, or matched-arm error found for
the planned reference -> 480-update arms -> 1920-update extensions. This is a
source review, not an executed CUDA forward/backward or native qualification.

Reviewed trainer:
`/home/philip/.codex/worktrees/w1a8-actual-training-20261103/scripts/w1a8_generalization/modern_train.py`

SHA256: `52e97f5d34cee487eaa3d8b077d7655af27d01785c835925a2eb975c6251b547`.
Python AST parsing passed. Immediate dependencies reviewed were
`modern_actual_data.py`, `modern_actor.py`, and `native_step.py`.

## Launch requirements

1. Use `final01/manifest.json`: it contains all 2138 windows and no missing
   native embeddings. The earlier top-level manifest still has missing embedding
   IDs and is correctly refused by the trainer.
2. The launcher must provide `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or the supported
   `:16:8`) before CUDA work. The trainer enables deterministic algorithms but
   does not set this environment variable itself; CUDA matrix multiplication can
   otherwise fail on this deterministic setting. This is a launch condition,
   not evidence that the actual launcher omits it.

## Objective and readout checks

`ActualDataset.materialize()` returns exactly the named fields used by
`forward()`. Prefix token/raw-feature rows have anchor length; the detached
current-actor context cache consumes the observed raw-feature producers, not an
assumed equal target/cache position. The seed is encoded once, the decoder uses
anchor+d, the next recurrent state is the actor's own pre-norm output, and the
next input token is the gold label from the previous step.

CE is included only for supported draft-vocabulary labels with the native
gold-prefix reach mask. A native first-step failure therefore excludes its
second-step CE; the first step remains usable. The actor unroll is teacher
forced, so a newly wrong actor first step does not by itself remove the second
step from that native-reachable objective. This matches the declared objective.
The fixed denominator is two windows times depth two; short/unsupported rows
are retained as zero CE contribution rather than silently resampling them.

The reference computes same-GPU actor logits/margins, uses absolute d2t for
token correctness, and intersects the actor's contiguous correct prefix with
the native emitted-success prefix. Unsupported labels cannot enter that
intersection through a correct actor prediction. The hinge uses the saved
same-actor margin, capped at two, rather than native-versus-Torch logit margins.
The final zero-update reference must be inspected to report the actual
intersection and native successes outside it. This objective alone does not
preserve every native success; those outside the intersection need separate
tracking/native acceptance checks.

Reference and post-training readouts preserve proposals, labels, support,
margins, and prefix correctness per window. Post-training reads all development
windows and a deterministic one-in-eight fit subset. These are actor readouts;
neither readout establishes total serving throughput or untouched answer quality.

## Matched controls and leakage

Both arms rebuild the same pinned initializer, enable exactly nine scale-offset
optimizer tensors, use Adam with identical settings, seed 37, fixed order, two
fit windows per update, and the same fixed normalization. Retention zero still
computes the same retention expression but multiplies it by zero. Reference
contract checks match manifest, actor semantics, sources, optimizer and precision
while intentionally allowing the two retention coefficients to differ.

Metadata inspection of `final01/manifest.json` found 1414 fit windows and 724
development windows. No group ID occurs in both roles. Training samples only
the fit list. Computing development reference logits does not update the model;
development-driven candidate selection still requires a later untouched set.
All label/mask lengths match max_depth. There are 356 fit and 193 development
native protected tokens before the same-actor intersection.

## Resume and stop behavior

Checkpoints contain scale state, Adam state, cursor, trainer time and CPU/CUDA
RNG states. Resume checks the full contract, restores those values, and selects
the next windows from the deterministic cursor. The planned 480 -> 1920
extension follows the same update order. Atomic checkpoint writes are roundtrip
checked; regular checkpoints occur every 480 updates, with an additional save
on exceptions/stop detected before an update. Signal handlers request stop at a
guard boundary rather than interrupting an arbitrary CUDA operation.

One bounded edge bug is outside the planned extension: resuming into a fresh
output directory with `saved cursor == --steps` at a multiple of 480 skips the
loop and then reads that new directory's nonexistent `latest.json`. Thus an
already-complete no-op resume fails. Do not use that invocation as a completed
resume/readout path. Normal 480 -> 1920 extension is unaffected.

No canonical code, raw traces, checkpoints, models or runtime pins were changed
by this review. The original bridge protocol remains the source of these views;
the independently repaired runtime is a separate control and cannot silently
replace their ancestry.

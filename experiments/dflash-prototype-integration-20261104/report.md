# Integration checkpoint — incomplete

Changes in assigned worktree only. Previous arithmetic foundation cherry-picked.
No GPU jobs, baseline mutations, source builds, pushes or gitlink changes.

Corrected RNE, original latent magnitudes, reciprocal/multiply dynamic contract,
zero/subnormal handling and exact fixed clipping values. Added explicit model
installer with gate/up shared gains, protected source BF16 tensor requirement,
feature-driven optimizer groups, file pin authentication, and strict checkpoint
config matching with optimizer/RNG/progress restoration.

Six test functions pass using canonical .venv Python, directly invoked with runpy
(pytest is unavailable). Includes independent tie/saturation expectations, exact
original A8 codes/output checks, six-layer fixture sharing/deduplication, and
exact resumed next update. This is small-fixture evidence, not full DFlash proof.

Outstanding assignment: training CLI integration; explicit parent adaptation;
packed schema/export/import; native loader/operations; isolated CUDA build;
manifest benchmark runner/count aggregation; real-model compatibility; GPU
qualification after explicit owner release. No claim that GPU ownership is the
only blocker. Existing native_admit deliberately continues to reject all configs.

Additional limitations: authentication helper is not invoked by resolve; optimizer
schedule is constant only; protected projection execution uses source-dequantized
F32 linear and is not qualified against native BF16 input rounding. Dynamic STE
has not been demonstrated gradient-equivalent to baseline. No native ready-to-run
qualification command exists yet because required implementation is missing.

## Follow-up arithmetic correction

Recovered both prior commits into the assigned checkout. Dynamic quantization now
uses detached integer forward codes and a dequantized identity surrogate, avoiding
zero-row gradient loss and division by subnormal steps. Sign gradients use the
original clipped hard_sign_ste. Seven CPU functions pass, including direct
BlockBinaryLinear input/sign/scale gradients and AdamW updates for zero,
subnormal and ordinary rows. Scale updates agree within F32 rounding tolerance;
input and parameter gradients agree exactly in these fixtures.

Installer now preserves prototype scales and enabled compensation on subsequent
conversion and uses the baseline NativeBF16Linear for protected projections.
An explicit authenticated adaptation entry records parent kind/config and resets
optimizer state. This adaptation entry is not yet qualified on a real model;
lineage persistence and shared-gain mismatch validation remain unfinished.

No native build or GPU work occurred. The requested runnable pipeline remains
incomplete: training runner, compact recoverable checkpoints, packed export/import,
native runtime/build, and benchmark aggregation are still required. This is not
an integration-ready handoff or evidence of native compatibility.

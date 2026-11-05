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

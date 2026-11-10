# Runtime and verify requirements

Foundation has no runtime, native build recipe, installed model or real verify command.
Runtime04 reproduction needs full source, ordered patches, compiler/ABI, loaded libraries
and build closure: old receipts link objects across ignored execution trees. The submodule
gitlink or an old binary alone is insufficient. Do not copy or execute those trees.

Historical receipts report Python 3.11.15, torch 2.14.0 / CUDA build 13.0, NumPy 2.4.6,
zstandard 0.25.0, GCC 15.2.0, nvcc 13.1.115, Nsight Systems 2025.5.2. Observed transformers
4.57.6 differs from declared 4.57.1. The small check environment excludes numerical packages. Torch/NumPy/GGUF are now
separately pinned numerical extras in the lock; the remaining historical runtime
packages and native toolchain have not been reproduced.

Future smallest real verify: two synthetic non-heldout prompts, at most 32 native input
tokens each, greedy cap 16, context 256, concurrency 1, depth 2, matched target/tokenizer/
template/runtime/seed/F16 KV. Sequential target-only and actual speculative generation
must match every token ID, text and finish reason. Require finite logits, intended loader
activation and nonzero proposals; low acceptance alone is not failure. Capture at most
16 generated positions on one prompt and verify all eight taps, next-token/EOG semantics,
full-logit argmax and honest window coverage. Proposed output cap is 128 MiB and timeout
five GPU minutes; no download/build. A separately declared long synthetic fixture covers
early-EOG zero-window cases. These are pending specifications, not performed GPU work.

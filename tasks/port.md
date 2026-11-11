# Port the HE4 path from the predecessor repository

## Goal

One auditable HE4 data, training, export, runtime and benchmark path in this repository,
replacing the per-campaign scripts of `/home/philip/binary-eagle-decoding`.

## Authorized

The source port through end-to-end CPU verification, including a lightweight fresh-agent
exercise. Not authorized: benchmark redesign, high-level agent operations, GPU work,
training runs, data transfer, cleanup of old artifacts, remotes or publication. Each
later stage below starts only on Philip's go.

## Done when

- Source: retained-artifact parity for latents, scales, masks, parameter groups, forward,
  loss, gradient and update; exact sign, bit order, packing and GGUF metadata. CPU
  equivalence does not certify native rounding.
- Runtime: a reproduced runtime04 build closure and a real tiny verify on the RTX 5080.
- Benchmark: one public entry with plan, run, verify, report and qualify modes that
  enforces [comparison rules](../docs/domain.md#comparisons) and selection.
- GPU qualification: hash-matched check and verify receipts, keep-best exercised on
  hardware, bounded disk and GPU cost, all failures retained.

## State

Done on CPU: metadata admission and the modern index (see FINDINGS F011), HE4 numerics,
transactional states, exact resume and branch, bounded supervision, retained source and
intent per attempt, verified GGUF export. Source inventory: [evidence](../evidence/source-inventory.json).
Agent exercises so far: [readiness](../evidence/agent-readiness.json),
[routine repair](../evidence/agent-window-exercise.json),
[failure diagnosis](../evidence/agent-diagnosis-exercise.json).

Not done: retained-artifact parity, native runtime, benchmark, selection, GPU
qualification. Runtime04 links objects across ignored run trees in the old repository;
a missing source closure blocks the runtime stage and is not a reason to reuse old
binaries. Historical receipts list Python 3.11.15, torch 2.14.0 with CUDA 13.0,
NumPy 2.4.6, zstandard 0.25.0, GCC 15.2.0, nvcc 13.1.115 and Nsight Systems 2025.5.2;
transformers was observed at 4.57.6 against a declared 4.57.1.

## Next

Retained-artifact parity for the source stage, once Philip confirms the stage is still
the priority.

## Proposed specifications

Smallest real verify: two synthetic non-heldout prompts of at most 32 input tokens,
greedy cap 16, context 256, concurrency 1, depth 2, matched target, tokenizer, template,
runtime, seed and F16 KV. Target-only and speculative generation must match every token
ID, text and finish reason, with finite logits, confirmed loader activation and nonzero
proposals. Capture at most 16 positions on one prompt and verify all eight taps,
next-token and end-of-generation semantics, argmax labels and window coverage. Proposed
limits: 128 MiB output, five GPU minutes, no download or build.

Benchmark workloads from the original plan: native smoke, capture correctness, DEV
diagnostic, formal runtime04 panel (consumed DEV3, depth 2, context 4096, greedy, cap
256, two excluded warmups, five alternating process repetitions), larger serving
selection, heldout confirmation, stress and recovery. These are replication references,
not approved requirements.

## Open questions for Philip

1. Permanent name, remote, license and publication visibility.
2. Initial support breadth. Proposed: HE4, target-only and Baseline on Qwen3.5-4B
   Q4_K_M, runtime04-equivalent, RTX 5080; other routes stay historical.
3. Recapture acceptance: explain the eight-prompt difference in F005 before admission.
4. Training budget and exposure. Proposed guards: at least 1 million admitted unique
   positions, at most 2 window epochs, mean and max exposure at most 30.
5. DEV, serving and stopping. Proposed: DEV60 every 30 update-minutes, formal serving at
   frozen mid-run points, stop after three checks with 5% relative cross-entropy
   deterioration and no top-1 improvement.
6. Deployment margin and an untouched confirmation source; heldout24 is already consumed.
7. How much conversion of historical state schemas to implement.

# Current project status

## Local RTX 5080 investigation — November 1, 2026

This session runs directly in WSL on the RTX 5080 machine. The sole project root
is `/home/philip/binary-eagle-decoding`. Use the local shell and GPU; no remote
access route, machine linking, host registration or separate device checkout is
required. Follow [local execution](AGENT_OPERATIONS.md#local-gpu-execution).

Actual local checks confirmed an NVIDIA GeForce RTX 5080 with 16,303 MiB VRAM,
driver 617.42, 0% utilization, and no listed CUDA compute applications. These
are a snapshot, not proof of future availability or every graphics allocation.
No existing project trainer/supervisor was found in the process census. No GPU
training/evaluation was launched by this reconciliation. Raw identity/process
checks and original files are preserved under ignored
`runs/local-execution-correction-20261101/`.

The user resumed two objectives:

1. Verify the narrower upstream CUDA consistency changes from llama.cpp issue
   25618 for Qwen3.5 4B/9B Q4_K_M and NVFP4; isolate appropriate changes and
   determine correctness and full-request performance against matched controls.
2. Diagnose why training and native evaluation gains have not improved serving
   throughput, following evidence through numerical behavior, teacher/data
   alignment, real verifier histories, recurrent state, acceptance and costs.

Keep the modern quantized-target track separate from historical F16-target
training. Preserve checkpoints and actual accounting; do not restart or shorten
historical allocations. Q4_0 EAGLE remains the primary EAGLE comparison, with
appropriate same-family/matched controls for other drafter configurations and
an unchanged target-only reference within each comparison.

Require token-level correctness, independent answer quality, repeated matched
full-request measurements and separate final held-out confirmation. Revalidate
after runtime changes. There is no user-imposed GPU-hour or candidate-count
limit. Report actual training, evaluation and preparation costs separately.

An isolated upstream source investigation was dispatched; its output still needs
review. GPU experiment resumption remains unfinished. The agent's initial claim
that missing remote access blocked work was incorrect and is withdrawn.

The existing [goal record](goals/nine-model-qat-overnight.md#local-execution-correction--november-1)
retains continuity and the current correction. [Archived status](STATUS_HISTORY.md)
preserves earlier evidence; historical jobs, addresses and checkout locators
must not be interpreted as current state or active instructions.

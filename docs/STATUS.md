# Current project status

## Local RTX 5080 investigation — November 1, 2026

This session runs directly in WSL on the RTX 5080 machine. The sole project root
is `/home/philip/binary-eagle-decoding`. Use the local shell and GPU; no remote
access route, machine linking, host registration or separate device checkout is
required. Follow [local execution](AGENT_OPERATIONS.md#local-gpu-execution).

Actual local checks confirmed an NVIDIA GeForce RTX 5080 with 16,303 MiB VRAM,
driver 617.42, 0% utilization, and no listed CUDA compute applications. These
are a snapshot, not proof of future availability or every graphics allocation.
No existing project trainer/supervisor was found in the initial process census.
Raw identity/process checks and original files are preserved under ignored
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

The source/evidence investigations are integrated. The isolated grouped-column
CUDA candidate passes 5,496 actual numerical/state checks with zero measured
drift, and both development serving screens match all output IDs. Development
throughput improves; final confirmation is pending. See the
[measured findings and patch](../experiments/local-resume-20261101/runtime-and-transfer.md).

The historical F16 diagnostic identifies losses on previously successful serving
frontiers, despite gains on difficult ones. A 96-state identical-input native
counterfactual has completed; analysis is in progress. No new training or historical restart occurred.
Final source/settings are frozen and fresh confirmation inputs are generated;
final model requests are running under the frozen plan. Current job/stop details are at the
end of the goal file.

The initial missing-remote-access blocker was incorrect and is withdrawn.
Declining the unnecessary link card does not pause research.

The existing [goal record](goals/nine-model-qat-overnight.md#local-execution-correction--november-1)
retains continuity and the current correction. [Archived status](STATUS_HISTORY.md)
preserves earlier evidence; historical jobs, addresses and checkout locators
must not be interpreted as current state or active instructions.
Current execution: final confirmation started on November 1 at 09:54 UTC under
`runs/local-confirmation-20261101-01/state.json`, local tmux socket
`binary-eagle-runtime`, same-named session. Root remains the sole GPU operator.
Exact boot/process birth identities and the authenticated stop procedure are in
`runs/local-confirmation-20261101-01/identity.json`.

The frozen final plan runs 288 independent quality requests followed by 1,920
matched full requests: four targets, four runtime/serving cells, 24 fresh prompts,
five alternating repetitions. No source/settings changes or other GPU work are
allowed during confirmation. No final result is claimed while the run is active.

# Current project status

## Follow-up: clean-engine qualification — November 1, 20:04 UTC

The user rejected developing around the broad CUDA workaround. Root tested the
existing F16 path with target width-mask overrides OFF. Both small development
screens passed, but fresh frozen confirmation FAILED: 2 of 24 new prompts differ
between Q4_0 EAGLE and target-only in all five repetitions (10 paired failures).
All 240 performance requests and 48 quality requests executed; no training.
Quality outputs match but raw scores0/24 are invalid as admission evidence because
the older API returns empty thinking tags in answer content. These failed gates
are preserved in `runs/clean-f16-qualification-20261101/`. No baseline migration.

Root is now qualifying ONE ordinary upstream-engine candidate, vLLM0.32.0 with
retained dense Qwen3-4B BF16 and ordinary EAGLE3 HF weights. Separate environment:
`runs/clean-engine-qualification-20261101/venv`. Package preparation only at this
checkpoint; no new engine GPU process yet. Existing training environment and
models remain untouched. No source patches/custom binary port are authorized
by this qualification. A candidate requiring engine source patches is rejected.
Only a verified normal-EAGLE foundation could justify a later binary port.
Root owns GPU scheduling; all native F16 test supervisors are terminal.

## Local RTX 5080 investigation completed — November 1, 2026

The requested runtime and training-transfer investigation reached its bounded
end goal. The isolated CUDA candidate improves Qwen3.5 MTP full-request
throughput by 32.0–51.0% over the conservative workaround across 4B/9B Q4_K_M
and NVFP4. Five alternating repetitions of 24 fresh prompts per cell produced
1,920 exact token/text/stop matches, with unchanged acceptance counters. All
candidate quality outputs match the conservative control (47/96 passing cases);
absolute quality remains modest. This is opt-in runtime evidence, not a
W1A1/Q4_0 EAGLE win or output equivalence to the unmodified default backend.

The upstream quantized-KV fix does not replace our F16-KV path, and the GB10
warp override does not act on RTX 5080. The validated candidate instead combines
restored two-column vector attention with grouped quantized projections while
retaining the conservative floating paths, fusion disable and one-partition
attention. Changes remain isolated and reviewable; parent gitlink unchanged.

Historical F16-target work stays separate. Improved difficult-state scores hide
losses on previously successful continuations and extra verifier rounds. A
96-state identical-input native replay retains most of those losses, supporting
a model tradeoff rather than history differences alone. Replay fidelity is
bounded. The justified next training experiment balances actual serving-history
coverage, successful-prefix retention and acceptance/depth weighting at matched
exposure, with original-group splits and fresh full-request confirmation.
No new training, restart or shortened historical allocation occurred.

[Full findings and patch](../experiments/local-resume-20261101/runtime-and-transfer.md),
[recomputable confirmation](../experiments/local-resume-20261101/confirmation-summary.json),
[cost ledger](../experiments/local-resume-20261101/cost-summary.json), and the
[latest goal checkpoint](goals/nine-model-qat-overnight.md#final-confirmation-and-release--november-1-2026)
retain details. The broader nine-model history is preserved; this checkpoint
does not declare every historical campaign complete.

Final confirmation ended at 10:53 UTC with exit zero. Root rechecked the
recorded boot/birth identities and absent process groups; all owned supervisors
are terminal and no CUDA compute application is listed. The RTX 5080 is idle
apart from Windows graphics allocations. No worker owns a GPU job. Current
release and retention evidence are ignored files under
`runs/upstream-consistency-20261101/{release-proof,retained-inventory}.json`.
About 12.1 GB of useful captures, arrays, source and build cache is retained;
paid checkpoints and pre-existing dirty/shared files are protected. No push.

This project runs directly in WSL on the RTX 5080 machine from
`/home/philip/binary-eagle-decoding`. Use the local shell and GPU; no SSH route,
host registry, machine link or device checkout is required. Follow
[local execution](AGENT_OPERATIONS.md#local-gpu-execution). The earlier access
blocker was incorrect; declining the unnecessary link card did not pause work.
[Archived status](STATUS_HISTORY.md) preserves earlier evidence as history,
not current jobs or launch instructions. Q4_0 EAGLE remains the primary EAGLE
comparison baseline. New runtime changes require revalidation.

# Current project status

## Active: Qwen3.5 correctness-mode baseline matrix — November 1, 2026

The user selected the existing Qwen3.5 correctness runtime and requested fresh
4B/9B × Q4_K_M/NVFP4 measurements. vLLM and Llama migrations are not selected.
Root is sole local RTX5080 operator. Actual supervised development-depth screen
began November2 00:19UTC (November1 local), run
`runs/local-qwen35-matrix-screen-20261101-01/state.json`; birth/boot/stop details
are in its identity.json. Local tmux socket binary-eagle-runtime, same session
name. GPU control is resumed; pilot release and empty compute-app list were
independently checked before this launch.

Canonical artifacts: `runs/qwen35-correctness-matrix-20261101-01/`. The target
runtime is the immutable previously validated quant4group CUDA overlay with
GGML_CUDA_DECODE_CONSISTENT=32, unchanged target weights, F16 target/draft KV,
context4096, batch/ubatch512, concurrency1, greedy generation and no prompt cache.
58 baseline groups: per4Btarget16, per9Btarget13. Target-only, embeddedMTP,
EAGLE3(BF16/Q4_0/Q4_K_M/Q8_0), DFlash(BF16/Q4_K_M/Q8_0),4BDSpark(sameformats),
and standalone0.8B(BF16/Q4_K_M/Q8_0/NVFP4). No matched9BDSpark identified;
NVFP4target does not imply an NVFP4sidecar. Q4_0EAGLE remains primary comparison;
the practical hurdle is the fastest correct deployable baseline pertarget.

Admission audit found two concrete modern EAGLE export defects: fullheadRoPE
instead of partial.25, and output-layer IDs interpreted as input-layer IDs.
Preserved originalexports and created separate derivatives with rotary40/64,
Q/Kpermutation restricted to the rotated slice, and input taps[2,16,29]. CPU
real-weight Q/K+RoPE checks, exact untouched-weight/header checks and a rejected
metadata-only negativecontrol pass. This changes drafter artifacts, not target
arithmetic; full native graph parity is not yet established. Q4_0/Q4_K_M/Q8_0
were exported from corrected BF16. Historical Qwen3 training is unchanged.

Pilot finished exit0 at00:16UTC. All four target serial/batched/state gates pass
1374 checks each with zero measured drift/flips; the executed original negative
control fails687 checks with53 flips. All16 pilot configurations pass exact
token/text/stop checks on8 performance+8 quality cases each (256 requests total).
Independent root recomputation agrees; shared target quality scores3–5/8 are
unchanged by speculation. This is development admission, not final ranking or
full native-versus-training graph parity. Raw/analysis: pilot-01/ and
root-pilot-analysis.json under the canonical run.

The166-configuration screen is ACTUALLY running with draft lengths varied on
reused development inputs, one repetition,8 performance+8 quality cases percell.
Depth selection/refinement remains; fresh final input/settings freeze and at
least five matched repetitions follow. No new training or throughput winner
claim. Source commits1742aa7e/7d09acb1; no push.

## Follow-up: upstream foundation qualified — November 1, 2026

The user requests a clean solution without a custom numerical workaround.
The old native F16 alternative FAILED its fresh test (two new prompts differ in
all five repetitions), despite passing the small development screen. No training
or baseline migration was started; evidence is preserved in
`experiments/clean-baseline-20261101/f16-rejection.json`.

Unmodified upstream vLLM0.32.0, retained Qwen3-4B BF16 and ordinary AngelSlim
EAGLE3 passed a 24-prompt development gate with positive draft acceptance and
exact token/text/stop equality under its supported batch-invariant mode.
Separate environment `runs/clean-engine-qualification-20261101/venv`; original
training environment untouched. No engine source patch or custom binary port.
Two failed preparation attempts (client API return type, long IPC path) are
preserved; fixed using normal API and documented RPC path setting.

Final confirmation completed all 20 cells, 480 performance and 96 quality
requests: five alternating repetitions with source/settings frozen before fresh
inputs. Invariant EAGLE3 equals target-only token/text/stop on 120/120 paired
performance requests and 24/24 quality cases; every arm repeats identically.
Both invariant arms score 14/24 on the frozen independent judge, with the same
ten failures. All 24 correct and 24 wrong controls behave correctly. EAGLE
acceptance is real: 7,965/22,155 draft tokens. Default mode fails parity on 14/24
prompts in every repetition.

Invariant EAGLE averages 19.951s versus 32.092s target-only per 24-request panel:
1.609× observed full LLM.generate speed, 37.8% lower time. Mode cost versus
default is 4.01% target-only and 11.78% EAGLE; cross-mode outputs differ.
Repetition CVs are 0.25–0.38%. Inference JIT warnings are not fully phase-localized:
these are observed full API wall times, not pure-GPU or fully warmed steady-state
costs, HTTP capacity, concurrency, a binary result or a Q4_0 win.

Recommend this pinned ordinary upstream foundation if dense Qwen3-4B BF16 and
an engine change are acceptable. No migration, new teacher, training or binary
port is adopted. Same-engine primary Q4_0 and custom binary execution remain
required before a project-level win. The user owns that major research choice;
see the pending decision in DECISIONS and
[full evidence](../experiments/clean-baseline-20261101/report.md).

Final phase wall 1117.33s: 503.74 performance, 53.24 quality, 368.02 initialization,
44.71 excluded warmups; remaining time is integrity/subprocess/harness work.
Development gate 177.30s; aggregate preparation clock unavailable; training zero.
Supervisor `runs/local-vllm-final-20261101-01/state.json` finished exit 0 at 20:37UTC.
Root rechecked same boot, absent process groups and no GPU compute apps. Removed
21 owned RPC scratch directories; reusable environment/evidence retained. Storage
and release receipt: `runs/clean-engine-qualification-20261101/release-and-storage.json`.

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

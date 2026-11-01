# Clean verifier foundation qualification

No clean replacement is adopted yet. The user rejected developing around the
broad local CUDA correctness workaround. This follow-up qualifies ordinary
target computation and an unmodified upstream engine before further training.
It does not start a binary port, migrate baselines or relabel historical runs.

## Native F16 alternative: rejected

The historical Qwen3-4B F16 setup also enabled target width-mask policies:
MMVF routing and serial attention geometry. Mask zero disables both; global
fusion disabling was separately unset. The retained runtime still contains
project-specific drafter integration, so this is not an upstream-stock claim.

Two exploratory gates passed on twelve reused development prompts and two
repetitions: ordinary target-only versus DSpark Q4_K_M, then Q4_0 EAGLE.
Each executed 48 requests. Ordinary mask-zero target-only also matched the old
masked reference on the checked 24 requests.

That success did not generalize. Source/settings were frozen before generating
24 new prompts. Five alternating target-only/EAGLE repetitions executed 240
requests and failed on two prompts in every repetition: ten paired failures.
See [the retained rejection summary](f16-rejection.json). The configuration is
not admitted as a lossless baseline.

The separate 48 quality requests match across arms, but the frozen scorer
reports 0/24 in each. The older API puts an empty thinking block into raw answer
content under `reasoning_format=none`; this rejects otherwise-correct answers,
alongside genuine task errors. Scores are preserved and are not quality
admission. Supported `deepseek` response parsing would separate reasoning from
answer for these ordinary Qwen3 requests, but no post-hoc scorer correction or
new quality pass was substituted: numerical parity already fails.

Native follow-up training cost is zero. Preliminary process totals are in
`runs/clean-f16-qualification-20261101/{attempt-01,eagle-attempt-01}/summary.json`.
Fresh confirmation costs: 285.42 seconds performance calls, 32.90 seconds quality
calls, 21.05 seconds startup and 24.17 seconds excluded warmups. Phase wall is
367.37 seconds, with integrity/harness overhead included. These are not trainer
or CUDA-kernel clocks.

## Unmodified upstream-engine candidate

An isolated environment installs PyPI vLLM 0.32.0, Torch 2.13.0 and Transformers
5.19.0. The existing training environment is untouched. Retained standard HF
Qwen3-4B BF16 and ordinary AngelSlim EAGLE3 weights are reused; there is no model
download, private engine patch or custom binary drafter port.

Installed upstream source contains explicit SM120/Qwen3-4B batch-invariant
matrix configurations, an ordinary EAGLE3 loader for this checkpoint layout,
and a compatible Triton attention backend. Source support is not by itself a
runtime correctness guarantee.

The development gate uses documented settings: `VLLM_BATCH_INVARIANT=1`, BF16,
TRITON_ATTN, one GPU/sequence, context 2048, no prefix caching and three EAGLE3
tokens with greedy drafting and standard rejection. All 24 development outputs
match target-only token IDs, text and stop state. Draft acceptance is positive;
this is not a no-speculation fallback. The prompts were borrowed from the failed
native confirmation and are development inputs for this new engine.

Two failed preparations are retained. Gate 01 used the old Transformers return
type assumption; the standard `return_dict=False` API fixed the client. Gate 02
exceeded Unix IPC socket path length under the supervisor's long temporary path;
the documented `VLLM_RPC_BASE_PATH` setting provides an owned short directory.
Neither repair changes engine computation or package source.

Evidence is under `runs/clean-engine-qualification-20261101/`: exact package
versions, PyPI metadata, installation logs, gate commands, output IDs and metrics.

## Fresh final confirmation is running

Source, settings and target/draft file hashes were frozen before generating
another 24 performance prompts and 24 independent quality cases. All 24 correct
and 24 known-wrong judge controls were executed. No tuning on final outputs.

The final four cells are stock/invariant mode crossed with target-only/EAGLE3.
Five alternating repetitions give 480 performance requests, with 96 quality
requests in repetition zero. The public `LLM.generate` call wall includes
scheduling, prompt processing and decoding; it is not HTTP serving capacity.
Model loading, first-use JIT and warmups must be distinguished. Any JIT observed
inside measured calls will be disclosed rather than treated as pure GPU cost.

Admission requires invariant target-only/speculative output and repetition
identity, positive actual acceptance, independent quality agreement and useful
request-time benefit. Stock-mode correctness, quality and cost are measured for
context; equality to stock-mode arithmetic is not assumed.

Root is sole GPU operator. The local supervisor is
`runs/local-vllm-final-20261101-01/state.json`, with authenticated birth/boot and
stop details in its `identity.json`. No engine or baseline switch is adopted
while confirmation is pending.

Even a successful ordinary-engine result would establish a clean verifier
foundation only. Our custom binary execution and a same-engine primary Q4_0
EAGLE comparator would still need implementation/admission before claiming a
Binary EAGLE win. Existing historical F16 and modern quantized-target evidence
remain separate and unchanged.

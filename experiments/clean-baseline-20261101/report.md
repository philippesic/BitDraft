# Clean verifier foundation qualification

Unmodified vLLM 0.32.0 with its built-in batch-invariant mode qualifies as an
ordinary Qwen3-4B BF16/EAGLE3 foundation on the local RTX 5080 within the frozen
test below. Recommend this maintained implementation over further development
of our private target-math workaround, if changing target precision/model and
engine is acceptable. No project migration or binary port has been implemented.
The user rejected developing around the broad local CUDA correctness workaround.
This investigation preserves historical evidence and does not relabel old runs.

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

## Fresh final confirmation: bounded foundation pass

Source, settings and target/draft file hashes were frozen before generating
another 24 performance prompts and 24 independent quality cases. All 24 correct
and 24 known-wrong judge controls were executed. No tuning on final outputs.

The final four cells are stock/invariant mode crossed with target-only/EAGLE3.
Five alternating repetitions give 480 performance requests, with 96 quality
requests in repetition zero. The public `LLM.generate` call wall includes
scheduling, prompt processing and decoding; it is not HTTP serving capacity.
Model loading and warmups are excluded from measured calls. Inference-time JIT
warnings occur, without enough phase markers to assign all of them to warmups
versus measured calls. Timings therefore describe the observed full API calls,
including any compilation within them; they do not establish pure GPU or fully
warmed steady-state cost. No requests or repetitions were removed after inspection.

Admission requires invariant target-only/speculative output and repetition
identity, positive actual acceptance, independent quality agreement and useful
request-time benefit. Stock-mode correctness, quality and cost are measured for
context; equality to stock-mode arithmetic is not assumed.

The run completed all 20 cells with exit zero. Root independently recomputed the
raw token/text/stop comparisons, native metric deltas, exact quality outputs and
frozen judge scores. Source/input hashes and cell kwargs still match the freeze.
[The recomputation script](analyze.py) produces [the compact result](engine-confirmation.json)
using only the standard library and retained run evidence.

| Mode | Target-only seconds | EAGLE3 seconds | EAGLE3 / target speed |
| --- | ---: | ---: | ---: |
| Built-in invariant | 32.092 | 19.951 | 1.609× |
| Default | 30.855 | 17.849 | 1.729× |

These are mean total request seconds per 24-prompt repetition. All performance
requests generate 128 tokens. The invariant arms measure 95.72 and 153.97 output
tokens per full-generate second. Every repetition favors EAGLE3; across the four
cells repetition CVs are 0.25–0.38%. This is a matched single-sequence API result,
not HTTP capacity, concurrent serving or a cross-engine benchmark.

Invariant EAGLE3 matches target-only on all 120 paired performance requests and
all 24 quality cases, including token IDs, text and stop state. Each arm repeats
identically. In default mode, EAGLE3 differs on 14/24 performance prompts in every
repetition. Default versus invariant arithmetic also changes 15/24 target-only
and 11/24 EAGLE outputs; invariant mode is not stock-output preservation.

The invariant mode's observed wall cost versus default is 4.01% for target-only
and 11.78% for EAGLE3. Equal output lengths make these useful operational cost
comparisons, but the generated contents differ across modes. Its 1.609× benefit
uses the invariant target-only denominator, whose outputs actually match.
Actual invariant draft acceptance is 7,965/22,155 proposed tokens (35.95%);
all five cells have positive draft and acceptance counts. No fallback-only gain.

The frozen independent judge scores all four arms 14/24 with the same ten failed
cases. Candidate invariant outputs equal its target-only quality outputs exactly.
All 24 correct controls pass and 24 known-wrong controls fail. These are eight
parameterized task templates, not a broad answer-quality certification. Strict
JSON/answer errors remain real model limitations; no scorer repair was applied.

Final cost: 503.74 seconds performance calls, 53.24 seconds quality calls,
368.02 seconds engine initialization, 44.71 seconds excluded warmups, and
1117.33 seconds total phase wall (18.62 minutes). The balance is subprocess,
integrity-check and harness overhead. Inference JIT cannot be separately billed
from those clocks. The earlier successful development gate costs 177.30 seconds
total wall. Installation, failed preparations and source/model integrity work
are preparation, not evaluation or training; their complete aggregate clock was
not recorded, so no exact total preparation cost is claimed. No new training.

Root rechecked the terminal local supervisor
`runs/local-vllm-final-20261101-01/state.json`, matching boot and absent recorded
process groups. `nvidia-smi` lists no compute applications. Twenty-one owned short
RPC directories/sockets were removed after release. Useful environment, caches,
models and evidence remain; the engine evidence/environment tree occupies
8.69 GB of allocated blocks, including shared package data (not a unique growth
or reclaim estimate). WSL free space measured 22.05 GB before and after that
small cleanup; no Windows-volume reclaim is claimed. Ignored release/storage
receipt: `runs/clean-engine-qualification-20261101/release-and-storage.json`.

This establishes an ordinary verifier foundation only. Our custom binary
execution and a same-engine primary Q4_0
EAGLE comparator would still need implementation/admission before claiming a
Binary EAGLE win. Existing historical F16 and modern quantized-target evidence
remain separate and unchanged.

## Recommended decision and limits

Use this pinned upstream stack as the next foundation if the research can use
dense Qwen3-4B BF16. It avoids maintaining custom target arithmetic, with a
measured cost that still leaves useful ordinary EAGLE acceleration. The tested
settings are the freeze above: one sequence, context 2048, greedy generation,
three draft tokens and Triton attention. Quantized Qwen3.5, longer contexts,
sampling and concurrent batches are not admitted by these results.

Upstream calls [batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)
beta and documents both deterministic kernels and disabled optimizations. This
is a maintained implementation with bounded local validation, not a promise
that correctness is universally solved or free. Ordinary [EAGLE3 support](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/)
does not provide our packed binary execution automatically.

The user owns the major engine/target migration decision. Until that is chosen,
preserve checkpoints and the primary Q4_0 requirement; do not resume training on
a new teacher or silently substitute BF16 EAGLE for the primary comparison.

# Modern packed W1A8 execution qualification

November 3, 2026. Local RTX 5080 / SM120, CUDA13.1.115, F16 target/draft KV,
Qwen3.5-4B Q4_K_M target and correctness mode32. **Initialization only: zero
training updates.** No generalization, answer-quality or throughput win is claimed.
The primary comparator remains frozen DFlash Q4_K_M; historical F16 work is separate.

## What is now measured

The retained modern EAGLE checkpoint can be serialized and executed with all nine
linear projections stored as packed one-bit signs and F32 row scales, using
per-token A8 codes and real CUDA DP4A. It borrows the quantized target's native
F32 embeddings. No expanded Q8 weight publication or private embedding is used.

The corrected exporter derives Q/K head counts and rotary dimensions from GGUF
metadata. It permutes only rotary40of160, including scales and affine offsets;
the unrotated tail is preserved. Original metadata-free historical fixtures keep
their full-head Q32/KV8 convention. F16 and BF16 base files are accepted with
logical tensor-shape checks. Twenty-six focused export/provider CPU tests pass,
including independent wrong-full-head rejection. The real initializer's nine
projections /76,880 rows pass the independent source audit. Its GGUF is40,802,176
bytes. The reproducible zero-update NPZ scratch was retired only after that audit;
retained BF16 source, recipe, manifest and output preserve reconstruction.

New fixed-A8 operations pass CPU and real-CUDA reference tests at K2560/5120/7680,
N1/4/7/33, including the50,000-row head, zero vectors, RNE ties, subnormal inputs,
and rejected metadata/layout variants. Codes, scales and final products are
byte-identical to the independent reference on those fixtures. Supported K must
be a positive multiple of32 with bounded integer sums; other modes fail closed.

The development server test completed six arms over three reused prompts, with
128-token limits and two excluded warmups per arm: frozen target-only, candidate
target-only, frozen DFlash Q4_K_M d2, candidate DFlash Q4_K_M d2, candidate ordinary
EAGLE Q4_0 d2, and candidate W1A8 initialization d2. Root independently audited all
18 measured raw responses: token IDs, text, stopping and truncation match both
current frozen target-only and the saved pilot references. All16 distinct loaded
runtime images match the appropriate frozen/candidate pins. All nine packed
projections load, and the drafter occupies28.47MiB of CUDA model buffer. The
untrained W1A8 arm has67 accepted /408 proposed draft tokens on the measured
requests. This is positive native drafting, not a training result or native
actor/parent-prefix qualification. One repetition on three reused prompts is
insufficient for a performance or untouched-quality claim.

## Data and actor gates

The CPU-only admission gate authenticates the baseline, exact target/runtime,
native capture/build/tokenizer, serving request and response, raw arrays and
history. The two known helper/server label disagreements are excluded from
supervision and numeric continuation-protection eligibility. Seventeen CPU tests
pass; real packets admit16/18 teacher rows, and altered eligibility flags cannot
bypass a label mismatch. Actual parent-success protection remains unproven.

The actor shift is explicit: feature[t] + token[t+1] predicts token[t+2] at
decoder/cache position t. Both adjacent teacher rows must pass the numeric gate;
the final teacher row has no following actor label. The real cases admit13/16
such transitions. A CPU zero-update actor with exact native scales/norms and
validated offset-to-absolute vocabulary maps predicts2/13 eligible labels
correctly; unsupported labels0. These are teacher-forced CPU estimates, with a
different denominator from native aggregate acceptance. Current Torch A8 dots
dequantized values before row scaling, while the native operation scales after
the integer dot. Full actor/native decision comparison is still required before
QAT. Do not change historical arithmetic or infer protection sets from aggregates.

## Source and reproducibility

[Packed source patch](packed-w1a8.patch) applies cleanly to the retained
quant4group correctness source commit
`aff0cbd567ef3ab4a3bf2f35c9b9c0e3e2fdc265`. It appends two GGML op IDs, adds fixed
row-A8 CPU/CUDA execution, and adds strict EAGLE packed loading and graph wrappers.
Ordinary paths remain available and were checked under both runtimes above.
The frozen runtime and parent gitlink are unchanged. This candidate backend has
not been broadly qualified on all targets or performance-equivalent workloads.

The source snapshot and build recipes are under
`runs/w1a8-modern-packed-runtime-20261103-01/`. Build03 incrementally fixes a missing
string-array template instantiation after Build02 compiled but failed at final
link. Build01 failed before source validation because it omitted the existing
private CUDA/glibc compatibility include. The retry reuses the exact baseline
include, verified to differ from system math headers only in the two recorded
noexcept declarations. No system headers were edited. The first op wrapper
failed an overstrict unused-library expectation after every numerical case
passed; the corrected wrapper also passes and exits zero. Failed evidence is kept.

## Evidence and next gates

- `runs/w1a8-modern-packed-init-20261103-01/{freeze,export,independent-audit,result}.json`
- `runs/w1a8-modern-admission-20261103-01/{qualification,actor-alignment-qualification,source-recovery}.json`
- `runs/w1a8-modern-actor-readout-20261103-01/{report,independent-readout-audit}.json`
- `runs/w1a8-modern-packed-runtime-20261103-01/{build03-result,op02-result,release-and-storage05}.json`
- `runs/w1a8-modern-native-admission-20261103-01/{freeze,result,root-independent-audit,root-library-audit}.json`

All owned GPU servers, supervisors and builders are terminal and their recorded
processes absent. Useful source/build caches and original checkpoints are kept.
WSL has approximately3.6GiB free and C:3.5GiB. The pending storage request still
applies to the two historical full-state retention arms and future full-sign
training. The next scientific work is bounded native/actor trajectory comparison,
native successful-continuation references, and broader authenticated training
data. Untouched correctness/quality and five repeated full-request timings remain
required before validating an early/long screening predictor or screening backlog.
No push.

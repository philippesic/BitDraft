# DFlash execution optimization: agreed scope

## Decision and limits

November 5, 2026: the user wants to understand and improve draft execution before
further head/embedding/KV precision reduction. Root and independent Astra xhigh
agree on the revised investigation below. This is a planning result, not a new
kernel implementation, GPU experiment or demonstrated speedup. The completed
[precision campaign](PRECISION_CAMPAIGN.md) and its frozen artifacts remain intact.

The first subject is the compact G4/D8 recipe: binary FFN weights, A4 gate/up
inputs, A8 down inputs, fixed/learned bounds, F32 affine metadata and gains,
Q4_0 fusion/attention, target-shared Q6_K head/embedding, F16 KV. Preserve a
specific checkpoint/config for each comparison. Fixed and learned checkpoints
are separate artifacts, not interchangeable controls.

Target/verifier, target precision, original proposal policy, context, model
features and training state stay fixed. No QAT, new loss, pruning of model
capacity, vocabulary shortlist, cache quantization or new architecture is part
of the initial execution experiment set. Shape-specific kernels and proven
dead-output elimination are execution candidates, subject to the gates below.

## What the source establishes

- All logical activation widths currently write INT8 codes and use DP4A.
  Binary signs remain packed in global memory and expand into register lanes.
  A4 is not first packed into nibbles and then upcast; it is quantized directly
  into byte storage. Smaller logical widths do not shorten the current dot loop.
- The quantizer scans gained inputs and reduces their maximum even when a stored
  fixed/learned bound will replace that maximum. It rereads/reapplies gains to
  quantize. Finite-input validation must survive specialization.
- The graph unconditionally requests a contiguous copy before gate/up packing
  and before down packing in each layer. The pack constructor already inserts
  a copy conditionally when necessary. These are twelve candidate redundant
  copy nodes per six-layer forward; trace whether they execute and cost time.
- The pinned depth-two scheduler ordinarily constructs an anchor plus two mask
  positions: expected generation N=3. Confirm actual shapes in the trace; depth,
  training block size and matrix column count are different quantities.
- The graph projects all final positions through the vocabulary head, while
  ordinary DFlash sampling skips the anchor. This identifies a possible dead
  output, not permission to delete the anchor from attention or cache state.
- The current W1 dot separates positions in its grid. The Q4 baseline has a
  grouped vector path sharing weight reads over up to four columns. Its Q4_K
  vector arithmetic also uses DP4A. Therefore DP4A versus Tensor Cores is NOT an
  established explanation for the relative latency gap.
- `GGML_CUDA_DECODE_CONSISTENT=32` is frozen in the campaign benchmark. It selects
  consistency-preserving dispatch and disables generic CUDA fusion. Do not
  globally remove it to make the drafter faster and inadvertently change the
  target/verifier. New fusion must have an explicit drafter-only boundary.

Existing full-draft host spans indicate the gap is in generation rather than
catch-up, but asynchronous GPU work can be charged at later synchronization.
They do not apportion packing, dot, head, attention or idle time. No causal
percentage breakdown, occupancy or DRAM-bandwidth claim is established yet.

## 0. Trace first; qualify replay for comparisons

Take one short observational trace of unchanged W1 and matched Q4. Identify
actual projection shapes/N, kernel dispatch, CONT copies, CPU/GPU synchronization,
allocation, graph captures/replays, launch gaps, output transfers and head work.
Nsight Systems and CUDA-event region timings are sufficient to start; denied
hardware counters are not a blocker. Estimated operand traffic is not measured
DRAM bandwidth. Record clocks/temperature and distinguish warm from cold work.

Then qualify a small model-specific replay for causal timing:

- Use the same external token history, native target features, positions/masks,
  required outputs and catch-up workload for W1 and Q4. Rebuild EACH model's own
  projected cache; do not share one model's cache contents with the other.
- For original versus optimized W1, restore equivalent state, including cache
  writes and proposal mutations, before repetitions. Replaying a mutating graph
  repeatedly without reset is not identical work.
- Cover both stable shapes and a short changing-context sequence. Graph warmup
  can reset when node properties change; two warmup requests alone do not prove
  each graph is warm. Report capture amortization and live-request behavior.
- Synchronize at region boundaries rather than after every production kernel.
  Use complete projection/forward timings alongside stage events. Verify the
  replayer reproduces its own original native trajectory before interpreting it.

## 1. Remove avoidable preparation and copies

Run four variants: unchanged, copy-removal only, bound-specialization only, both.
This small factorial identifies their independent and combined contribution.

**Copy removal:** pass the original tensor to the pack constructor and retain its
conditional contiguity fallback. Check actual noncontiguous cases, memory lifetime,
aliasing and graph capture. Do not globally bypass CONT operations.

**Bound specialization:** for fixed/learned inference bounds, remove the maximum
scan/reduction and quantize in one retained pass. Preserve finite gained-value
checks, overflow handling, RNE, saturation, signed zero and subnormal behavior.
The sum of quantized codes is still required for nonzero midpoints; that reduction
does not disappear when the maximum reduction disappears. Specialize A1 arithmetic
only where semantics permit: positive gains do not justify using the raw input
sign, since a negative value may underflow to negative zero after multiplication.

If preparation remains important, compare a narrowly chosen warp/block mapping
for the observed small N; the current one-block-per-position arrangement may not
use the device well. Do not assume more blocks improve total cost. Gains are
already fused into preparation, gate/up already share the prepared inputs, and
output scaling/midpoint corrections already share the projection epilogue.

## 2. Compete cross-position DP4A reuse against signed INT8 MMA

On the retained step-1 implementation, test only two initial matrix candidates:

1. A packed-weight DP4A kernel reusing unpacked signs over the actual draft
   positions. Inspect sign-word reuse, row/position tiling, reduction overhead,
   activation reuse and register pressure. Cache reuse already exists, so source
   loads avoided are not automatically proportional DRAM savings.
2. A signed INT8 Tensor Core adapter, expanding packed signs within its execution
   tiles. Parameterize actual N and zero inactive columns correctly. Preserve
   gains, quantized codes, nonzero midpoints and the F32 epilogue contract. Tiny
   N and padded columns may make it lose despite higher hardware peak throughput.

Start with gate/up 9216 x 2560 and down 2560 x 9216 at observed generation N.
Check N=1 and tile-boundary cases for correctness; extend performance coverage
only for shapes that actually occur. Shape-specific dispatch can retain different
winners for gate/up, down, small N and other admitted shapes.

The historical DSpark kernel hardcodes seven positions and excludes the modern
affine contract. Its unsigned-sign variant also requires a RAW integer code sum
for `2*dot - sum`. Today's midpoint stream stores a scaled F32 sum. Do not recover
an integer by dividing that float. Prefer signed MMA initially; explicitly add
and charge a raw-sum representation if later needed.

Keep resident-expanded INT8 weights out of the first set: the eighteen matrices
would use 405 MiB for signs instead of 50.625 MiB packed, before metadata. On-load
binary-layout prepacking is a possible optimization; expanded residency is a
separate memory tradeoff, not a one-bit runtime-storage claim. Count initialization,
conversion, workspace and persistent memory in all timings/resource reports.

## 3. Choose one fusion or dead-output experiment from the profile

Do not implement the whole inventory at once. Advance the larger measured,
removable cost; retain the other candidates for a later bounded round.

**Fusion candidates, from narrower to broader:**

- Fuse gate/up outputs through SiLU, multiplication, gains and down quantization.
  The midpoint's whole-position code sum still requires a reduction.
- Fuse down output epilogue with residual addition if the graph permits it.
- Joint gate/up matrix execution, then prenorm/gain/quantizer fusion only if
  measured launch/materialization costs justify the larger implementation.

A tiled joint-matrix kernel cannot independently obtain the full down-input sum
in every tile. Include partial sums, completion launches or other synchronization
in its design/cost; do not claim that fixed clipping makes all reductions local.
Watch register pressure, occupancy and arithmetic changes in SiLU/reductions.
The build uses fast-math flags plus FTZ disabled for the custom translation unit;
explicit rounding intrinsics are part of the actual contract. Preserve operation
semantics rather than assuming flags were uniformly conservative.

**Unused anchor head output:** audit every head/embedding consumer, output-index
mapping, sampler/backend and diagnostic callback. Keep the anchor through
attention, where other positions depend on it. Prune only its unused vocabulary
projection if proven dead. Changing a requested-logit flag alone may not prune
the graph's explicit projection. Later consider anchor work after FINAL attention,
including its last FFN, only with a broader dependency proof. Neither change
implies a one-third latency saving: the grouped head kernel shares weight reads.

If a safe optimization applies to Q4 too, measure Q4 with it under the same
qualified runtime. Distinguish absolute W1 improvement from advantage over the
fairly optimized comparator. All changes remain isolated from target/verifier.

## Conditional reserves and other recovery opportunities

- INT4: separate packed storage from actual native instructions and full-pipeline
  speed. Verify SM120 compiler lowering and machine code before claiming a
  mechanism. Do not apply SM100-only facilities to the RTX5080.
- Exact-code FP16 MMA: if integer MMA disappoints and dot cost remains large,
  consider converting existing integer codes and signs to exactly representable
  FP16 operands with F32 accumulation, retaining the original scales. Current
  sum magnitudes are below 2^24. Qualify exact dots and conversion-inclusive timing;
  this is not a proposal to requantize activations to an FP4 grid.
- A1 XOR/popcount, A2 specialized arithmetic, bitplanes/LUT and persistent kernels
  are bounded reserves for a remaining measured bottleneck, using the previous
  binary-method screen. Do not restart a broad algorithm search automatically.
- Audit graph recapture, allocator/scratch reuse, synchronization/readbacks, head
  output transfer, attention dispatch and target-feature/cache update work where
  the observational trace identifies cost. Reuse features/caches only with exact
  ancestry and mask/position semantics. These are leads, not measured defects.
- NVFP4 conversion, head/embed/KV precision changes and model/topology changes
  stay outside this execution-only plan; they alter the precision hypothesis.

Official support references: [PTX MMA and precision contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma),
[cuBLASDx architecture-specific pipelines](https://docs.nvidia.com/cuda/cublasdx/performance_guide_api_blackwell.html).
The latter explicitly routes SM120-class parts away from its tcgen05 path.
Instruction interfaces alone do not demonstrate the machine execution or speed
of a particular low-bit kernel; the existing local binary-MMA lowering result is
the concrete cautionary example.

## Correctness, performance and stopping gates

Copy removal, preparation specialization and integer-dot replacement initially
belong to the EXACT lane. Independently check codes, scales, sums, integer dots
and affine outputs, including rounding ties, saturation, zero/signed zero,
subnormals, nonfinite/overflow rejection, nonunit gains, nonzero midpoints, every
admitted width/mode, actual N and inactive columns. Check native proposals,
required outputs and cache effects, not only final verified target tokens.

Fusion changing floating reductions/SiLU belongs to a separately declared
approximate lane with an error envelope and bounded native trajectory/acceptance
checks. Changed quantizer codes are not dismissed as ordinary float drift.
Timebox irrelevant cross-backend parity work; preserve meaningful native gates.

Microbench savings must survive complete projections and clean native forward
cost, including preparation and synchronization. Retain useful low-risk savings
below ten percent. A ten-percent reduction in clean W1 forward cost is a proposed
CAMPAIGN milestone, not a per-patch threshold or promise. Report paired intervals;
an interval crossing zero leaves a claimed gain unresolved, and "at least 10%"
requires interval support. Beating Q4 draft cost is a separate outcome.

Final confirmation compares original W1, retained optimized W1 and common-runtime
Q4: two warmups and at least five alternating paired repetitions, complete draft
and request timings, model/workspace/peak memory and unchanged settings. Five
repetitions are a minimum, not guaranteed resolution. Include short changing
contexts as well as replay. Poor existing acceptance does not veto a proven
execution improvement; report full-request throughput as context without claiming
an execution speedup repaired model quality. No ceiling is inferred by subtracting
asynchronous host spans from request time.

## Debate outcome and source map

Root accepts Astra's revisions: preserve consistent target dispatch, promote
copies to the first experiment, qualify each model's replay/cache separately,
use actual expected three-position geometry, adapt MMA to affine sums/epilogue,
keep DP4A competitive, narrow fusion, audit unused anchor head work and reserve
true low-bit mechanisms for measured residual costs. No material disagreement
remains. This consensus supplies an experiment order, not empirical speed evidence.

Pinned implementation references (paths relative to the repository root):

- `scripts/dflash_baseline/precision_benchmark.py`: frozen runtime environment.
- `runs/precision-native-20261104-04/ggml/src/ggml-cuda/ggml-cuda.cu`:
  fusion restriction, grouped consistent dispatch, graph warmup/reuse.
- `runs/precision-native-20261104-04/src/models/dflash.cpp`:
  unconditional pack-input copies, affine FFNs and full-position head.
- `runs/precision-native-20261104-04/ggml/src/ggml.c`: conditional packing copy.
- `runs/precision-native-20261104-04/ggml/src/ggml-cuda/w1a1.cu`:
  two-pass quantizer, per-position DP4A, existing fused metadata operations.
- `runs/w1a8-modern-packed-runtime-20261103-01/source/common/speculative.cpp`:
  anchor/mask construction and sampling output consumers.
- `experiments/binary-pipeline-profile-20261104/README.md` and
  `experiments/binary-methods-20261104/README.md`: historical evidence and limits.

Astra review: <t3-thread://v2/92296584-2a2c-4998-8f7d-4ecb7d7f5ff2/thread:delegated-task:command%3Amcp%3Af5603e63-bb40-42ba-9075-59c15144f9d8%3Adelegate-task%3Adec6eef2-1f97-4d6d-b35f-ff2e54128744>

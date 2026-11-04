# Preliminary binary-arithmetic candidate screen

**Keep several options open: shape-dependent W1A8 bit-serial execution, selective
A4/A1 layers, and extra binary weight capacity.** No final architecture/precision
choice or permanent rejection follows from this exploratory screen. Training
updates: **zero**. Supervised GPU-job wall: **126.66 seconds**, including startup,
model/data loading and checks; this is not pure GPU execution time.

This uses the existing modern4B geometry, twelve already-TRAIN groups and actual
RTX5080 kernels. It does not establish new serving performance, 9B/NVFP4 transfer
or generalization. The original held-out/final pool remains untouched.

## What arithmetic was actually tested

The current SM120 W1A8 runtime stores packed one-bit weights but expands signs
into register lanes for **DP4A signed-int8 dot products**. Its current execution
is not bit-serial; the historical SM75 implementation included bit-serial paths.
Precision names alone do not specify arithmetic or hardware speed.

New standalone prototypes implement W1A1 XOR/popcount, W1A4 and W1A8 bit-plane
AND/popcount accumulation, and a DP4A reference based on the current kernel.
Disassembly confirms POPC and IDP.4A instructions. These are ordinary CUDA
instruction paths, not a demonstrated binary Tensor Core implementation.

A1 uses the existing F64 mean-absolute activation scale; A4/A8 use per-token
absmax, nearest-even signed codes and F32 scaling. A1 scaling can itself be
costly at long reduction widths; these are specific quantizers and prototypes,
not limits on all possible binary implementations.

The final kernels pass200 correctness cases across the two architecture geometry
sets. Every packed activation code and scale is checked, all seven output rows
of the small zero/subnormal/large/random fixture are checked, and eight selected
rows per actual-size matrix are checked against independent integer dot and
scaled-output references. A first successful timing implementation was refined
to remove unused code-sum reduction work from DP4A/A1; final results use v2.

## Kernel costs: useful, shape-dependent savings

Five alternating repetitions time CUDA graphs containing32 repeated operations.
Weights and inputs are warm and resident. Quantization/packing is included in the
following numbers; cached-dot and packing-only measurements are also retained.
These are standalone device costs, **not complete draft/request latency**. Native
scheduling, shared packing, attention, catch-up and verification are absent.

| Current50k×2560 head shape | DP4A A8 | Bit-serial A8 | Bit-serial A4 | XOR/popcount A1 |
| --- | ---: | ---: | ---: | ---: |
|1 input row|26.25µs|22.44µs|16.53µs|12.62µs|
|2 input rows|48.56µs|40.07µs|28.52µs|20.51µs|

A8 bit-serial preserves the same integer quantization/dot arithmetic and costs
about15–17% less on these head cases. However, at1input row it is about9% slower
on the2560×7680 fusion shape and14% slower on the640×5120 K/V shape. A shape
choice is more plausible than blanket replacement. A4/A1 usually offer larger
kernel savings, but do change precision. N8/N32 are exploratory batch sizes,
not promises of a deployed parallel architecture's speed.

The previous complete-cost study already shows the importance of catch-up and
verification rounds. These small kernel wins alone do not establish a path to
beating DFlash; improving accepted continuations remains necessary.

## Zero-update precision probes

Four TRAIN groups/domain, one predeclared window/group, two teacher-forced
positions/window. Prefix anchors include the rendered prompt and lie near128.
The paid A8 signs/scales are fixed throughout. **This favors A8:** the other
precisions have received no adaptation. These counts are not native acceptance,
independent generalization, or a fair equal-QAT model ranking.

| Activation policy; all weight signs stay binary | Target top1 matches /24 | Agreement with A8 /24 |
| --- | ---: | ---: |
|A8 control|12|24|
|A4 everywhere|8|9|
|A1 everywhere|4|6|
|A4 body, A8 head|7|8|
|A1 body, A8 head|6|7|
|A4 only in the three MLP projections|10|16|
|A1 only in MLP; exploratory follow-up|9|13|
|A4 except fusion; exploratory follow-up|6|10|
|A4 except fusion and head; exploratory follow-up|7|10|

Selective MLP precision was less disruptive in this small probe. It is worth
bringing into design discussion, with A1 requiring particular care. Full A1 is
a higher-risk option here, not rejected forever.

Fusion inputs have a median absmax/RMS ratio33.66. A4 absmax quantization maps
93.72% of their entries to zero and has input normalized squared error0.415,
versus0.00555 for A8. Yet preserving fusion at A8 did **not** rescue the whole
model in the adaptive follow-up. This directly illustrates why a local error
score cannot choose the complete architecture.

A8 repeats bit-exactly after each perturbation sequence and between the two
runs. Parameter versions remain unchanged. No optimizer is constructed, no
checkpoint is trained, and no variant is exported as a native model.

## Weight capacity options

Separately, original BF16 weights are approximated by row-scaled binary weights,
group128 binary weights, and a greedy sum of two row-scaled binary planes. All
use the same captured current-actor inputs. These are zero-training local
reconstruction checks, not replacements for the paid checkpoint.

| Representation on current nine projections | Sign+scale payload | F32 scale values | Effective bits/weight |
| --- | ---: | ---: | ---: |
|One binary plane, row scales|28.05MiB|76,880|1.011|
|One plane, group128 scales|34.70MiB|1,819,200|1.250|
|Two binary planes, row scales|56.10MiB|153,760|2.021|

Payload excludes metadata, norms, mappings and runtime staging. Two planes need
two binary dot contributions; groups need scaled partial sums. Neither has a
measured native runtime or whole-model quality result in this screen.

Two planes reduce output reconstruction error on all nine measured projections.
For the head with A8 inputs, normalized squared error falls0.1442→0.0387. But its
agreement with dense head top1 is18/24 versus19/24 for one plane: even this local
error improvement is not a guaranteed token-choice improvement. Group128 improves
many local errors but worsens fusion output error0.1762→0.2336. Its much larger
trainable scale capacity may still matter after QAT; this initializer probe does
not test that hypothesis. Selective extra capacity is worth discussing alongside
activation precision, rather than adding it everywhere by default.

## Architecture branch: parallel binary drafting

The released4B DFlash checkpoint has six layers,43body matrices and634.39M matrix
parameters. Its fusion uses eight target taps, versus three in the current
recurrent actor. Its checkpoint contains no own output-head/embedding matrices;
the local runtime supports sharing these with the target. Binary body arithmetic
would therefore not imply that the shared target head becomes binary.

A hypothetical row-binary body has80.02MB of signs/scales, versus the actual
381.46MB Q4_K_M file; these are different accounting categories, not an exported
model-size claim. F32 sign masters, gradients and two Adam moments alone would
be10.15GB, plus2.54GB for a materialized hard-sign copy if retained. Actual memory
fit and training speed are untested. The new kernels also pass the six unique
DFlash body shapes atN1/2/8/32, but no binary DFlash accuracy was measured.

This remains a worthwhile structural option for joint design, with explicit
costs: new eight-tap captures, parallel objective/cache semantics, representation
adapters, export/native qualification and actual memory checks. Existing TRAIN
source prompts/exclusion ancestry are reusable; the current7680-wide feature
captures are not automatically an eight-tap teacher. No new parallel model port
was undertaken during this lightweight screen.

## Suggested discussion order

1. Keep W1A8 with shape-selected bit-serial/DP4A execution as a low-change kernel
   building block. Native end-to-end qualification is still required.
2. Compare selective A4/A1 policies and selective group/extra-plane capacity as
   candidate recipes. The small current probes justify attention, not adoption.
3. Decide whether to retain the recurrent draft or invest in a parallel binary
   drafter. The latter has a clear implementation/data/resource cost and no
   current binary-quality evidence; it should not inherit released DFlash scores.

[Compact evidence](screen.json), [raw summary](../../runs/binary-screen-20261104-01/summary.json)
and [scripts](../../scripts/binary_screen/README.md) preserve the exact scope,
inputs and measurements. All owned jobs are stopped. No push.

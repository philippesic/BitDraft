# Quick binary pipeline profile — RTX 5080

The retained DSpark W1A8 path already uses integer Tensor Cores for seven-position
FFNs. Replacing that path with existing bit-serial `planes7` preserves tokens and
speculative decisions but makes its dot products much slower. Binary arithmetic
is not automatically faster than the retained implementation.

## Native comparison

Original Qwen3-4B F16 target, retained DSpark parent3504 export and runtime, D5,
seven-position body. Same pinned config as the preceding DSpark comparison.
One already-consumed code DEV prompt, one warmup and one measured request per
arm, 128 output tokens. Both arms: 33 rounds,164 proposed,94 accepted. Exact
historical target token IDs and inter-arm output signatures/counters agree.
Only GGML_W1A8_ALGORITHM changes: imma_mix versus planes7. No training.

Request-bounded Nsight Systems CUDA kernel sums (milliseconds):

| Stage | Retained integer Tensor Core path | Existing bit-serial path |
| --- | ---: | ---: |
| Activation quantization |0.809|0.810|
| Code sums / bit-plane preparation |0.174|0.363|
| Dot products, including output scaling |3.361|21.940|
| Binary-weight pipeline total |4.345|23.113|

429 corresponding dot launches per arm. Bit-serial dot time is6.53× the
retained integer path; complete listed pipeline is5.32×. This is a one-request
instrumented diagnostic, not a repeated throughput benchmark or optimal-kernel
claim. Binary matrix MMA was not benchmarked. The existing planes7 implementation
uses ordinary packed-bit arithmetic. The retained imma_expanded kernel expands
packed signs for integer matrix arithmetic; other activation counts have fallback
paths. Do not describe this DSpark endpoint as the recent modern actor's DP4A path.

Retained binary pipeline accounts for0.70% of all kernel-duration sums in this
historical F16-target request. This is neither draft-only fraction nor a removable
wall-time estimate. F16 matrix operations dominate the whole trace; no precise
split of target versus other drafter operations is attempted. Request wall and
kernel-active interval union are retained only as instrumented diagnostics;
they are not hardware occupancy or proof that the remaining gaps are removable.

## Isolated projection comparison

Actual DSpark binary FFN shapes:9728×2560 gate/up and2560×9728 down, with1and7
input positions. Random resident operands; existing standalone DP4A/A1/A4/A8
kernels, three alternating repetitions of32-operation CUDA graphs. Quantization,
packing, dot-only (including scale/store), and complete projection timed separately.
20 cases pass independent activation-code, integer-dot and scaled-output checks.

Representative down projection (2560×9728), median microseconds:

| Positions | Mechanism | Prepare | Packed dot + scale/store | Complete |
| --- | --- | ---: | ---: | ---: |
|1|DP4A A8|8.24|5.70|13.69|
|1|XOR/popcount A1|8.64|2.07|10.59|
|1|Bit-serial A8|10.64|4.47|14.74|
|7|DP4A A8|8.04|27.54|35.57|
|7|XOR/popcount A1|8.83|7.16|15.74|
|7|Bit-serial A8|10.96|20.35|31.08|

Independent stage medians need not add exactly to complete timings. A1 changes
activation precision; these timings say nothing about acceptance after that change.
Preparation is substantial, especially for one-position long-input operations.
The prototype assigns separate token blocks, and its dot time grows substantially
with positions. Cross-position reuse is an opportunity for that prototype, not a
new discovery missing from the already optimized native endpoint. Do not compare
Nsight native timings directly to uninstrumented standalone timings as a speedup.

## Interpretation

1. Treat the retained integer Tensor Core implementation as the real engineering
   comparator for any new binary kernel, not the basic standalone DP4A control.
2. Investigate efficient matrix-level binary execution and data reuse before
   committing to plain bit-serial loops. This profile does not establish BMMA's
   speed on either GPU.
3. Reduce/share activation preparation where it dominates small operations. The
   native path already shares some quantization:264 quantizer launches versus429
   binary projections. Avoid assuming every projection packs separately.
4. Optimizing existing binary FFNs alone has limited leverage on this historical
   request. A broader drafter cost breakdown would be needed before deciding
   where to introduce additional binary layers; no architecture chosen here.

Nsight Compute was attempted once and denied hardware-counter access with
ERR_NVGPUCTRPERM. Therefore SM occupancy, memory throughput and compute saturation
remain unmeasured. No driver settings changed. Nsight Systems worked after
preserving CUDA injection variables and allowing graceful server exit; the first
attempt's empty CUDA trace is retained as a diagnostic failure, not evidence.

## Evidence and scope

Raw: runs/binary-pipeline-profile-20261104-01. `analysis.json` is recomputed by
scripts/binary_pipeline_profile/analyze.py. Native source/config pins, responses,
request bounds, maps, Nsight reports and supervisor receipts remain outside Git.
Request UNIX timestamps are aligned with the Nsight session epoch; selected
kernels must lie wholly inside the measured request. Startup and warmup excluded.
All owned GPU work finished. No serving changes, new model, training or push.

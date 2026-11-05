# Binary execution screen on RTX 5080 — November 4, 2026

The retained W1A8 integer Tensor Core path remains the fastest tested native binary pipeline. None of the alternatives produced a throughput win. W1A1 XOR/popcount remains interesting for individual shapes, but did not win the complete seven-position DSpark pipeline. This is a bounded mechanism screen, not an architecture or precision selection.

[Interactive comparison](comparison.html) · [Native chart](native-comparison.png) · [All projection shapes](projection-comparison.png) · [Machine-readable results](analysis.json)

## Matched setup and limits

- RTX 5080, CUDA13.1, native SM120a; standalone SM120. No Turing run. The 2080Ti pause remains unchanged.
- Historical **Qwen3-4B F16 target**, DSpark retained parent3504 weights, D5, full Markov rows, target-width mask3. Thirteen binary FFN projections; two down projections and other model parts retain their existing precision. This is not a fully binary model.
- W1A1 is an explicit experimental CUDA activation-precision override on the **same A8-trained weights**, with no new training or model export. Both activation packing and matmul read the same overridden layout. It is not a properly trained W1A1 competitor, nor the older untrained EAGLE comparison.
- Nine methods in one isolated runtime. Two reversed-order repetitions, one warmup per server, first reused code/prose/reasoning DEV prompts, maximum128 output tokens. Each method produced716 measured tokens across six requests. No confidence intervals are warranted.
- Separate Nsight Systems run: one warmup + one measured code request per method. GPU figures are summed binary-kernel durations divided by speculative rounds; they are not occupancy, whole-draft latency, or removable end-to-end wall time. Memset/copy operations are not included in the kernel-duration sums. Full-operation microbenchmarks and native throughput include those costs.
- Modern Qwen3.5 4B/9B, Q4_K_M/NVFP4 targets and DFlash Q4_K_M were not evaluated. Their selected primary-baseline status is unchanged.

## Native comparison

| Method | Pack µs/round | Extra prep µs/round | Dot/output µs/round | Binary total µs/round | Decode tok/s | Versus A8 default | Request tok/s | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A8_tensor | 24.6 | 5.3 | 101.9 | 131.8 | 231.70 | +0.00% | 197.45 | 52.4% |
| A8_dp4a | 24.7 | 0.0 | 192.1 | 216.8 | 230.93 | -0.33% | 194.50 | 52.4% |
| A8_planes7 | 24.6 | 11.0 | 672.6 | 708.2 | 223.96 | -3.34% | 190.28 | 52.4% |
| A8_lut4 | 24.7 | 13.0 | 693.3 | 731.0 | 223.44 | -3.57% | 189.71 | 52.4% |
| A8_sharedlut | 24.6 | 162.6 | 242.6 | 429.8 | 227.68 | -1.73% | 193.20 | 52.4% |
| A1_xor | 60.9 | 0.0 | 284.1 | 345.0 | 154.72 | -33.23% | 137.95 | 28.6% |
| A1_bmma | 60.8 | 0.0 | 226.5 | 287.3 | 154.66 | -33.25% | 137.66 | 28.6% |
| A1_tensor | 60.9 | 9.4 | 102.4 | 172.6 | 156.63 | -32.40% | 139.35 | 28.6% |
| A1_sharedlut | 60.8 | 172.8 | 238.3 | 472.0 | 153.06 | -33.94% | 136.78 | 28.6% |

Decode throughput excludes prompt processing. Request throughput includes it. Both divide total output tokens by total measured time; rates are not averages of prompt rates. Acceptance is accepted/proposed draft tokens. A8 and A1 throughput differences combine arithmetic and acceptance changes; within each precision all methods have identical draft counters.

All **90 final native responses** (72 plain including warmups,18 traced including warmups) match frozen target-only output tokens. Trace emitted-token/counter reconciliation passes, as does trace/plain parity. All exact methods preserve within-precision acceptance: plain A8 total198 rounds /984 proposals /516 accepted; A1 total296 /1462 /418. The profiled code request has33 A8 rounds versus50 A1 rounds.

## What the stages reveal

1. **Binary-MMA PTX is not native binary hardware execution here.** The m8n8k128 XOR/POPC instruction compiles and runs correctly on SM120/SM120a, but disassembly lowers this tested form to four `IMMA.16832.U8.U8` sites plus conversion/bit-manipulation instructions; no BMMA machine instruction appears. This establishes the tested compiler path, not an exhaustive claim about all Blackwell instructions.
2. **Quantization is not free.** Native A8 packing costs about24.6µs per round; A1 packing about60.9µs. This implementation computes A1 mean-absolute-value scales with double-precision accumulation. Packing/reduction/fusion is a concrete optimization target, but this screen does not isolate which instruction causes that cost.
3. **The full draft does much more than binary FFNs.** In the instrumented code request, A8 default binary kernels total0.132ms per round while the whole draft CPU-wall span averages2.565ms. The binary sum is roughly5% of that draft span. The target verification span is larger still. These scopes differ; this is not an Amdahl bound or a universal percentage for future architectures.
4. **W1A1 does not automatically halve current DSpark draft latency.** The Tensor Core A1 adapter has a2.626ms instrumented draft span per round versus2.565ms for A8. XOR/popcount averages2.776ms. This does not contradict or reverify the older untrained EAGLE lead: model, weights and execution differ.
5. **Shared LUT preparation matters.** The exact adaptation spends about163µs per round transposing immutable weights on every invocation. A production export could store that layout once. Even subtracting that entire measured transpose cost leaves roughly267µs of A8 binary kernels, versus132µs for the default on this block shape. Thus layout prepacking alone would not reverse this particular result.
6. **Small throughput gaps are not decisive.** DP4A is within0.4% of the default despite a substantially slower binary kernel sum. The whole workload dilutes binary-kernel differences. A1 method throughput gaps also remain small compared with the acceptance penalty.

## Actual-shape microbenchmarks

Resident synthetic F32 activations and packed binary weights, learned-style row scales; three CUDA-graph timing repetitions of32 operations. Separately measured phases need not add exactly because they are different timing samples. All54 correctness cases pass exact scaled-output checks against CPU integer references: all18 edge-fixture rows and16 selected rows per full matrix. Tensor kernels only specialize in seven positions. DP4A uses the native seven-position weight-reuse kernel at n=7 and a simple per-position prototype at n=1. XOR/popcount is a standalone prototype, not byte-identical native code; native profiling above is the serving measurement.

### 9728 × 2560, 1 position(s)

| Method | Prep µs | Dot/output µs | Complete operation µs |
|---|---:|---:|---:|
| A8_dp4a | 3.00 | 5.50 | 8.31 |
| A8_planes1 | 3.99 | 8.44 | 12.75 |
| A8_planes7 | 4.55 | 11.60 | 15.15 |
| A8_lut4 | 3.76 | 21.85 | 24.92 |
| A1_xor | 3.65 | 2.67 | 6.15 |
| A1_bmma | 3.74 | 9.96 | 13.62 |
| A8_sharedlut | 16.01 | 6.91 | 22.59 |
| A1_sharedlut | 17.27 | 6.89 | 23.96 |

### 9728 × 2560, 7 position(s)

| Method | Prep µs | Dot/output µs | Complete operation µs |
|---|---:|---:|---:|
| A8_tensor | 3.84 | 5.62 | 8.51 |
| A8_dp4a | 3.11 | 13.19 | 16.08 |
| A8_planes1 | 3.89 | 49.78 | 53.22 |
| A8_planes7 | 4.45 | 50.03 | 53.57 |
| A8_lut4 | 3.80 | 45.85 | 49.32 |
| A1_xor | 3.68 | 12.40 | 15.43 |
| A1_bmma | 3.66 | 9.71 | 13.15 |
| A1_tensor | 4.32 | 5.58 | 9.86 |
| A8_sharedlut | 14.80 | 18.13 | 32.83 |
| A1_sharedlut | 16.14 | 18.22 | 33.99 |

### 2560 × 9728, 1 position(s)

| Method | Prep µs | Dot/output µs | Complete operation µs |
|---|---:|---:|---:|
| A8_dp4a | 8.01 | 5.62 | 13.71 |
| A8_planes1 | 8.88 | 8.71 | 17.37 |
| A8_planes7 | 8.87 | 9.86 | 18.51 |
| A8_lut4 | 9.29 | 24.00 | 32.64 |
| A1_xor | 8.87 | 2.06 | 10.51 |
| A1_bmma | 8.65 | 27.78 | 36.17 |
| A8_sharedlut | 19.80 | 5.76 | 25.34 |
| A1_sharedlut | 21.25 | 5.89 | 26.70 |

### 2560 × 9728, 7 position(s)

| Method | Prep µs | Dot/output µs | Complete operation µs |
|---|---:|---:|---:|
| A8_tensor | 9.81 | 10.66 | 20.23 |
| A8_dp4a | 8.12 | 13.84 | 21.34 |
| A8_planes1 | 9.32 | 48.15 | 57.17 |
| A8_planes7 | 9.03 | 52.58 | 61.35 |
| A8_lut4 | 9.42 | 43.16 | 52.38 |
| A1_xor | 8.69 | 7.31 | 15.82 |
| A1_bmma | 8.76 | 27.87 | 36.41 |
| A1_tensor | 9.49 | 10.25 | 19.46 |
| A8_sharedlut | 20.38 | 22.52 | 42.91 |
| A1_sharedlut | 21.79 | 22.63 | 44.24 |

## Published LUT-GEMM and coverage

The official [LUT-GEMM FP16 kernel](https://github.com/naver-aics/lut-gemm/blob/master/lutGEMM/src/cuda/kernels/mv_fp16.hpp) was downloaded, compiled and run on the5080 with one binary basis. Six small launch configurations were screened per shape. The kernel computation was unchanged; unused host wrappers were removed. FP16 inputs/scales and pretransposed weights were already resident. Output clearing and LUT construction are included. These are **different numerical semantics**, not equivalent A8/A1 pipeline timings.

| Published FP16 LUT kernel | Best1-position µs | Best7-position µs |
|---|---:|---:|
| 9728 × 2560 | 4.69 | 27.16 |
| 2560 × 9728 | 4.83 | 29.00 |

On sampled random outputs, relative RMS error versus FP32 accumulation is 0.081–0.216%. This is a sanity check, not model-quality qualification.

The exact native adaptation changes table/accumulator storage to FP32, feeds integer activation codes, applies F32 row/activation scales afterward, and batches positions through a grid dimension. Integer sums remain exact in these shapes. It includes a per-call weight transpose, output clear and separate scale kernel. It is a small screening adapter, not an optimized LUT-GEMM port. Apache2 license and notices are preserved.

| Search lead | What was actually screened |
|---|---|
| Retained Tensor Core execution | Actual native default; A1 adapter with identical signs/scales |
| TCBNN / APNN-TC / CUTLASS binary MMA | Underlying binary matrix primitive and existing bit-plane mechanisms, compile/disassembly, exact correctness, native serving. Full CNN packages were not installed; their whole networks are not our drafter. |
| PrismML / Microsoft BitNet DP4A family | Existing native packed-sign DP4A, including token7 weight reuse. Their training recipes, formats and full products were not benchmarked. |
| LUT-GEMM | Published FP16 GPU kernel plus exact A8/A1 native adaptation; also existing global LUT4 |
| FluxBin | No runnable public code at the checked advertised repository; not tested |
| BitBLAS | Library not installed or benchmarked; search did not establish a distinct available SM120 binary arithmetic route beyond the tested families. This screen does not rule out a better BitBLAS-generated kernel. |
| FLUTE / T-MAC / FIGLUT | No compatible ready-made binary CUDA drafter route from search: documented FLUTE2/3/4bit, T-MAC CPU/NPU, FIGLUT accelerator design. Not tested. |

## Recommendation

Keep integer Tensor Core expansion as the default engineering path while deciding architecture and activation precision. Preserve W1A1 XOR/popcount as a shape-specific candidate, particularly for single-position and wide down projections; do not assume a whole-draft speedup. Shared LUTs merit another look only if the architecture makes single-position cost or prepacked layouts especially valuable. No newly discovered Blackwell binary instruction shortcut was demonstrated.

## Reproduction, provenance and release

Raw run: `runs/binary-methods-20261104-01/`. Final evidence: `throughput-final/`, `trace-final/`, `native-trace-final.sqlite`, `native-trace-final.nsys-rep`, `kernels.json`, `lutgemm.json`, `native-lut/build.json`, and `native-bmma-sass.txt`. Config/model/target hashes are frozen in run receipts; source snapshots and the final evidence manifest are retained there. Earlier seven-arm pilot results are excluded from the comparison; all nine final arms were rerun under one frozen runtime after adding the LUT adapter.

Scripts and native patch: `scripts/binary_methods/`. The runtime is an isolated relink of existing pinned objects with only the experimental CUDA translation unit replaced. The baseline libraries, model files, target/verifier code, main submodule and training were not changed. A1 override is a test-only environment switch; it must not be treated as an exported model format.

No occupancy/bandwidth utilization percentage is claimed: Nsight Compute hardware counters were denied in the prior screen. No further driver-permission changes were attempted. All final supervisors and servers are reaped; GPU is empty. Cleanup/retained-byte inventory is in the raw run. No push.

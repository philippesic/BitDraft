# Quick broad web search: binary execution on Blackwell

November4,2026. Discussion/research only; no downloads of models, GPU runs,
installations or candidate changes. Binary quantization is the objective; pure
binary arithmetic is optional. User has not selected W1Ax versus W1A1.

## Findings

- LUT-GEMM is the strongest available alternative mechanism to investigate.
  Its CUDA kernel builds shared-memory tables of signed activation sums, then
  uses binary weight patterns as addresses. Floating activations are supported;
  it is not W1A1 or XOR/popcount. Official build instructions target SM80, not
  validated SM120. Portability and speed on small draft blocks remain untested.
  [Code](https://github.com/naver-aics/lut-gemm)
  [Kernel](https://raw.githubusercontent.com/naver-aics/lut-gemm/main/lutGEMM/src/cuda/kernels/mv_fp16.hpp)
- Binary matrix/bit-plane algorithms already exist: TCBNN for Turing and APNN-TC
  for Ampere, plus CUTLASS primitives. However current CUTLASS sm80 binary-AND
  wrappers gate at architecture<=900 and XOR at<=890, excluding SM120. This is
  evidence against assuming ready-made Blackwell support, not proof every PTX
  binary form is unavailable. Our old5080 functional success did not establish
  SM120 BMMA machine instructions: the linked report's disassembly was SM75.
  Inspect generated SM120 instructions before calling that native binary hardware.
  [TCBNN](https://github.com/pnnl/TCBNN)
  [APNN-TC](https://github.com/BoyuanFeng/APNN-TC)
  [CUTLASS gates](https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/mma_sm80.h)
  [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- Microsoft BitNet's official GPU kernel is W2A8 GEMV with packed-value expansion
  and DP4A; its documented timings are A100. Useful engineering reference, not
  a new packed-bit arithmetic route.
  [Official GPU README](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)
- PrismML's public Q1_0 CUDA vector-dot implementations inspected use DP4A after
  sign extraction/expansion, including an activation-layout variant. Do not
  infer secret XOR/popcount GPU execution from proprietary quantization training.
  This inspection does not characterize every Bonsai format/backend.
  [Source](https://github.com/PrismML-Eng/llama.cpp/blob/prism/ggml/src/ggml-cuda/vecdotq.cuh)
- FluxBin proposes newer binary-basis LUT algorithms with scale fusion. Paper
  evaluations are A100, not Blackwell; its advertised GitHub URL returned404
  through web and GitHub API checks. Research lead, not reusable code verified here.
  [Paper](https://arxiv.org/html/2608.15602v1)
- BitBLAS supports binary-weight mixed precision but its published support matrix
  does not establish a5080 binary-arithmetic advantage. A Blackwell-named fork
  explicitly calls its adaptation a bypass for Orin/Thor, not a true adaptation.
  [BitBLAS](https://github.com/microsoft/BitBLAS)
  [Fork](https://github.com/wzqvip/BitBLAS-blackwell)
- FLUTE uses lookup-table dequantization with matrix multiplication, not the same
  partial-sum LUT method as LUT-GEMM. Its documented support is2/3/4-bit, not a
  ready-made binary5080 solution. T-MAC is relevant LUT prior art for CPU/NPU,
  not a CUDA5080 implementation. FIGLUT/FIGFLEX are accelerator-design work.
  [FLUTE](https://github.com/HanGuo97/flute)
  [T-MAC](https://github.com/microsoft/T-MAC)
  [FIGLUT](https://arxiv.org/abs/2503.06862)

## Recommendation and coverage

Do not reinvent lookup-table kernels or binary bit-plane tiling from scratch.
Inspect/adapt available LUT-GEMM and APNN/TCBNN ideas, while retaining the existing
integer Tensor Core implementation as the measured comparator. First establish
what the SM120 compiler actually emits for the old binary-MMA path. No discovered
source establishes an off-the-shelf speed win over our retained DSpark kernels.

Search covered official NVIDIA PTX/CUTLASS, BMMA/SM120/XOR/popcount, mixed-precision
libraries, binary/ternary LLM kernels, LUT methods, older binary CNN implementations
and recent2026 papers. Primary README/source checks distinguished mechanisms and
hardware. Blackwell-specific searches and older/adjacent vocabulary converged on
the same families; no exhaustive absence claim. Some GitHub PR pages timed out,
FluxBin code was unavailable, and one OpenReview PDF hit a browser challenge.
No external speedup ratios are transferred to our GPU or drafter workloads.

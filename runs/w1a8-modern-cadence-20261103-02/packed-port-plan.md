# Modern packed W1A8 port: source findings

Read-only source investigation, November 3, 2026. No model load, hash sweep,
build, GPU work or implementation occurred. Parent owns the GPU lane.

## Recommended first executable route

Port the existing historical **EAGLE packed loader and linear wrappers** onto
the retained modern EAGLE graph; do not port the DSpark model or use expanded
Q8 weights. The historical code already supports all nine linears, strict GGUF
admission, target-borrowed F32 `ggml_get_rows`, shared A8 packing and a separate
ordinary dense path. This matches the current training adapter/exporter much
more closely than DSpark. The shortest complete training/export/runtime route
is therefore nine row-scaled binary projections with fixed A8 quantization.
This is an implementation recommendation, not evidence that full binarization
will retain enough quality or beat DFlash.

An FFN-only candidate is a smaller *runtime* edit and plausibly preserves more
of the ordinary model's quality. It is explicitly mixed precision, and needs a
new training-adapter admission path: `NativeStepAdapter` currently rejects any
of the nine projections that is not `RowBinaryLinear`/`GroupedBinaryLinear`,
and requires a common type/activation contract. `export_recurrent_binary.py`
also emits all four groups. Do not describe an FFN-only fixture as an admitted
full W1A8 checkpoint. Historical loader group selection can support it after
an appropriate narrow training/export extension.

## Exact reusable interfaces

Historical source root:
`runs/shortlist-throughput-20261027/native/source/`.

- `src/models/eagle3.cpp`: `load_arch_tensors`, local `load_w1a1_linear`, and
  the `eagle_linear` wrappers in both `graph<true>` and `graph<false>`.
  The loader implements `eagle3.w1a1.version` v1/v2/v3, named groups,
  packed/scales companions, per-tensor logical-K metadata, no dense shadows,
  row/group scales and selected dense exceptions. Use only the fixed row-A8
  subset initially; learned thresholds, affine midpoints, fusion correction,
  dense diagnostics, KV-only catch-up and historical ambient optimizations
  are unnecessary dependencies for a first executable model.
- `ggml/include/ggml.h`: `ggml_w1ax_pack`, `ggml_w1ax_mul_mat`,
  `ggml_w1ax_mul_mat_shared` (and their learned variants only if retained).
- `ggml/src/ggml.c`: implementations around lines3370–3462. Matmul uses
  `GGML_OP_W1A1_MUL_MAT`; a shared code pack uses `GGML_OP_W1AX_PACK`.
  Matmul source0=I32 signs, source1=F32 row scales, source2=F32 activations,
  optional source3=shared packed activations. Output is F32 `[M,N]`.
- `ggml/src/ggml-impl.h`: `ggml_w1ax_pack_layout`. A8 uses
  `ceil(K*N/4)` I32 words holding token-major signed int8 codes, followed by
  one F32 activation scale per token; no bit planes. Weight signs are I32
  `[ceil(K/32),M]`, little bit order, bit1 meaning nonnegative, and row scales
  are F32 `[M]`.
- `ggml/src/ggml-cuda/w1a1.cu` / `.cuh`: `ggml_cuda_w1ax_pack` and
  `ggml_cuda_w1a1_mul_mat`; `w1ax_quantize` computes per-token F32 absmax,
  beta=absmax/127, round-to-nearest-even signed codes in[-127,127], zero
  vector -> zero codes. `w1ax_integer_dot` consumes packed weight signs
  directly and can use DP4A. This is real packed binary storage/execution;
  sign expansion in registers is not an expanded Q8 model publication.
- `w1a8_algorithms.cuh` is only needed for optional seven-token DSpark
  algorithms; `imma_mix` is explicitly gated on N==7, so it is **not** a
  ready speed optimization for the ordinary EAGLE serial decode path N==1.
  First port should use the generic packed integer-dot/DP4A path and measure.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: include, compute dispatch and backend
  `supports_op` cases for both ops. GPU support assertions include canonical
  contiguous source layout, F32 activations/scales, I32 signs, bits8.
- CPU fallback/reference: `ggml/src/ggml-cpu/ggml-cpu.c` implements
  `ggml_compute_forward_w1ax_pack` and `ggml_compute_forward_w1a1_mul_mat`,
  compute/work-size cases; `.cpp` supplies supports-op cases. Keep this
  for bounded integer-dot correctness fixtures rather than silently falling
  back to CPU in performance evaluations.
- `ggml/src/ggml-cuda/CMakeLists.txt` explicitly compiles `w1a1.cu` with
  `--ftz=false`; preserve this narrow override if exact quantizer edge
  handling is retained. Avoid copying unrelated historical common-header
  instrumentation: modern `common.cuh` does not define old
  `ggml_cuda_eagle_scope` / `ggml_cuda_matmul_audit` helpers used by that file.

## Modern model/graph edits

Modern root: `runs/spec-shortlist-20261030/runtime/`.

1. Add only required packed/scales pointers to `llama_layer` and `llama_model`
   in `src/llama-model.h`; borrow declarations from historical counterpart.
   Add head tensor-name entries to `llama-arch.h/.cpp` if using its explicit
   head enums; other projections can use `LLM_TN_IMPL(...,"w1a1_packed",...)`.
2. Port strict loader subset into modern `src/models/eagle3.cpp`. Dimensions
   come from current hparams, not historical Qwen3 defaults: H2560, FF7680,
   Q16, KV4, head160, rotary40, target taps[2,16,29], draft head50000 rows.
   Omit private token embeddings so decoder borrows exact target `get_rows`.
3. Encoder: replace fusion `build_lora_mm(model.fc,cur)` with packed-or-dense
   wrapper, retaining modern output bookkeeping.
4. Decoder: replace Q/K/V projection calls with that wrapper. For packed O,
   call `build_attn` with null output projection, then apply packed O to its
   result (historical code already does this). Keep current partial RoPE,
   F16 KV, GQA, norms and residual ordering.
5. Replace only packed FFN path: shared pack for normalized input, packed
   up/gate, SiLU(gate)*up, second pack and packed down. Retain dense
   `build_ffn` when no packed tensors are declared. Replace the draft head
   matmul before the existing d2t expansion. Preserve all server/verifier
   behavior and target-only paths.
6. Freeze A8 in model metadata. Historical code reads `GGML_W1AX_ACT_BITS`
   and rejects a mismatch against metadata; the first modern launcher must
   explicitly pin8 and reject contamination. A model-owned field is preferable
   once implemented; do not silently default to A1.

## Export traps to fix before admitting a model

`scripts/export_recurrent_binary.py` already emits packed I32 signs, F32 row
scales, strict complete per-projection metadata, no dense shadows and native
norms. But its Q/K row permutation and the standalone
`scripts/audit_eagle_w1a1_gguf.py` **hardcode Q32/K8 and full-head rotation**.
Those are incorrect for modern Q16/K4 and rotary40of160.

Use `perm_indices(head=160,rotary=40)` from the repaired ordinary exporter
`experiments/qwen35-correctness-matrix-20261101/repair_eagle_export.py`.
For HF-order latent tensors, apply that prefix permutation independently in
each head to latent signs **and row scales** before packing; leave channels
40..159 unchanged. Do not call `repair_rows` directly on HF tensors: that
function first undoes the old fully-permuted GGUF layout. Carry repaired
`eagle3.rope.dimension_count=40`, theta1e7, native taps and exact draft-vocab
mapping into the base GGUF. Keep original exports intact. CPU serialization
audit must inspect every packed word and scale under the new dimensions.

## Runtime compatibility and unchanged controls

Historical runtime libraries are llama0.5 / GGML0.25.1; modern ones are
llama0.8 / GGML0.28.0. Old custom CUDA is not a drop-in modern backend.
The modern immutable quant4group CUDA module contains neither packed op.
The generic modern `GGML_OP_CUSTOM` API does not provide a ready CUDA custom
kernel dispatch in this module.

An in-graph packed port therefore needs a separately built **candidate**
CUDA/core/runtime from the modern source, carrying the exact frozen
quant4group correctness changes plus the two new ops. Preserve existing enum
values by appending new ops immediately before `GGML_OP_COUNT`, update name/
symbol tables and assertions, and do not copy the whole historical enum.
Never overwrite or relabel the selected frozen module/baseline receipts.

Qualify an ordinary target-only and ordinary drafter under the candidate
runtime against the frozen runtime on the same development requests before
attributing throughput to W1A8. Measure an unchanged ordinary arm in each
runtime as an additional diagnostic and keep the official DFlash baseline
frozen. If the project requires byte-identical frozen CUDA for both arms,
this in-graph route cannot satisfy that constraint; a separately implemented
native custom backend/drafter boundary would be needed instead. That is more
work and is not already available in these sources.

## Minimal execution evidence

- Tiny real-CUDA op fixtures for A8 pack/DP4A at K2560,5120,7680 and encoder
  K7680, N1 plus a small prompt chunk. Compare exact integer dots, F32 scales
  and final products with the CPU/reference contract; include zero rows,
  RNE ties and invalid metadata/layout rejection. Test modern output widths
  including50000 in bounded chunks. CPU tests alone do not prove GPU use.
- An actual nine-linear GGUF from a retained modern checkpoint, exported with
  prefix-only Q/K permutation, loaded by the candidate native server; audit
  all nine dispatch sites, packed byte counts, absence of dense shadows and
  actual loaded library maps. A fresh quantized initialization is only a
  model-admission control, not a trained continuation or quality win.
- Bounded actor/native drafts using the same captured native target features,
  borrowed F32 embeddings, positions and cache histories. Investigate label/
  drafter-decision-changing errors; floating drift alone is not a stop gate.
- Actual trained/exported candidate full requests versus current target-only
  and frozen DFlash correctness controls, exact token/text/stop audit,
  independent quality and five alternating repetitions. Untouched inputs
  remain sealed until the selected protocol/candidate are frozen.

## Storage estimate and uncertainty

Bounded `du` observations: modern `src`4.9MiB, modern `ggml/src`27MiB,
modern retained build97MiB, historical retained build57MiB. Relevant custom
CUDA files are ~39KiB +8.5KiB + small group/header dependencies. New source
overlay should be well below1MiB; do not copy model weights or a repository's
unrelated runs. Source-only isolated checkout is tens of MiB if shared Git
objects are used, but a full checkout includes other tree content and was not
measured here.

Theoretical all-nine packed weights are29,107,200 bytes plus307,520 bytes of
row scales, ordinary norms, maps and GGUF metadata: approximately30MiB,
with no private target embedding. FFN-only sign payload is7,372,800 bytes,
but its preserved dense fusion/head/attention dominates total model size.
Latent+Adam state for all nine is roughly2.6GiB before other state when all
latents are trained; a compact native export does not imply a compact full
training checkpoint.

The97MiB retained build is **not** a reliable peak-space bound for a clean
CUDA build: template object files/compiler temporaries may have been retired.
Plan a measured isolated build with an explicit storage floor and limited
SM120 targets; reserve at least1–3GiB as an initial operational allowance,
and stop before violating the floor if actual peak exceeds it. No clean-build
peak or compile duration was measured by this source-only task.

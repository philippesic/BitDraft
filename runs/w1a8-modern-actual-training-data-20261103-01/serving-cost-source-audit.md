# Serving catch-up cost source audit

**Finding: yes. The current native EAGLE decoder computes the packed 50,000-row LM head and expands its logits to the full 248,320-token target vocabulary for every catch-up input row, even when every batch output flag is false.** Zero output flags suppress host extraction and sampling; they do not currently suppress this graph branch. This is a concrete serving-cost risk independent of whether training improves acceptance. It is not established that the head alone explains the complete observed prefill delay.

This was a bounded CPU/source/log inspection. No build, GPU/model execution, source edit, new optimization screen or benchmark was performed. The only output is this report. Paths below are relative to `/home/philip/binary-eagle-decoding`; code references use the frozen packed runtime `source/` tree.

## Call chain and retained graph

1. `tools/server/server-context.cpp:4751–4766`: after successful native target decode, the server always calls `common_speculative_process(spec.get(), batch.view)`. The comment explicitly says it currently always re-evaluates drafted tokens for simplicity.
2. `common/speculative.cpp:654–733`: EAGLE `process()` copies the target layer features, runs the feature encoder and obtains a fused embedding for every row. This is distinct from the decoder LM head; the encoder graph (`src/models/eagle3.cpp:229–253`) ends at its feature projection and has no LM head.
3. `common/speculative.cpp:746–788`: catch-up shifts pairs to `(token[k+1], g[k])` at position k, supplies the cross-ubatch boundary pair, and calls `llama_process(...DECODE...)` with **all output flags false**. It defers the final boundary until `draft()`. Under the uninterrupted single-sequence prefill geometry, a P-token prompt therefore writes P−1 catch-up decoder rows. The last row is subsequently evaluated with output=true for actual drafting; it is required work, not an extra unused catch-up row.
4. `src/llama-context.cpp:4769`: DECODE dispatches to `ctx->decode`. At lines2007–2019, the context counts the output flags and sets `n_outputs=0` before building the zero-output ubatch graph. `graph_params` passes that value (line2729). The batch allocator (`src/llama-batch.cpp:200–244`) copies explicit false flags and counts zero outputs; it does not silently force the last token true. In this context `embeddings=false`; `embeddings_nextn=true, masked=true` is a separate flag and does not promote all rows to output.
5. **The EAGLE decoder ignores the output selection for its tail.** `src/models/eagle3.cpp:295–314` creates token/features inputs with n_tokens columns. There is no `build_inp_out_ids`/`ggml_get_rows` selection of requested output columns anywhere in the decoder. At lines417–438, it marks the full prenorm state as t_h_nextn, normalizes all columns and calls `project(output, output_w1a8, cur)` unconditionally. The packed branch is genuine W1A8, not a fallback to a full floating-point weight matrix.
6. `src/models/eagle3.cpp:440–456`: its local variable named `n_outputs` is **cur->ne[1]**, i.e. the full input-column count here, not the context's zero requested-output count. It allocates/fills `[1,248320,N]` with −infinity, scatters `[1,50000,N]` mapped draft logits through d2t and reshapes to `[248320,N]`. The result is assigned to res->t_logits.
7. `src/llama-model.cpp:3386–3406` always invokes `llm->build_logits()`. `src/llama-graph.cpp:4150–4159` skips the LM head only for pooled embeddings; otherwise any non-null t_logits is passed to **ggml_build_forward_expand**. There is no n_outputs==0 guard. This makes the head/fill/scatter explicit graph roots, so they survive graph construction. `ggml.c:7389–7427` recursively marks their source ancestry COMPUTE; `ggml_build_forward_expand` at line7490 requests compute=true. They are not dangling unused tensors that later lack a consumer.
8. Backend sampling does not rescue this branch: `src/llama-graph.cpp:4173–4231` finds zero active sampling rows, but a configured sampler selects `outs[0]=res->t_logits` when `i_out=0`. Inactive sampling chains can be non-compute, while the logits root remains selected.
9. `src/llama-context.cpp:1588` submits that graph for compute before its output-copy decisions. The logits host-copy guard at line2071 requires n_outputs>0, as does masked nextn extraction at lines2151–2157. Those later guards cannot remove already submitted graph work. `ggml_set_output` affects retention/allocator behavior; lack of a host read is not a graph-pruning instruction.

## Executed dimensions versus reserved capacity

For a catch-up decoder ubatch of N rows, source-derived shapes are:

| Operation | Actual shape/work implied by that graph |
|---|---|
| Output-norm/activation packing | N vectors, width2560 |
| Packed LM projection | K=2560, M=50000, N=actual decoder input rows; F32 result `[50000,N]` |
| Full vocabulary fill/scatter | F32 `[248320,N]`, mapped 50000 rows per column |
| Context-requested outputs / host copies | zero / no logits or masked prenorm host extraction |

`ggml-cuda/w1a1.cu:87–114` uses n=activation.ne[1] and m=dst.ne[0], launching `w1ax_integer_dot` on a `(ceil(M/4),N)` grid, four output rows/block, one warp/row. There is no output-flag argument. The CUDA dispatcher handles W1AX_PACK and W1A1_MUL_MAT at `ggml-cuda.cu:2606–2611`; the offload rule at 6317 explicitly prefers these genuine CUDA operations. Thus for N=508 this head has65.024 billion integer products,6.35 million CUDA blocks,96.89 MiB of draft logits and481.21 MiB of full-vocab fill storage. These are arithmetic/storage counts, **not measured latency or a kernel profile**.

A559-token prompt ordinarily entails558 catch-up rows in total, split according to actual server ubatches/checkpoints, not necessarily511+47. Summed head work would be71.424 billion integer products, with528.57 MiB of full-vocab rows written across those calls; that summed storage is not a simultaneous peak. The server can split its target prefill around checkpoints; preserve actual batch history when diagnosing it.

The existing prompt04 debug log contains draft-context `n_outputs_max=512`, reserve graphs with n_tokens=512/n_outputs=512, and 596.41 MiB CUDA compute-buffer reservation. Those are **capacity/reserve observations**, not a trace that512head columns ran on every request. The same context initially reports only 0.95 MiB CUDA-host output buffer (one full-vocabulary F32 row). `output_reserve` uses max(actual requested outputs, sequence count) at `llama-context.cpp:2249–2276`; that host buffer is not a512-row host copy. The actual head width follows N in the architecture independently of these requested-output capacities. This explains why reserve logs alone cannot establish executed matrix dimensions; the call chain above supplies the missing distinction.

## Existing native observations, diagnostic only

From `runs/w1a8-modern-native-admission-20261103-01/*/response-04.json`, the same559-token old smoke request had:

| Arm | Native prompt_ms | Native predicted_ms | Accepted/proposed |
|---|---:|---:|---:|
| Frozen target-only |109.863|360.455|—|
| Candidate-runtime target-only |90.813|359.548|—|
| Frozen DFlash Q4_K_M |259.383|264.025|38/62|
| Candidate-runtime DFlash Q4_K_M |258.507|262.532|38/62|
| Candidate-runtime ordinary EAGLE Q4_0 |1951.395|328.837|36/66|
| Candidate-runtime W1A8 initialization |1832.819|627.763|9/120|

These are single smoke/native-component timings, with startup/cold-cache/context differences and no repeated full-request timing protocol; they are **not a performance claim or a denominator for final speedup**. Ordinary EAGLE has similarly expensive prefill, so this observation does not isolate the packed head kernel as the sole cause. It supports inspecting the common EAGLE process/catch-up path, including full decoder work, feature copies, attention/KV and checkpoints.

`runs/w1a8-modern-serving-trace-20261103-01/prompt-04/server.log` separately records 559 prompt tokens, 2289.44prompt_ms, 581.324predicted_ms and 9/120 accepted/proposed with debug logging. Its logs around lines5650–5702 show checkpoint/prefill boundaries at 47 then 555 then 559 tokens; a queue-yield interval spans roughly 1.344–3.159seconds during the middle processing, but does not label individual GPU operators. There is **no existing zero-output executed-node shape dump or per-op profile** in this evidence. Do not convert that wall interval into a measured head time. The trace's reserve-node input shapes are explicitly reserve-time output, not executed-node profiling.

## Implication and next bounded validation

Better accepted continuations can coexist with poor full-request throughput because mandatory target-feature catch-up currently pays for unused vocabulary projection/expansion before sampling. Preserve this as a phase-one serving generalization/cost issue; do not mark any backlog optimization as tested or rejected from this audit.

A future isolated diagnostic should first record actual EAGLE process ubatch token/output counts and the expanded graph's head/fill/scatter node dimensions, then separately time catch-up versus true draft steps with instrumentation excluded from request comparisons. A source change that selects requested output rows or omits a zero-output tail would need to preserve all K/V writes, shifted/deferred boundary semantics, verifier/replay behavior and real sampled prenorm states; removing mandatory context work on the strength of zero host outputs is unsafe. Native parity/trajectory and the unchanged full-request controls would be required afterward. This report implements none of those changes and predicts no speedup size.

## Source snapshot pins

Small source files only were hashed; no model/checkpoint was opened. The native admission and serving-trace freeze/build/loaded-library records remain the binary execution provenance. The inspected source snapshot SHA256s are:

- `runs/w1a8-modern-packed-runtime-20261103-01/source/common/speculative.cpp`: `667f3e71026a4701d4c3e6338ca25ba38b7f83d6a41ae9f9c33763beb12a64ad`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/src/models/eagle3.cpp`: `f459736f08ba5207864616ad47cdd5c811516c90fbef064b9997e69417288993`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/src/llama-model.cpp`: `0428a8e43eed0a11470ac12e55fde8d6d9c76d25d31dc9ec597e4f2df13e92d2`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/src/llama-graph.cpp`: `d72fa58939970f16372d36e1a401986a7b44ad1569973080467191933f9b5c74`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/src/llama-context.cpp`: `63645057d8f017d651544499fc1f3bcff5997cf25270b06479558ebfb5f9f3d2`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/ggml/src/ggml.c`: `0a442e6b9da2576e98ad8753b022ea689d0cf3752ff7c7287f07cd51ec9a8c54`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/ggml/src/ggml-cuda/w1a1.cu`: `9af2c0df0adab935c335aeac65628c5f54a1529eb6a4da40245f23848d11355e`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/ggml/src/ggml-cuda/ggml-cuda.cu`: `673d11deaa2c8c0c8bff65102552b4de59eaa6dc25cb7b4aa89c9fa60af6af03`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/src/llama-batch.cpp`: `bdaa2cea85556483d008a214bd20b8d8c3a851a95e26a1ec7fc0151b1b5ced25`

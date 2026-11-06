# Smallest correct EAGLE output-tail change — source advice only

**Recommendation: first implement the EAGLE-local head-only skip, preserving full prenorm computation and every existing K/V root.** Add selected-output gathering to the positive-output tail using the existing output-ID helpers so masked nextn rows match the logits. Do not begin with a K/V-only catch-up rewrite. That rewrite is mathematically valid for the current single-layer serving contract, but has additional cache/layout/instrumentation obligations and is a separate change to isolate.

No source edit, build, model/GPU execution, tests or large hashing was performed during the parent's 480-request quiet measurement phase. This report contains proposed hunks only. Preserve the existing native timing/source freeze; any later candidate runtime is a new pinned comparison, not a relabeling of those measurements.

## Minimal head-only skip

At the end of `source/src/models/eagle3.cpp::graph<false>`, immediately after publishing the existing full prenorm state and before `model.output_norm`/head/d2t, this is sufficient for the zero-output case:

```cpp
cur = inpL;
ggml_set_output(cur);
res->t_h_nextn = cur;

if (n_outputs == 0) {
    // Keep the full original decoder/prenorm and all existing K/V writes.
    // Output marking alone does not add a compute root.
    ggml_build_forward_expand(gf, cur);
    return;
}

// Existing output norm, packed/dense LM head, d2t fill/scatter follow.
```

The explicit forward expansion is essential. Merely returning after `ggml_set_output` does not guarantee the prenorm/O/MLP branch is computed; the existing attention helper already added some independent Q/K/V/cache roots. This variant retains the original full decoder arithmetic and `t_h_nextn` shape for all-false catch-up while making `t_logits` remain null. It requires no changes to generic graph/context code. It does not skip encoder feature projection or required shifted/deferred-boundary work in `common/speculative.cpp`.

## Positive output selection and masked nextn

The current decoder publishes all prenorm rows regardless of output flags. For a batch with output flags `[false,true,false,true]`, masked nextn extraction currently copies the **first** two prenorm rows, while callers expect rows1and3. This matters beyond performance: narrowing only the LM head but leaving masked `t_h_nextn` full would preserve that existing selection error. The normal one-row draft call hides it.

A complete local replacement for the old prenorm assignment is:

```cpp
cur = inpL;

if (n_outputs == 0) {
    ggml_set_output(cur);
    res->t_h_nextn = cur;
    ggml_build_forward_expand(gf, cur);
    return;
}

// Positive outputs only: avoid introducing a zero-length unused input.
auto * inp_out_ids = build_inp_out_ids();
if (crop_before_nextn(inp_out_ids)) {
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
}
ggml_set_output(cur);
res->t_h_nextn = cur;

if (crop_after_nextn(inp_out_ids)) {
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
}

// Existing output norm/head/d2t path, with n_outputs columns.
```

Use the existing helpers from `src/llama-graph.h:1087–1094`: crop-before is true when nextn is disabled or masked; crop-after is true for unmasked nextn. This keeps the prenorm before the model output norm, which EAGLE recurrence expects. The encoder `graph<true>` remains unchanged and continues to produce every fused input-feature row.

| Requested rows | nextn mode | Published decoder prenorm | Head/vocabulary columns |
|---|---|---|---|
| zero | masked | all original prenorm rows retained internally; no extraction | none |
| zero | unmasked | all original prenorm rows, extract all N | none |
| positive M<N | masked | exactly M selected rows in input/output order | M |
| positive M<N | unmasked | all N original rows | M |
| all N | either | all N, identity output selection | N |

`build_inp_out_ids()` sets indices from actual `ubatch.output` flags (`llama-graph.cpp:269–299`), preserving exactly the order used by backend sampling and `output_ids`. The standard always-positive gather includes an identity gather for all-output batches. A no-gather all-output special case can be considered only with its own reuse/shape check; the existing helper intentionally uses a stable output-ID mechanism. Gathered tensors feed the existing `project` helper, which already makes noncontiguous activations contiguous before packed projection.

## Why generic code need not change

- `llm_graph_result::reset` sets all result pointers and sampled vectors to empty. A fresh zero-output EAGLE graph therefore has null t_logits, not stale previous logits.
- `llama_model::build_graph` calls `build_pooling`, `build_logits`, `build_sampling`, then `set_outputs`. `build_logits` and `build_sampling` already tolerate null t_logits; `set_outputs` separately marks non-null t_h_nextn. Because this proposal explicitly expands prenorm, the graph remains nonempty even without a head.
- `process_ubatch` allocates/submits the resulting graph normally. Logit extraction requires t_logits and n_outputs>0. Masked nextn extraction uses n_outputs rows; unmasked uses ubatch.n_tokens. The hunk supplies the required shape in each case. Existing encode extraction and output buffer reservation are unchanged.
- Backend sampling's graph construction returns immediately for null logits. There are no actual output tokens to sample in that zero-output transaction. `decode` still begins its sampling transaction; the next positive-output graph creates/resets active backend chains normally. Verify this transition with a real sampler; do not invent dummy logits or dummy sampled results for the catch-up graph.
- `llm_graph_params::allow_reuse` (`llama-graph.h:839–909`) rejects different n_outputs, gtype, nextn flags and sampler/output masks. Thus a zero-output prenorm-only graph cannot be incorrectly reused as a positive-output sampling graph. `llm_graph_input_out_ids::can_reuse` also checks n_outputs and repopulates the selected indices for each batch. Existing graph reserve with positive outputs remains a capacity path; it must not be used to assert zero-output executed work.

## Is K/V-only catch-up safe?

**Mathematically yes under the current one-layer EAGLE serving preconditions.** Its K and V come directly from the concatenation of normalized borrowed token embedding and fused external target feature. Neither depends on the current row's Q, attention output, residual or MLP. Future rows receive their own external target features; they do not consume a previous catch-up row's computed prenorm. Only real `draft()` output rows use prenorm autoregressively. `NativeStepAdapter.build_context_cache` (`src/w1a1_eagle/native_step.py:544–600`) already implements that one-layer dependency with F16-exact cached K/V, excluding the deferred seed.

However, **do not implement K/V-only by calling `build_attn` and assuming Q disappears**: `llama-graph.cpp:3185–3206` explicitly expands Q, V, K and cache writes. A K/V-only graph needs an early branch before `build_attn`, explicit cache writes, and no prenorm/head root. Sketch after Kcur/Vcur reshape and K RoPE, before existing build_attn:

```cpp
const bool kv_only = n_outputs == 0 && !cparams.training &&
        (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);
if (kv_only) {
    if (inp_attn->self_k_rot[il]) {
        Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot[il]);
    }
    if (inp_attn->self_v_rot[il]) {
        Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot[il]);
    }
    ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(
            ctx0, Kcur, inp_attn->get_k_idxs(), il));
    ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(
            ctx0, Vcur, inp_attn->get_v_idxs(), il));
    // No masked rows requested, so no prenorm may be retrieved here.
    return;
}
```

This is a dependency sketch, not an implementation-ready patch without validation. Existing Q tensors created before this branch would remain unexpanded and should not execute; shared activation packing used by K/V must remain identical. Preserve the native copy helpers rather than writing F16 cache arrays manually: those helpers own layout, slots, sequence positions, rotations and dtype conversion. Never bypass the pending-boundary/rollback bookkeeping in `common/speculative.cpp`. The one-layer assertion must remain; multi-layer decoders generally require intermediate attention/MLP outputs to construct later-layer K/V.

There is a precedent in `src/models/dflash.cpp:956–1014`: K/V-injection graphs explicitly rotate/cache K/V without attention. `llm_graph_input_attn_kv::set_input` already skips an unallocated attention mask (`llama-graph.cpp:551–556`). Still, validate all input allocations, nonempty roots, memory-slot updates and reuse across transitions. An unmasked-nextn caller expects all prenorm rows even with zero logits, so it must fall back to the head-only variant; returning null would violate its contract.

K/V-only also removes trace-visible catch-up prenorm tensors. The current native actual-training-data recorder associates FC/prenorm producers with serving decoder states. A changed runtime would need a newly frozen recording contract and explicit optional-prenorm/short-record handling; do not reuse old tracing assumptions or manufacture zero prenorm arrays. This additional obligation is a reason to start with the prenorm-preserving head-only change.

## Validation after the quiet measurement finishes

No tests below were run in this advice task.

1. Graph tests/source checks: with zero outputs, require K/V-copy roots plus prenorm computation and no output norm, packed LM-head projection, full-vocab fill/scatter or backend sampling roots. With M selected outputs, require M-column head/full-vocab tensors and masked prenorm rows at exactly selected indices. Encoder graph must remain unchanged. Capacity/reserve logs do not satisfy these execution assertions.
2. Transition tests: same-length batches with output counts0→1→0→all, two different noncontiguous masks with the same count, encoder→decoder switches, masked/unmasked nextn, backend sampling on/off. Fail on stale logits, missing sampler state, unused unallocated inputs or incorrect output-index mapping. Use N≥4 with a selected final row, not just row0.
3. Native state gates: old/new runtime on the same frozen initializer, raw inputs, features, source/checkpoint and sampler. Compare exact K/V placement/content where the same native kernels permit it, meaningful prenorm/logit numeric gates and actual draft decisions. Exercise split prefills, the cross-ubatch bridge, deferred last row, checkpoint restore/partial acceptance and rejected-prefix rollback. Include the known559-token old smoke request and whole-request token/text/stop equality. CPU-only adapter agreement is diagnostic, not the native gate.
4. State-preservation test: trace every previously accepted native continuation and require all retained successes, while separately checking previously rejected proposals; do not confuse a conditional teacher replay with actual serving history. Preserve all source and checkpoint pins and the old timing evidence.
5. Only after those pass, conduct repeated quiet full-request timings with unchanged target-only/frozen DFlash Q4_K_M depth2 and untrained/trained controls. Distinguish a reduced catch-up cost from better trained acceptance. No speedup amount is established by this advice or a single smoke timing.

## Source pins

The EAGLE/graph/context/ggml source SHA256s and executed binary association are linked in `serving-cost-source-audit.md`; those files were not rehashed during the quiet phase. The two newly consulted small files below were hashed only for this source advice (no model/arrays):

- `runs/w1a8-modern-packed-runtime-20261103-01/source/src/llama-graph.h` (53090bytes): `0502b531fb610bc94ea0b398b0e7233de048aece69317b9e01d1a95398f271b8`
- `src/w1a1_eagle/native_step.py` (39322bytes): `b5fa92bbf7c64d1e2601b2938ec8f919a5c313e9eb18cb12cf5eb000002a667c`

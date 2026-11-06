# Confirmed EAGLE cross-ubatch bridge buffer alias

Read-only CPU/source audit, November 3, 2026. No source mutation, GPU call,
model/weight opening, helper change or delegation occurred. One existing raw
native trace was independently joined to its actual FC output rows. Original
36-group traces remain untouched.

## Finding

This is a concrete input-buffer lifetime bug, not representation drift, a
different layer boundary, or a mistaken global row offset. EAGLE attaches the
previous deferred FC row to a bridge token through a non-owning `common_batch`
view, then overwrites that row before the batch is materialized and decoded.
The bridge therefore receives the **current ubatch's final FC row**, although
its token and decoder/RoPE position still refer to the prior boundary.

The source path is present in both the candidate source and retained frozen
modern source. It was not introduced by the new packed W1A8 projection graph.
This audit does not modify either runtime or retroactively change results.

## Exact ownership and copy timing

All candidate paths below are under
`runs/w1a8-modern-packed-runtime-20261103-01/source/`.

1. `common/common.h:1089` defines `common_batch::token.embd` explicitly as a
   **non-owning view** of data supplied to `add_embd()/set_embd()`.
2. `common/common.cpp:2253`, `common_batch::set_embd`, only executes
   `tokens[idx].embd = embd`. No float data is copied there.
3. `common/speculative.cpp:767-768` creates the bridge row and calls
   `batch.set_embd(idx, {pending_g_last[seq_id].data(),1,n_embd_dec})`.
4. The same `process()` call then refreshes the deferred state at line783:
   `memcpy(pending_g_last[seq_id].data(), g_embd + end*n_embd_dec, row_bytes)`.
   The bridge view still points to this same allocation.
5. Only at line789 does `llama_process(..., batch.get())` materialize the batch.
   `common_batch::get()` calls `get_sub_batch(0,size())` in `common.h:1110`.
   `common.cpp:2278-2296` iterates the views and calls
   `llama_batch_ext_set_embd_token(res,idx,t.embd)`.
6. That public setter forwards to `llama_batch_ext::set_token_embd` at
   `src/llama-batch.cpp:1238-1269`, which finally copies the float range with
   `embd.insert(...,embd_in.data,embd_in.data+n_total)`.

The lower-level extended batch owns its copy correctly. The overwrite occurs
**before** that copy. Changing only the public setter cannot recover the lost
old values. This also explains why the direct native replay helper, which
constructs extended batches immediately from stable encoded arrays, did not
reproduce the actual server behavior.

## Independent exact raw-vector confirmation

Case: `fit-prose-00`, existing
`runs/w1a8-modern-actual-parent-capture-20261103-01/capture01/` trace.

- Decoder call7, batch row0: token248068, decoder position31. The decode batch
  positions are `[31,32,33,34]`.
- Its observed input vector is byte-identical to encoder call6 row3, associated
  with target call5 **target position35**. Observed SHA256:
  `939133f391c8de6dd1a11686595af995da1efddb20ccaa918f21187559bd48b3`.
- The correctly shifted FC row for target position31 is encoder call3 row31,
  associated with target call2. Its SHA256 is
  `823cbfd40f51046b85ef2ea6a5dcf6e47070e9d5796ec541b2bf517c191f7f84`.
- The observed row differs from that expected row; maximum absolute component
  difference is2.9786014556884766. It matches the newer row exactly, rather than
  approximately resembling it.

Target positions were recovered from the linked target call's original batch
rows, with a separate ENCODE cursor for each target call. They were not inferred
from encoder placeholder positions or a single cumulative file-row counter.
This independently confirms the reported31→35 example. The data worker's wider
36-group observation is consistent with the same source path, but this audit
did not redundantly rescan every raw vector in all36 groups.

## Minimum scoped repair

Keep an owned, per-call bridge copy alive through batch materialization:

```cpp
// Local to EAGLE process(), sized before the per-sequence loop.
std::vector<std::vector<float>> bridge_g(n_seq);
...
if (bridge_is_needed) {
    bridge_g[seq_id] = pending_g_last[seq_id];
    const int32_t idx = batch.add(first_token, pending_pos, seq_id, false);
    batch.set_embd(idx, {bridge_g[seq_id].data(), 1, (size_t) n_embd_dec});
}
// Existing refresh of pending_g_last can now occur without changing bridge_g.
...
llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
```

Do not resize or overwrite the copied inner vector after attaching its view.
One2560-F32 row per active bridge is enough; the local copy needs to survive
until `batch.get()` has copied it, and keeping it through the native call makes
the lifetime obvious. Do not globally change `common_batch` ownership merely
to fix this one caller. An alternative is deferring all pending-state refreshes
until after materialization, but that is a broader control-flow change.

## Scientific and runtime boundaries

The affected object is `common_speculative_impl_draft_eagle3`. The selected
DFlash baseline uses a different implementation; a local EAGLE bridge repair
does not require changing its frozen binaries/settings. Keep the official
frozen DFlash baseline, historical F16 results, old candidate build and original
actual-parent traces intact. A repaired candidate needs a new source/build pin.

The old traces still document actual emitted successes and their input vectors
under the old protocol. They must not be relabeled as correctly shifted
feature31 examples, silently rewritten, or treated as proof of the repaired
protocol. In particular, rebuilding an ideal shifted CPU cache from F31 cannot
reproduce a native cache entry that actually consumed F35. This is a material
training/serving feature-distribution mismatch independent of small FP32 dot
rounding differences.

For the repaired candidate, use unchanged model weights and the same frozen
inputs to verify that the bridge vector now equals the prior deferred FC row.
Then compare ordinary EAGLE as well as packed EAGLE before/after the protocol
repair, so a protocol fix is not attributed to W1A8 training. Recollect or
re-audit actual serving parent histories before using repaired-protocol masks;
retain previous success masks separately and record which successes change.
Token/text/stop parity and repeated requests remain required. This source audit
alone does not establish a corrected serving measurement or training gain.

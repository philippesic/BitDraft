# Native replay source and serving-protocol audit

Read-only CPU/source audit, November 3, 2026. No GPU operation, model opening,
model hash sweep, helper mutation or canonical-source edit occurred. Sources:

- `runs/w1a8-modern-replay-build-20261103-01/source/modern_replay.cpp`
- `runs/w1a8-modern-packed-runtime-20261103-01/source/common/speculative.cpp`
- Candidate `src/models/eagle3.cpp`, `src/llama-context.cpp`,
  `common/sampling.cpp` and `ggml/src/ggml-cuda/w1a1.cu`
- Existing `cases.json`, `prepared.json`, `freeze01.json`, `result01.json`,
  `native01/replay.json`, `comparison.py` and `comparison.json`

The helper uses the correct EAGLE feature/token/cache shift and actual native
FC, decoder pre-norm, vocabulary expansion and borrowed embedding paths. Its
successful continuations are conditional native replay results on reconstructed
gold histories. It does **not** execute the serving verifier or prove that those
same continuations were accepted during an actual serving round/rollback cycle.

## Encoder and shifted decoder alignment

Serving `common_speculative_impl_draft_eagle3::process()` concatenates the three
configured target input-tap rows in order, runs the EAGLE ENCODE graph, then
pairs `(token[j+1], encoded_features[j])` at draft memory/RoPE position j. Its
last target-feature row is deferred. `draft()` seeds that row with target's
freshest sampled token and samples the following token.

The helper concatenates the supplied feature files in packet order, uses the
same `LLAMA_PROCESS_TYPE_ENCODE`, and copies the returned per-token `t_h_nextn`
FC output. The packet/comparison bind that file order to native taps2/16/29.
In the candidate model's encoder graph, `t_h_nextn` is the raw FC result, not
an extra normalized target feature. The nextn getter synchronizes before
returning, so the helper's copy does not read unfinished asynchronous output.

For an anchor P, helper `prefix()` clears draft memory and decodes exactly
`j=0..P-1` as `(tokens[j+1], encoded[j])` at position j. It checks maximum cache
position P-1. The first sampled pair is then `(tokens[P+1], encoded[P])` at P.
The label index `P-prompt_tokens+2+depth` selects token[P+depth+2], matching
the previously verified two-position offset from source feature to actor label.

This excludes the deferred seed from prefix initialization. Prompt02 starts
at P54 and prompt04 at P558; their reconstructed context sizes are54 and558.
The helper does not accidentally pair a target feature with the unshifted
token or use input position P+1 as its decoder/RoPE position.

Encoder batching is a replay convenience: it pre-encodes every supplied row in
chunks up to512, including future gold continuation rows. Those future rows
are not fed into an earlier decoder/cache state. Each packed FC row is a
per-token integer-dot projection; the CUDA quantizer uses one token per block
and the dot kernel one token dimension, so the source has no cross-token FC
reduction. Actual serving schedules ENCODE per target ubatch instead. This audit
does not independently measure all possible batch-shape effects.

## Teacher and recurrent modes have different meanings

Teacher mode initializes prefix once, requires contiguous anchors, then advances
one decoder step per anchor with the actual gold token and that anchor's
target-derived FC state. A wrong proposed token is logged but never fed as
the next teacher input. The accumulated K/V therefore continue along the gold
shifted context. A teacher success at a later anchor is not a successful
free-running continuation from the first anchor.

Recurrent mode rebuilds the gold prefix separately for every anchor. After the
seed step, it copies `llama_get_embeddings_nextn_ith(ctx,0)` into the next state
and feeds the native proposed token, advancing position P+depth. Candidate
`eagle3.cpp` sets `t_h_nextn` at the residual pre-norm decoder state **before**
the output RMSNorm/head. Thus recurrence does not accidentally use a normalized
head state or refresh from the next target feature.

The helper stops after the first proposal differs from the saved gold label.
No later gold label is used on an alternative proposed-token prefix. It also
stops at the configured depth ceiling or missing next teacher label. Current
packet depth is2, not a proof for longer continuations. The bound of8 anchors
per case also does not establish a native serving occupancy distribution.

## Embedding and logits evidence

The helper creates a CPU-mapped target model/context only to supply native
embedding weights; it never runs target forward. The draft context uses
`ctx_other=target_ctx`. Candidate EAGLE decoder takes the target's `tok_embd`
when no private draft embedding exists, then executes `ggml_get_rows`.

The callback is active only for the single-row decoder step, requests exactly
the F32 `inp_embd` tensor with shape2560x1, and requires one hit. It therefore
observes the actual decoder lookup, rather than the encoder's differently sized
`inp_embd` placeholder. Existing saved-output comparison reports every observed
step embedding byte-identical to the admitted native layer0 row. That qualifies
the observed token rows, not the entire target embedding vocabulary or an
arbitrary future proposal token.

The candidate decoder expands its50000 head rows through **absolute** I64 d2t
IDs into248320 full-vocabulary logits, with unsupported IDs at negative infinity.
The helper chooses the maximum full-vocabulary logit, saves the50000 mapped
entries, and the independent comparison verifies that the saved subset yields
the same proposed target ID. The prepared map's raw I64 hash is checked against
the frozen export. This avoids confusing source-checkpoint offset-form d2t with
the modern GGUF absolute map.

The helper itself only validates packet geometry and reads the map; it does
not perform cryptographic ancestry checks. Those are supplied by the pinned
launcher/prepared packet and saved-output comparison. `prepared.json` retains
its original preparation status, while `result01.json` and native replay output
supply the actual execution evidence. Runtime library maps are captured after
execution and checked against the candidate build. No claim here replaces
those receipts with source inspection alone.

## Exactly what can be retained from native01

The saved native/CPU comparison joins by actual anchor, depth and decoder
position and reports no proposal or correctness disagreement on these cases.
For retention, numerical eligibility must additionally hold through the
reachable prefix for **both** teacher rows used by each actor transition.

Two conditional native-replay slots satisfy the conservative intersection of
native-correct, CPU-actor-correct and numerically eligible reachable prefix:

- Prompt02, anchor57, depth0: label/proposal2702. Native label margin2.27661;
  CPU label margin2.30119. The following depth1 diverges, so only depth0 qualifies.
- Prompt02, anchor60, depth0: label/proposal93008. Native margin0.11919;
  CPU margin0.18358. Its depth1 proposal25 is also native/CPU correct, but it
  uses excluded teacher row8 and is **not** numerically eligible for protection.

Prompt04 has no native-correct prefix in this bounded panel. The native raw
recurrent prefix sum is3, but only2 positions remain eligible after the recorded
teacher-row exclusion. Do not turn the raw sum3 into a protected continuation
count, and do not protect the failing step following a correct one.

These two slots can support an explicitly named *native-replay gold-prefix
retention reference* for the unchanged zero-update parent on the same captured
histories. Recompute the reference on exact pinned parent state, retain the
native/CPU intersection and keep the existing numerical exclusions. They do
not by themselves prove `parent_success_protection_proven` in the stronger
sense of preserving observed serving-verifier accepted continuations.

## What actual serving still adds

The helper bypasses `common_speculative_impl_draft_eagle3::{begin,process,draft,
accept,get_state,set_state}`. It reproduces their inner pair/recurrence algebra,
but it never exercises their persistent protocol state:

- Serving `process()` performs the cross-ubatch deferred-row bridge, stashes
  `verify_g`, and updates `pending_g_last/pending_pos_last` after target batches.
  The helper starts from a newly cleared, reconstructed gold cache at each
  recurrent anchor. It does not audit an actual retained cache after prior
  speculative rejection, removal or catch-up.
- Serving `draft()` removes the deferred boundary from draft memory before
  reseeding it. Serving `accept()` rebases that boundary using the number of
  accepted verifier rows. For hybrid Qwen3.5 targets, `get_state/set_state` also
  preserve and restore that boundary around recurrent target checkpointing.
  None of those accept/reject/restore transitions occurs in native01.
- Native01 consumes offline target features and previously recorded serving
  labels. The target is never evaluated at the helper's proposed batch, so
  the actual serving verifier, target recurrent-state history, rejected rows,
  bonus/corrected sample and acceptance accounting are absent. Label agreement
  under the retained target-only history is a conditional comparison, not an
  observed serving acceptance receipt.
- Helper sampling is raw full-logit argmax. Serving builds a top-k10 sampler,
  may offload sampling, takes its first candidate and applies `p_min`/depth
  controls. The two qualifying slots have positive margins and no observed
  tie, so they are consistent with the intended greedy/p_min0 regime. Native01
  does not directly test tie handling, backend-sampler routing, EOS/stop gates,
  or a nonzero confidence threshold.
- Replay has output reservations512, two CPU threads, a trace callback and
  startup/context reconstruction. Actual server scheduling/reservations and
  instrumentation differ. Its4.1-second execution wall is no full-request speed
  measurement, acceptance rate, quality check or untouched generalization result.

The next serving-level proof should join actual process/draft/verifier/accept
events on the same gold anchors, including at least a genuine rejection and
the subsequent corrected boundary, with source/checkpoint/target/runtime pins
held fixed. Record actual token prefixes, selected target features, deferred
position/state and accepted count. Until that evidence exists, use the narrower
native-replay reference name and leave serving acceptance, full request timing,
independent answer quality and fresh-example improvement unproven.

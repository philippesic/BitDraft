# Modern 4B actor loader and shifted training alignment

Source-only investigation, November 3, 2026. No model tensors were loaded,
GPU/network calls made, or model hashes computed. The safetensors header alone
was read (1,464 bytes) to verify names, dtypes and dimensions. This report is not
an executed modern actor or native drafter qualification.

## Shortest CPU construction route

There is no existing turnkey loader for this modern native-feature-only actor.
Reuse the small `nn.Module` construction pattern in
`scripts/check_recurrent_real_step.py::_build_drafter`, the existing
`RowBinaryLinear` constructor, and `NativeStepAdapter`; do not call that historical
function unchanged, because it rejects anything except Q32/KV8/head128/FF9728.

Construct the shell directly from authenticated `config.source.json` using
`SimpleNamespace` (preserving the nested `rope_parameters`), `nn.Module` children
at the nine canonical paths, and four norm modules. No full target, tokenizer,
`nn.Embedding(248320,2560)` or HF target forward is necessary. With explicit
`ExternalNativeF32Embedding`, the adapter does not require `embed_tokens` at all.
This avoids allocating a random 635-million-element private embedding table.

The existing official path `official_loader.py::load_official_eagle3` is unsuitable
as the modern constructor: it loads a target and tokenizer, its AngelSlim
`ModelLoader.SUPPORTED_ARCHITECTURES` covers old Qwen3 rather than Qwen3.5, and
`BaseEagle3Drafter.__init__` always allocates the entire private embedding even
with `load_emb=False`. Directly using `ModelLoader.load_eagle_state_dict` may
also attempt a network download for its absent `.bin` before trying the local
`.safetensors`; use explicit local `safetensors.safe_open(..., framework="pt",
device="cpu")` instead.

The installed AngelSlim `Llama3Eagle3Drafter` has compatible parameter paths, but
its ordinary forward is not a modern-native proof: its RoPE constructor uses the
whole head, and its forward accesses `self.fc.weight.dtype`, which the binary
wrapper does not expose. `NativeStepAdapter` deliberately supplies the relevant
graph and now handles the partial rotary slice. A meta-device AngelSlim shell
could avoid initial allocations, but was not tested and adds unused rotary-cache
state; the direct shell is clearer and needs no official generation machinery.

## Exact parameters and reusable modules

Use `src/w1a1_eagle/recurrent_binary.py::CANDIDATE_D_BASE_TO_PATH` for the nine
names and `recurrent_qat.py::{W1AxContract,RowBinaryLinear}` with one common
`W1AxContract(8, scale_layout="row")`. All nine are required by the adapter.

| Source/path | Shape |
|---|---|
| `fc.weight` | 2560 x 7680 |
| `midlayer.self_attn.q_proj.weight` | 2560 x 5120 |
| `midlayer.self_attn.k_proj.weight`, `v_proj.weight` | 640 x 5120 each |
| `midlayer.self_attn.o_proj.weight` | 2560 x 2560 |
| `midlayer.mlp.gate_proj.weight`, `up_proj.weight` | 7680 x 2560 each |
| `midlayer.mlp.down_proj.weight` | 2560 x 7680 |
| `lm_head.weight` | 50000 x 2560 |

All nine source weights are BF16, in original checkpoint Q/K row order. For each
local tensor, read on CPU, widen exactly to F32, set initial row scale to the F32
mean absolute weight, and latent signs to -0.5/+0.5 using the existing
`install_joint_linears` initialization convention (zero is positive). Construct
`RowBinaryLinear(latent, scales, W1AxContract(8))` and install it at its path.
The generic installer can be reused only with a legitimate already-constructed
drafter/target ownership boundary: it explicitly requires a target for alias
protection. Do not pass a dummy target and imply that this authenticates ancestry.
Direct construction from source tensors with no target module ownership is the
smaller explicit native-feature-only path.

The four source norms are BF16 vectors of length2560 at
`midlayer.input_layernorm.weight`, `midlayer.hidden_norm.weight`,
`midlayer.post_attention_layernorm.weight` and `norm.weight`. Build modules with
F32 `nn.Parameter` weights and `variance_epsilon=1e-6`; freeze them through the
adapter. `bind_frozen_norms` can bind the corresponding four F32 GGUF norm arrays
when matching the actual exported native model. The source widening and exported
norm arrays must agree; do not replace norms from an unrelated historical model.

Required config is H2560/FF7680/Q16/KV4/head160/one layer/SiLU/pretraining_tp1,
max positions262144, theta1e7 and rotary40. Preserve the nested theta and partial
factor; `NativeStepAdapter::_rope_settings` already parses those. Set
`early_stop_method=None` and `tree_mask=None`; no tree or early-stop graph is
admitted. Biases are absent in the source. Keep every norm F32 when moving the
actor to an accelerator; do not call `.half()` on the whole actor.

Borrow token embeddings from the admitted capture's native layer0 rows. Build a
token-ID lookup over those exact F32 rows, check repeated occurrences agree, and
declare target/capture pins with `ExternalNativeF32Embedding`; choose
`embedding_precision="external_native_f32"`. On CUDA, return detached F32 copies
on the actor device. The small captured row vocabulary is incomplete: fail closed
on an uncaptured proposal token or capture additional native rows. Never silently
substitute the dense HF target embedding or round through F16.

## Vocabulary mapping must remain explicit

The safetensors header confirms `d2t`: I64[50000] and `t2d`: BOOL[248320]. The
checkpoint `d2t` values are offsets: AngelSlim `BaseEagle3Drafter` lines227-229
maps `target_id = draft_id + d2t[draft_id]`. Modern
`runtime/conversion/llama.py` lines339-350 adds `arange` and writes **absolute**
target IDs to GGUF. Modern native `eagle3.cpp` lines307-320 uses those absolute
IDs directly in `ggml_set_rows`, filling unsupported target-vocab logits with
negative infinity. These formats are different.

Validate uniqueness/range and equality of the converted map to the exported
GGUF. Build the inverse using `validate_offset_d2t(..., draft_vocab_size=50000)`
or the absolute-map validator in `prepare_frozen_d_head_data.py`; never feed an
absolute GGUF map to the offset validator. Derive membership from the inverse
and check the source BOOL t2d mask agrees. Unsupported labels stay in coverage
denominators and are not assigned to an arbitrary draft class. The adapter
returns 50000 draft logits; mapping belongs at the loss/verifier boundary.

## Verified offset: actor predicts two positions beyond the feature

Let `x[t]` be the target input token at zero-based position t and let
`F[t] = concat(native_layer2[t], native_layer16[t], native_layer29[t])`.
The native feature taps are input boundaries corresponding to HF layer outputs
1/15/28; this is already the repaired model's tap convention.

Modern `common/speculative.cpp` lines731-776 explicitly pairs
`(token[k+1], g_embd[k])` at memory position k, with `g_embd = fc(F)`. Lines808-826
complete the deferred row with target's newest sampled token. Lines857-898
sample the next draft, feed the decoder pre-norm hidden state to the next depth,
and advance decoder position. Independently,
`recurrent_rollout.py::rebuild_prefix_cache` documents and implements the same
shift; `recurrent_trace.py` requires `input_position=P+d+1` and
`label_position=P+d+2`. AngelSlim `BaseEagle3Drafter` lines198-210 also drops
the first input token before pairing tokens with target features.

Therefore a supervised actor step at decoder position t is:

```text
feature: F[t]
input token / embedding: x[t+1]
decoder / cache / RoPE position: t
target label: x[t+2]
```

The current admission receipt instead records a **target teacher-logit row** at
input position t, whose serving label is x[t+1]. That receipt is correct, but
same-index use as an actor label would be wrong. For actor step i, use feature
at `receipt.rows[i].input_position`, input token `rows[i].serving_label`, and
**target label `rows[i+1].serving_label`**. Require contiguous row positions and
drop the final teacher row, which has no next teacher row for actor supervision.
Conservatively admit an actor transition only when both teacher rows pass the
numeric agreement gate. An excluded middle teacher row then excludes both
adjacent actor transitions. Native-success protection also needs the separate
parent evidence described below.

For a prompt length L, the capture's teacher row0 has input position L-1 and
label generated token0. The first actor step uses `F[L-1]` and generated token0
to predict generated token1 at decoder position L-1. A capture with9 teacher
rows supports at most8 such adjacent teacher-forced actor steps, before masks.

## Context and rollout

For an anchor P, rebuild current-student K/V for positions j=0..P-1 from
`(x[j+1], F[j])`; exclude the deferred seed `(x[P+1], F[P])`. Existing
`rebuild_prefix_cache` accepts token prefix length P+2, features length P+1,
positions0..P, and can use `adapter.build_context_cache` for detached,
K/V-only context construction. That optimized path is valid for this single
layer because its context K/V depend on shifted embeddings plus fused target
features, not on a preceding draft decoder layer.

Then encode `F[P]`, decode seed x[P+1] at P, and retain the attached pre-norm
state and updated draft cache. The next depth consumes the previous **drafter**
pre-norm state and previous proposed token at position P+1; do not substitute
the target feature F[P+1] for this recurrent state. Existing
`rollout_captured_prefix` implements this recurrence and optional batched head.

The current target-only captures supply the gold serving trajectory, not native
verifier labels for arbitrary drafter proposals. Their next labels remain
usable in free-running rollout only while the proposed prefix matches that
trajectory and the relevant rows pass admission. Stop at the first divergent
proposal, or capture the target at that exact alternative proposal prefix.
Teacher-forced per-position restarts and a recurrent rollout are different
objectives; record which was trained.

## Preserving successful continuations and remaining native evidence

Snapshot the unchanged initial binary actor/export before updates and compute
its CPU actor continuation reference with the correct shift. Existing captures
can supply gold target labels/features and the numeric masks, but they cannot
prove a drafter continuation was natively accepted: no modern packed parent has
yet been executed. Do not infer native parent successes from target argmax
agreement or an ordinary drafter's aggregate acceptance rate.

The actual protection set must intersect an independently replayed native parent
accepted prefix with the actor-correct reachable prefix under the same history,
excluding all teacher/actor admission failures. Historical generalization code
already follows that conservative intersection. Rebuild caches from the current
student, not saved parent K/V, and keep the unchanged parent as the control.

Minimal next evidence is an actual all-nine packed modern native model under
the candidate runtime, with the corrected Q16/KV4/rotary40 export, exact borrowed
target rows, norms, d2t, positions, and retained/rejected cache semantics. Compare
bounded native and actor trajectories on those same admitted histories, including
intermediate FC/pre-norm/cache values where needed to localize decision-changing
errors. Then run real native full requests against target-only and frozen DFlash
controls with token/text/stop audits, quality and repeated timings. Small CPU
actor checks or successful serialization do not qualify a serving candidate.

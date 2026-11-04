# Current fixed-candidate QAT readiness audit

The successful modern lineage trains **76,880 per-output-row scale values in nine tensors**, with every binary weight sign fixed. It is real scale adaptation with native serving confirmation; it does not establish all-sign QAT readiness. The selected attention-output sign branch has separately exercised meaningful sign changes, checkpoint/resume/export and native admission. Repeating its two-step plumbing test would add little.

This is a CPU/source and retained-evidence audit, with no GPU work, model selection or source changes. [Machine-readable receipts](qat-source-audit.json) include current endpoint/source hashes, exact shapes, tests and cost projections.

## What the present implementation actually does

[modern_actor.py](../../scripts/w1a8_generalization/modern_actor.py) constructs nine `RowBinaryLinear` modules with `W1AxContract(8)`. Metadata requires `scale_group_size=0`; each native F32 scale array has shape `[output_rows]`. `enable_scale_only` freezes all parameters and enables exactly the nine `scale_offset` vectors. Collect/load/export verify frozen sign hashes. The 50,000-row output head accounts for 50,000 of 76,880 scale parameters.

Direct CPU inspection of the selected 5760 checkpoint verified all nine vector shapes, finite offsets, nine Adam entries at step5760, CPU/CUDA/Python RNG and all stage/replay/exposure fields. The checkpoint is 1,509,425 bytes; its native model is 40,802,176 bytes. The periodic checkpoint status `RECOVERY` is expected at a terminal multiple of480; the terminal trainer/export result is separate. Current actor and added-replay trainer files exactly match the saved source contracts.

`export_scale_only_gguf` copies the pinned initializer, patches exactly nine effective-scale payloads and proves every other byte unchanged. It applies the existing partial-Q/K row permutation. It rejects changed signs, overwriting existing files and altered initializer ancestry. An exported file still requires native admission.

## Teacher, cache and loss semantics that must survive new features

The actual successful training view comes from [modern_actual_data.py](../../scripts/w1a8_generalization/modern_actual_data.py), reconstructed from **serving traces**. This differs from the older conditional target-capture admission path. Raw target-feature positions may differ from decoder cache positions; `prefix_raw_rows` and `seed_raw_row` preserve the observed native lineage. No implicit position repair is valid.

The current actor rebuilds detached prefix K/V in chunks of64 from current weights and exact captured raw rows, then encodes the seed and carries its own pre-norm state through a bounded two-step gold-token unroll. K/V pass through F16 before attention and append. Within-window recurrence remains attached to autograd. See [native_step.py](../../src/w1a1_eagle/native_step.py), `build_context_cache` and `decode_step`.

Ordinary CE uses two FIT windows, labels supported by the fixed 50k vocabulary, and the **static original-parent** `native_supervision_reachable_mask`. It sums admitted token losses with denominator4. It does not dynamically drop labels when the current actor first diverges. The current candidate also replays original356 native FIT successes and an independent added407-success newFIT bank, each at lambda10. Replay minimizes mean squared frozen-parent margin deficit, with targets `max(.1,min(reference_margin,2))`. DEV is readout/selection only.

These traces establish hard-label CE plus native margin replay. They do not supply a qualified full probability-distillation teacher. Generic `compact_probability_loss` approximates the mapped-vocabulary tail uniformly and is conditional CE, explicitly not full KL. A current-runtime soft-target objective needs authenticated probability capture and a declared approximation.

Torch A8 dequantizes before the dot, whereas native uses an integer dot then scales; accelerator RoPE arithmetic also differs. Bounded native trajectory/decision checks are the practical gate. Neither source inspection nor current confirmation proves global bit-exact arithmetic parity.

## Supported branches and remaining design work

| Branch | Actual evidence | Remaining scope |
| --- | --- | --- |
| Nine row-scale vectors, fixed signs | Meaningful training, exact restoration, native export/admission, fresh24 five-repetition confirmation | New feature configuration needs explicit migration/noop parity and split-resume from the current added-replay checkpoint |
| Selected attention-output signs plus nine scales | 570 actual packed-bit flips after1920 new updates; exact false-control3840 parity; strict resume to4320 gives727 flips at1e-3 and4981 at3e-3; both mature arms evaluated on all549 native references and28 HTTP responses | No five-repetition speed or fresh generalization claim for this sign branch; only6,553,600 O latents trainable |
| Selected512 output-head rows | Real1,310,720-element finite nonzero gradient; exact split-resume and zero-export qualification | Early zero hard flips do not establish capacity benefit. Global clipping changes the treatment; initial head/total gradient norm ratio0.002166 |
| All nine sign matrices plus scales | Generic row-linear STE and joint optimizer code exist | Modern all-sign contract, latent+scale masters/Adam/RNG/cursor checkpoint, changed-sign exporter, actual full-shape GPU resource/resume/native gates remain unqualified |
| Within-row grouped scales | Historical group128/A16 CPU reference and export path exist | User design decision; current A8 metadata/loader/state/forward do not admit groups. New grouped A8 kernels and partial-sum scaling, initialization, optimizer and export/admission are required |
| Learned activations, affine midpoints, FC corrections | Generic/historical implementations exist | Current exact modern actor/state metadata reject them. Integrate only after a relevant design decision, then measure cost and native behavior |

The initial selected-sign admission report had zero learned flips after two unique updates. Later paid training supplies the missing meaningful-flip evidence; use both records with their dates rather than quoting the old limitation as current fact. Native mature-sign preservation loses46/549 positions in both arms; HTTP parity is exact, but single-pass qualification is not a throughput result.

Generic `save_joint_checkpoint` writes export NPZ latent/effective-scale arrays; it omits optimizer/RNG/cursor. It cannot replace the modern resumable training checkpoint.

## Resource and cost bounds

The selected replay3840→5760 stage measured210.839 incremental trainer seconds for1920 updates. Its whole child process took495.083s including setup/readouts. The `trainer_seconds=545.226` field is cumulative paid trajectory time, so it must not be compared directly with current-process wall time. Extra replay forward alone cost45.165s; changing objective packages is not automatically equal compute.

The resident cache held594,790,400 bytes under a1GiB limit with a2GiB host-availability reserve. The real all-matrix resident proof compared3807 matrices /600,442,880 bytes,60 max-anchor windows and2135 embedding rows exactly, taking56.693s. These are actual current data checks, not regenerated historical captures.

Selected-O qualification measured3,149,585,920B peak CUDA allocation,3,875,536,896B reserved and3,789,348,864B process RSS; checkpoint79,603,493B. These measurements do not describe all-sign training.

At the same current geometry there are232,857,600 sign elements. Algebraically, F32 latents+gradients+two Adam moments total3,725,721,600B; materialized hard signs may add931,430,400B. A latent-plus-moments checkpoint alone is2,794,291,200B, before scales, metadata or staging. Activations/KV/allocator/scratch and serializer duplication require actual measurement. No all-sign fit or long-QAT speed guarantee follows from scale-only timings.

For cost mapping only, hypothetical group128 scales on the same shapes would number1,819,200, versus76,880 row scales:23.663× as many. F32 scale payload rises307,520→7,276,800B (+6,969,280B). This is dimensional arithmetic, not a recommendation or representation experiment. Native partial sums and launch/bandwidth overhead remain to be measured after a user decision.

## Cheapest material checks

1. Qualify any new objective wrapper with strict disabled/noop numerical parity and split-resume from selected5760. Preserve Adam by named path, all RNG, static CE masks/fixed4 denominator, both replay streams and exposure ancestry. Old resident-only resume evidence does not cover a new added-replay wrapper.
2. Test a small compatible loss/objective package, such as depth weighting or label smoothing with unchanged replay. Budget480-update pilots, then a promising combination and an informative apparent loser to1920 when warranted. Existing stage costs imply roughly211s incremental training per1920, with substantial separate setup/readout cost. Use consumed DEV, frozen checkpoints and five timing repetitions; no universal early-pruning inference.
3. Rebuild/materialize/hash an existing FIT capture view and native-admit a newly trained endpoint if current prep needs fresh exercise. Do not mix bridgefix-overlay captures into the immutable original-runtime lineage. Soft probability objectives need new teacher qualification; hard-label experiments do not.

No additional selected-O plumbing or architecture/precision sweep is needed to answer these gaps. All-sign/group-scale/correction work remains explicitly dependent on the later design discussion.

## Verification and retention

46 existing synthetic CPU contract tests passed in2.056s: actor compact state, actual-data lineage, added-replay contracts, selected-sign migration/resume and resident trainer. Exact command is in the JSON receipt. The first pytest invocation found pytest absent; unittest succeeded without environment changes. No GPU process was launched.

Relevant proofs include current [fresh confirmation](../w1a8-modern-native-20261103/natural-confirmation.json), [selected-sign admission](../w1a8-modern-native-20261103/selected-sign-admission.json), and ignored source/run receipts pinned in the audit JSON. Authorized historical cleanup removed raw captures/intermediates while retaining endpoints and compact evidence. Exact historical raw replay needs regeneration; retained reports do not imply those bytes are still present.

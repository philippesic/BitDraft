# Modern 4B generalization data plan

Prepared 2026-11-03, CPU/source inspection only. **Recommendation:** recapture a balanced, whole-group subset of authenticated existing TRAIN messages on the modern native target. The attached inventory concretely selects 24 fit groups and 12 development groups. This is a proposed phase-one generalization pilot, not a validated short-run screening workflow, a historical training replacement, or evidence of a speed improvement. No GPU/model work or deletion was performed. No final/sealed evaluation payload was opened.

## Usable source inventory

| Existing file | Raw groups | Domains | Use |
|---|---:|---|---|
| `runs/w1a8-algorithms-20261020/loop2-answer-expand01/bindings.json` | 404 | Included in union below | Raw `messages`, original receipt, prompt ID, prompt SHA, TRAIN ancestry and original group ID |
| `runs/w1a8-algorithms-20261020/loop2-mode-capture05/bindings.json` | 108 | Included in union below | Same authenticated raw-message fields; includes inherited probes |
| Both bindings together | 512 unique groups | code171, prose171, reasoning170 | All raw text is available locally; no original remote corpus recovery is required for this pilot |
| `/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031/split.json` | 288 unique groups | fit80/domain; development16/domain | All288 map to the raw bindings above; retains the previous role split |
| `runs/w1a8-generalization-20261102/data-policies03/occupancy/manifest.json` | 1152 windows from those288 groups | 4windows/group | Historical serving-occupancy evidence and sampling ancestry only |

`generalization-data-inventory.json` records exact absolute source paths, small-file hashes, source-row references, all36 selected group IDs and original receipt references. The raw bindings total741,521bytes. Each candidate has a source prompt ID/SHA and `source_split: TRAIN`; the text is not copied into this report.

The frozen original full-pool plan also names10 original TRAIN shards/indices, 10,000 prompts, with3286/3285/3285 optimizer TRAIN prose/code/reasoning assignments plus96fit/48calibration-validation. Those old local-corpus paths point to another machine and are not proof that the original corpus is currently accessible here. The native bindings are the concrete recoverable source, so do not block this pilot on transport of the full corpus.

## Proposed groups and split limitations

Selection is content-independent: seed `w1a8-modern4b-generalization-data-v1`, SHA256 of seed + NUL + group ID, sorted independently inside each inherited role/domain. Select8fit and4development groups per domain. This preserves the existing role assignment and never splits windows/continuations of a group across roles. Existing historical prompt lengths across288 groups are20–271; selected36 are20–200. These counts use the **old Qwen3 tokenizer**, not Qwen3.5; no modern lengths have been measured in this assignment.

This selection is deliberately **not frozen yet**. Before capture the operator must compare prompt hashes/text against the three already inspected modern initialization requests and all modern training/probe inputs, and exclude exact or group/topic-related overlap from the untouched development arm. Record exclusions and deterministic next-ranked replacements. Retain the original grouping metadata and verify its topic-component construction; the group ID alone is not an independent proof of complete semantic deduplication. Freeze the final roster and source pins before observing modern labels or outcomes.

These are authenticated historical TRAIN prompts, and the full512 pool has historical experimentation exposure. Development is untouched by the proposed modern fit updates only after the overlap check. It is not a pristine original-corpus holdout or a final performance/quality panel. Results must say so. A24/12 pilot can detect severe transfer failures but gives weak domain-level estimates; do not reject slow-learning candidates or calibrate the backlog workflow from this alone.

## Capture and training semantics

Use Qwen3.5-4B Q4_K_M with the frozen correctness runtime from `experiments/w1a8-standing-20261102/baseline.json`, fresh native target features/logits and the corrected shifted actor mapping in `experiments/w1a8-modern-native-20261103/report.md`. Historical Qwen3 F16 tokens, hidden vectors, masks and teacher logits are incompatible and must not be relabeled modern.

Capture each original user message using the actual serving rendering and bounded target continuation; retain source message, rendered input, tokenizer identity, native target labels, generation stop reason, eligible starts and per-round native acceptance/committed-token history. Proposed ceiling is384total target rows/group and80generated tokens; measure native prompt lengths before freezing. Never silently truncate an original task to fit this budget: if a group exceeds the cap, report it and use a predeclared length stratum/replacement rule or request the required storage. The largest selected old prompt is200tokens, but modern rendering/think tokens still need measurement.

Preserve full prefix features for recurrence. Within each fit group sample uniformly among complete eligible serving starts, with a frozen seed and recorded inclusion probability; weight by the number of eligible starts so short prompts and early difficult positions do not dominate occupancy. Keep short/EOS and unsupported-label cases in the accounting. For a compact first packet, retain17consecutive full-vocabulary teacher-logit rows around a preselected start, yielding at most16shifted actor transitions before the adjacent-row numeric gate. More starts can be captured later under the same group roles, without outcome-based selection.

The existing numeric gate rejects2/18teacher rows and leaves13eligible shifted actor transitions on the original initialization example. Apply the same recorded adjacent-row gate; do not infer its incidence on new prompts. Resolve the50,000draft versus248,320target vocabulary using actual mapped absolute IDs. Report supported-label fraction alongside correctness. Unsupported targets remain in total serving coverage; do not silently drop them and report only a favorable supported subset.

**Native success preservation is still missing.** Aggregate67/408acceptance on three initialization requests is not a per-parent protection mask. First capture unchanged-parent proposals and verified prefix matches on the exact new fit continuations. Preserve every verified previously correct continuation in the loss/constraint and test it after training; do not infer successful spans from teacher-forced predictions or from historical F16 labels. Development trajectories stay evaluation-only. Compare initialization and trained exports on identical untouched groups, token correctness, answer quality and repeated complete-request wall time, with unchanged DFlash Q4_K_M controls. Historical F16 measurements remain separate.

## Storage and compact pilot feasibility

With four F32 taps (layers0/2/16/29), width2560, features cost40,960bytes/target row. A full-vocabulary F32 logit row costs993,280bytes. For36groups ×384feature rows plus17logit rows/group:

- Features:566,231,040bytes (0.527GiB).
- Logits:607,887,360bytes (0.566GiB).
- Total:1,174,118,400bytes (1.093GiB), before small metadata and model exports.

Against the reported3.6GiB WSL free space, this leaves roughly2.5GiB, so the packet plus a40.8MBpacked export is plausible while preserving a2GiB floor. This is an upper-bound estimate, not an allocation or fresh disk-space audit; the operator must recheck competing jobs/checkpoint growth before capture. Avoid duplicate staging and retain all useful evidence. No C: space is assumed available.

A **separate modern scale-only arm** is storage-feasible in principle:76,880row scales consume307,520F32bytes; parameters plus two Adam moments are922,560bytes before minor metadata, rather than billions of full-sign optimizer bytes. Frozen signs can be referenced by immutable native initialization/source/mapping pins. A real compact checkpoint must round-trip/resume the identical effective actor and produce a numerically checked native export; omitting arbitrary state without this proof is not preservation. Scale-only is a research choice with limited ability to change sign representation, and is not assured to generalize.

This independent modern pilot must not shorten, substitute for, or declare complete any authorized historical full-sign/full-state allocation. Storage for those full checkpoints remains the existing consequential unresolved question. Do not delete artifacts to make room without parent/user resolution. First obtain modern zero-update capture, valid protection masks, and token/quality/timing evidence; only then consider whether the pilot is informative enough to expand. No quick-feature predictor or optimization-backlog screening is authorized by this report's results.

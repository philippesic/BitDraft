# Data admission

For the scoped Qwen3.5-4B Q4KM/HE4 recipe, FP16 feature storage passed the
[production qualification](../results/fp16-storage-adoption/result.md). It changes file
storage only: the loader keeps features in FP16 and the model converts them to FP32 on the
training device, an exact conversion. Other targets need separate checks.

An intent with `corpus_format: corpus-mixture-v1` binds a compact mixture artifact rather
than copying feature arrays. Its fields are `schema: corpus-mixture-v1`, `target_sha256`
and `components`. Each component has `kind` and authenticated `source`: `index-intent`
binds a retained-index intent; `corpus-v1` binds a direct TRAIN-only corpus. Two to eight
components are supported. All targets must match. Protected aliases and roles remain
enforced, and source/group aliases plus exact tokenized prompt prefixes cannot overlap
across components. Repeated captures already inside a legacy component are preserved.
The mixture identity and sampling artifact bind the new candidate recipe. This supports
explicit same-target mixtures, not automatic transfer between models.
To use part of the original index, copy the selected rows unchanged into a new index with
updated `sequences`, `windows` and `unique_labels_in_windows`, and bind it through a
retained-index intent with the same counts; rows are still authenticated against the
original receipts. [The S4 data curve](../results/s4-data-curve/result.md) records one
such selection.

Training data enters through a typed ledger. Admission reads metadata, checks roles
and counts, and refuses before any training starts. Raw tensors are authenticated when
they are read, not at admission. Terms are defined in [domain](domain.md).
Feature arrays are read once into memory and hashed from those exact bytes, with file
identity checked before and after the read; they never reuse a digest. Other files
can reuse a SHA256 within one process for unchanged device/inode, size,
mtime and ctime, under a bounded 1024-entry cache. Reads check identity before/after hashing;
edits and replacements invalidate reuse. New processes hash independently. This assumes
ordinary filesystem metadata integrity, not a hostile kernel or filesystem. No persistent
receipt or caller-declared digest substitutes for authentication.

The maintained [bank definition](../benchmarks/definitions/prompt-bank-v1.json) records
coverage, source revisions/licenses, native template identity and authenticated external
selected-ID artifacts. Prompt bodies and derived identity lists are data artifacts.
[Supplement selection](../benchmarks/definitions/supplement-v1.json) records the focused
2K slice. `benchmarks select-supplement` authenticates the bank and retained index, excludes
legacy identities and exact tokenized prefixes, solves category/length and output-cap
quotas, and forecasts maximum feature bytes. It does not generate captures or establish
semantic difficulty. Inspect source quality before production acquisition.

The frozen 6K bank's [completed FP16 corpus](../results/frozen-train-bank-capture/result.md)
is indexed at `/home/philip/BitDraft/artifacts/datasets/train-bank-v8-fp16-01/corpus.json`.
It references the existing supplement and both capture shards; every supporting root
is now inside `artifacts/datasets/`, including the copied Windows-produced files. This is TRAIN data with recorded prior
exposure, not independent evaluation data or a new training allocation.

The prompt-bank bodies and role/ID selections remain under `bank/v8`. Portable curation
metadata is under `bank/v8-relocated-01`; its relocation receipt changes only the old
exclusion-file locator. Historical admission/specification files remain byte-identical.
The legacy prompt text and opaque role indexes used by curation are preserved under
`artifacts/prompt-sources/legacy-freeze-002`, outside tracked source.

## Prompt identity and roles

A prompt is identified by its source revision and license, canonical messages,
token-path digests, role origin and version, group, topic, aliases and consumption
provenance. Roles and aliases are reconciled before training metadata is opened.
heldout24 is always protected. Separate capture attempts of the same prompt do not
count as more unique data.

Group and topic relations use separate keys: `(source, group, value)` and
`(source, topic, value)`. A null, absent or blank topic creates no relation, so two
prompts with missing topics are never merged. Connected components are stable under
row reordering and under adding an unrelated row with a missing topic.

## Counts

Every input ID gets exactly one terminal disposition, so totals always reconcile.
Unique supervised positions are keyed by canonical trajectory and logit position,
never by file path. Records with zero windows stay in the accounting but never
enter sampling.

## Captures

The supported capture contract has eight taps at layers 2, 6, 10, 14, 18, 22, 26 and 30,
with zero-based pre-block input semantics and an authenticated teacher, tokenizer and
template. Hard labels must match the argmax of finite full logits. Logit `p` predicts
token `p + 1`. The final prompt token is a valid first anchor: logit `prompt_tokens - 1`
predicts the first generated token. End-of-generation tokens inside the prompt do not
stop generation; the first generated one is the last supervised label. The original `full15` policy requires fifteen labels. Explicit `up-to-15` metadata
also admits shorter terminal windows, normalizing the same decay weights over available
labels. It retains short answers and EOS without inventing padding or post-EOS labels.

The reader handles modern eight-tap F32 and explicitly declared FP16 storage and optional pinned int32 NPY copies.
It authenticates files before use, keeps features in their stored precision, bounds how many
features it materializes, and reconstructs the seeded grouping order from the saved cursor,
repeating it each pass ([groups and prefetch](execution.md#sampling-and-clocks)). Historical DSpark captures are not modern HE4 teacher data.

## The modern index

The modern index adapter (`he4fresh-index-v1`) accepts three source-record forms:
records with role and ancestry, retained training-catalog locators, and authored
broader-domain records. Any other form is refused. Prior use is reported per record as
historical training, unknown or unobserved. Agreement with retained receipts shows the
metadata chain is consistent; it does not show the data is unused or adequate.
The reconciled counts are in [findings](../FINDINGS.md#f011-does-the-retained-modern-index-reconcile).

## Exposure

Admission estimates position visits for the current corpus and supplied run budget,
accounting for overlapping windows. These are not measured lifetime per-position counts.
Same-recipe resume restores the sampler cursor and aggregate exposure counters. Branching
keeps aggregate counters and parent lineage but resets the cursor and optimizer; its
recipe, seed or corpus may change. Payloads 4 and 5 preserve exact observed label totals and dataset identity/origin for the
new pipeline. No per-position lifetime exposure ledger is retained
across those changes. Exposure limits and minimum-data policies are caller inputs; test
values are examples, not approved scientific thresholds.

## Moving metadata

`python -m tools.snapshot_metadata` packs or unpacks authenticated JSON metadata
under explicit byte and time limits; `--help` lists its arguments. It never moves raw
features and grants no admission. Unpacking refuses path traversal, duplicate members
and non-regular files, and the index reader must still reconcile the result.

## Capture and format conversion

[Benchmark commands](benchmarks.md) prepare/admit prompt banks and capture pinned native
teacher trajectories. TRAIN and DEV acquisitions keep their roles. A DEV capture corpus
is diagnostic input and cannot pass training admission as an empty TRAIN corpus.
Opaque heldout identity metadata protects heldout24 without reading its prompts/answers.

An acquisition can reuse completed captures of the exact same bank entries. Authenticate
the original corpus and arrays; match messages/tools, token identity, output cap, target,
storage precision, role and supervision policy. Keep one record per requested ID and
retain its original references and validation receipt. Fresh shards use public `capture`
entry points. A combined corpus is complete only when every requested ID is present and
all product validations passed. Storage can span host filesystems; record physical roots
and preserve their references. Reusing captures does not make previously seen data fresh.

Use the configured numerical interpreter reported by the doctor for
`python -m w1a1_eagle.numerical.conversion SPEC OUTPUT`; the check-only environment
does not contain its numerical dependencies. It converts a schema-1 F32 corpus to
F16 in a new `artifacts/datasets/` root. The spec binds `corpus`, `maximum_relative_l2` and
`capture_limit_bytes` under schema 1. It checks all finite values and overflow, streams
bounded chunks through atomic assets, preserves labels/roles and supporting metadata,
and records per-array errors. It never deletes the original. Conversion success is a
numerical/storage result; production suitability requires matched learning/native checks.

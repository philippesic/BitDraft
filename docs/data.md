# Data admission

Training data enters through a typed ledger. Admission reads metadata, checks roles
and counts, and refuses before any training starts. Raw tensors are authenticated when
they are read, not at admission. Terms are defined in [domain](domain.md).

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
stop generation; the first generated one is the last supervised label. Each window
covers fifteen labels.

The reader handles the modern eight-tap F32 format and optional pinned int32 NPY copies.
It authenticates files before use, bounds how many features it materializes, and
reconstructs the original seeded permutation from the saved cursor, repeating it each
pass. Historical DSpark captures are not modern HE4 teacher data.

## The modern index

The modern index adapter (`he4fresh-index-v1`) accepts three source-record forms:
records with role and ancestry, retained training-catalog locators, and authored
broader-domain records. Any other form is refused. Prior use is reported per record as
historical training, unknown or unobserved. Agreement with retained receipts shows the
metadata chain is consistent; it does not show the data is unused or adequate.
The reconciled counts are in [findings](../FINDINGS.md#f011-does-the-retained-modern-index-reconcile).

## Exposure

Exposure counts visits to each unique position across overlapping windows. Parent
exposure, seed, cursor and trajectory identity carry through resume and branch.
Exposure limits and minimum-data policies are inputs to each run; the values in tests
are examples, not approved thresholds.

## Moving metadata

`python -m tools.snapshot_metadata` packs or unpacks authenticated JSON metadata
under explicit byte and time limits; `--help` lists its arguments. It never moves raw
features and grants no admission. Unpacking refuses path traversal, duplicate members
and non-regular files, and the index reader must still reconcile the result.

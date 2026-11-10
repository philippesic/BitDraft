# Data contracts

The source port has a typed ledger, canonical schema-1 capture reader, selected modern
index adapter and deterministic bounded CPU sampler. The selected real index has been
reconciled against retained metadata and source receipts; raw tensor bytes and native
teacher execution remain separate gates. See [execution](execution.md). heldout24 and every alias remain permanently evaluation-only;
exclude them from train, calibration, selection, sampling and conversion. Consumed DEV
is selection data, not fresh confirmation. Unknown prior training ancestry excludes a
prompt from new protected evaluation.

Prompt identity needs source revision/license, canonical messages and token-path digests,
role origin/version, group/topic, aliases and consumption provenance. Separate capture
attempts cannot inflate unique data. Unique supervised positions are keyed by canonical
trajectory and logit position. Optional null/absent/blank topic produces no relation.
Group and topic keys have separate namespaces and source namespaces; connected components
are stable under row permutation and adding an unrelated missing-topic row.

Count conservation requires exactly one terminal disposition for every input ID. The
supplied partition is 10,000 = 96 cal-fit + 48 cal-val + 9,856 candidates; 9,856 = 12 DEV
+ 23 real collisions + 5,505 wrong rejections + 4,316 admitted. Identical genuine exclusions
would admit 9,821. The approximately 5,497 recovery count differs by eight and is open.
The earlier gross 10,150 sequences / 970,121 windows / 1,111,811 labels were provisional,
not an admitted corpus. These are historical investigation counts. CPU reader work does
not authorize transfer, scientific allocation or reinterpretation of exclusions.

Capture schemas require eight taps [2,6,10,14,18,22,26,30], zero-based pre-block input
semantics, authenticated teacher/tokenizer/template, hard labels matching finite full-logit
argmax, logit p predicting token p+1, first-EOG supervision stop and complete 15-label
window coverage. Zero-window outcomes stay zero. Historical DSpark captures cannot be
silently renamed as modern HE4 teacher data. Shape/byte/hash rejection is exercised on synthetic files. Actual teacher parity still
requires retained source receipts and the separate native gate.

Exposure sums visits across overlapping windows into each unique position; window passes
and mean/max position visits are distinct denominators. Parent exposure, seed, cursor
and trajectory identity survive resume and branch. Threshold examples are tests only,
not an approved allocation or optimum. See [decision 003](decisions/003-data-and-exposure.md).

The modern adapter accepts the three observed source-record forms explicitly: rich
role/ancestry records, retained training-catalog locators, and authored broader-domain
records. Unknown forms refuse. Verified metadata ancestry means the retained identity
and receipt chain agrees; it does not establish that data is unused or scientifically
adequate. Prior-use classification remains a separate field. The selected index has
9,813 historical-training, 5,658 unknown-use and 176 unobserved-use records. Earlier
recovery discrepancies remain unresolved and are not reinterpreted by this intake.

The final prompt token is a valid first anchor: logit position prompt_tokens - 1 predicts
the first generated token. Prompt-internal EOG token IDs do not truncate generation.
The first generated EOG remains the terminal supervised label. The sampler preserves
index row order and the original single seeded permutation, repeated across passes.
Compact anchors avoid constructing every fifteen-label Window at admission. All
zero-window records remain accounted for without entering sampling.

For slow metadata mounts, `python -m tools.snapshot_metadata` packs or unpacks only
authenticated JSON metadata under explicit byte/time bounds. Its help lists exact
arguments. It never transports raw features or grants admission. Unpack refuses
traversal, duplicate members and nonregular files; the ordinary index reader must
still reconcile the result. Metadata and raw-data mounts can be relocated separately.
See [complete intake](evidence/modern-index-complete-intake.json) and
[transport receipt](evidence/metadata-snapshot-transport.json).

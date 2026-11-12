# The frozen 6K TRAIN bank is captured and admitted in FP16

All 6,000 frozen TRAIN IDs have complete Qwen3.5-4B Q4KM teacher captures. Every record
passed the capture's product loader, and the actual training Corpus reader admitted
the complete collection on CPU. Feature arrays occupy about 230 GiB. This establishes
data readiness for the scoped recipe, not useful learning or serving performance.

## Outcome

| Component | Validated records |
| --- | ---: |
| Reused exact-match supplement | 2,000 |
| New Linux shard | 1,789 |
| New Windows shard | 2,211 |
| Complete unique TRAIN IDs | 6,000 |

All 6,000 records are eligible, with 2,283,089 supervision windows. Metadata, source
messages/tools, token identities, caps, roles and validation receipts match the frozen
bank. Native input bands and category counts reconcile with the
[bank definition](../../benchmarks/definitions/prompt-bank-v1.json).

FP16 arrays total 246,868,172,800 bytes; all referenced record files total 247,218,907,914
bytes, excluding coordinator/manifests and shared runtime objects. Training materializes
FP32 features. The longest captured sequence is 13,471 tokens. Every record uses the
declared eight taps and up-to-15 supervision policy, including terminal short windows.

## Scope and limitations

Acquisition used the pinned Q4KM teacher/native helpers on the RTX 5080. FP16 suitability
comes from the earlier [matched qualification](../fp16-storage-adoption/result.md);
this acquisition did not rerun training or qualify another target precision.
The actual Corpus metadata/label/role admission used an 8 GiB CPU budget. Arrays passed
product validation during capture and are authenticated again when consumed; final
verification authenticated complete manifests, metadata and validation receipts and
checked array sizes. It did not repeat every numerical operation over all arrays.

There are 2,248 records with known historical training exposure; exposure for the other
3,752 public-source records is unknown. Unique prompt IDs are not independent heldout
examples. Neither 6K size nor category coverage proves sufficient training breadth,
resistance to overfitting or real-application transfer. DEV and heldout data remain
separate. The original 15,647-record corpus and original final220095 remain preserved;
this result does not silently combine them with the 6K collection.

The progress observer failed while reading Windows `progress.json`. The independent
capture process continued and completed every record. Its error/logs are retained;
resuming the recorded coordinator authenticated reuse, skipped completed shards and
published the combined corpus without another teacher launch. The recovered result's
`wall_seconds` covers finalization recovery, not total data-generation time.

## Decision

Retain the complete FP16 corpus for future authorized same-target work. No additional
training ran during closeout. Preserve all three supporting roots: the reused supplement,
the Linux shard and the Windows shard. Their arrays are referenced rather than copied;
the supplement remains a dependency even though hybrid training was cancelled.

## Evidence and reproduction

[Measurements and authenticated references](result.json) identify the complete corpus,
frozen bank, exact capture specifications, source snapshots, native execution outputs,
record validation, CPU admission and failed observer/recovery.
The collection is indexed at
`/home/philip/binary-eagle-artifacts/datasets/train-bank-v8-fp16-01/corpus.json`.
The Windows shard physically resides under
`/mnt/c/binary-eagle-artifacts/train-bank-v8-fp16-01/captures`.

Use the recorded `benchmarks capture` specifications and runtime/source for acquisition
or verified-record resume. Existing completed data need no regeneration for inspection.
Recompute counts and byte totals from the authenticated corpus/record metadata; the
product Corpus reader supplies role, label and eligible-window admission. These checks
do not allocate a model or authorize training.

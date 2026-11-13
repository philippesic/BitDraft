# Reading and recovering evidence

Start with [FINDINGS](../FINDINGS.md), then the result's scope, decision and evidence.
[Source catalog](../results/sources/catalog.json) maps imported files to their original
revision, path, length and SHA-256. `.txt` reports are byte-identical archival documents,
not current instructions: their old job ownership, commands and approvals are historical.
A JSON source reference identifies a larger record or implementation in the ignored
Git archive under `artifacts/archive/`; it is not a substitute for that record when recomputing a claim.

## Independent archive

The historical project and native-source bundles are stored inside the checkout at
`/home/philip/BitDraft/artifacts/archive/legacy-20261110/`. Their identities, recovery checks
and exclusions are recorded in the [archive receipt](../results/archive-preservation/result.json).
Git tracks compact reports and conclusions. The ignored archive retains
full committed history, including old diaries, proposed designs, implementation and
supporting records not copied into the ordinary source tree.

Both bundles were restored into independent bare repositories and checked with
`git fsck --full`. They do not borrow objects from the original checkout. A source record
can be read without that checkout or network access:

```sh
git --git-dir /home/philip/BitDraft/artifacts/archive/legacy-20261110/recovered.git show REVISION:PATH
```

Use the revision and path from the catalog or result's source reference. Relative paths
inside an archived report are relative to its original path at that revision. Retrieve
those paths from the archive, rather than treating them as links in today's source tree.
Native implementation history is in `native-recovered.git` beside it. To rebuild these
readers elsewhere, clone each bundle with `git clone --mirror BUNDLE DESTINATION`.
Verify the bundle's SHA-256 against the receipt before using a relocated copy.

## Limits

This preserves documented knowledge and committed implementation; it does not preserve
all ignored run trees, datasets, weights, caches or frozen runtime objects. Some exact
model states were intentionally retired before adoption. Source-linked raw paths remain
historical locators, not verified current availability. Experimental reruns require their
own input/dependency verification; the archive does not establish GPU or native parity.

The initial November10 archive retains interim records. Final committed history is
independently preserved under `/home/philip/BitDraft/artifacts/archive/legacy-final-20261111/`
with its own `preservation.json` and verified bare reader. Final working reports and
the original run/data now live under
`/home/philip/BitDraft/artifacts/datasets/retained-he4broad-original/`; its
`closeout-reports-preservation.json` verifies the copied reports. This includes the
220095-update,15647-capture run, distinct from the November8 forty-sequence namesake.

Source history is published to [BitDraft](https://github.com/philippesic/BitDraft).
The ignored bundles and large artifacts remain local; a Git push does not back them up. The archive lives under ignored `artifacts/archive`; include it when moving the whole
project or backing up artifacts. Deleting the whole checkout deletes these local bytes. Neither archive adoption nor the document cleanup authorizes deleting live
jobs, original model inputs or ignored execution records.

## Compact legacy artifact evidence

The [legacy closeout](../results/legacy-artifact-closeout/result.md) records the later
explicitly authorized artifact retirement. Its exact receipts supersede older statements
that every historical checkpoint or raw run remains available. Historical measurements
and source identities are unchanged; discarded optimizer states cannot be resumed and
raw tensors/events not retained cannot be recomputed from a hash.

`/home/philip/BitDraft/artifacts/archive/legacy-closeout-20261112/` contains deduplicated compact
support, source snapshots, training-curve aggregates, failure diagnostics and exact
worktree WIP patches. Compressed indexes map the original path to `objects/SHA256` in the
corresponding tar archive. The tracked closeout receipt authenticates these files. Look up
the relevant old path in the index, then read just that object, for example:

```sh
tar -xOzf /home/philip/BitDraft/artifacts/archive/legacy-closeout-20261112/compact-support.tar.gz objects/SHA256
```

`compact-support` preserves reports, per-request measurements and supporting tables;
`native-source-support` preserves source/header/build snapshots. `external-worktree-support`
adds compact records from retired external checkouts. Native uncommitted changes were
restored in independent checkouts to verify the preserved staged/unstaged patches and
untracked files. `training-series.jsonl.gz` stores256-row descriptive aggregates, not the
full optimization stream; it cannot support new per-example or uncertainty claims.
`failure-diagnostics.jsonl.gz` preserves matching failure excerpts with source identities;
text matches are diagnostic evidence, not newly classified scientific outcomes.

Current model files, reusable benchmark controls and their required request/source
closure, original220095, the original teacher corpus and frozen6K FP16 corpus live under
`artifacts/` inside the checkout. Historical absolute locators can be resolved through the
closeout relocation receipts; retained originals are not silently rewritten. The retained
records keep their original bytes and names. The maintained readers translate old
absolute locators using `w1a1_eagle/storage.py`; no external directory is required.
BitDraft is the project name; old names inside immutable receipts identify historical runs.

## BitDraft source identities

Current main includes the final legacy main history, followed by the current project's
implementation snapshots. The 52 replayed commits retain identical file trees. Philip
subsequently requested removal of Cursor coauthor trailers: five messages in each original
and replayed history were cleaned, preserving authors, committers, timestamps and topology.
`archive/pre-bitdraft` points to the cleaned original history. Exact original revisions
remain in the independent `pre-bitdraft.bundle`/`pre-bitdraft.git` under the closeout
archive. `history-rebase.json` records the earlier replay; the later
`artifacts/archive/cursor-attribution-removal-20261112/rewrite.json` maps the attribution
rewrite. Keep original IDs in immutable execution records. Relocation and attribution
cleanup do not change measured results or executed source-file identities.

The canonical checkout is `/home/philip/BitDraft`; the T3 project points there. Both old
checkout paths and the legacy quarantine are absent. Philip authorized their final removal
after independent operation and preservation checks. Retirement receipts under
`artifacts/receipts/final-old-directories-20261112/` distinguish raw-file retirement from
checkout removal and verify that shared files and retained data survived.
`origin` is the user-owned `git@github.com:philippesic/BitDraft.git`. Main preserves
legacy main ancestry; `archive/legacy-main` and `archive/pre-bitdraft` preserve the
original legacy tip and the port history with the requested attribution cleanup. Publication keeps existing repository
visibility; it does not upload ignored datasets, model files or supporting archives.

## Historical 2080 Ti records

`artifacts/archive/2080-closeout-20261112/` preserves the old 2080 Ti checkout's compact
reports, diagnostics and source, plus larger source capsules and JSON reports/indexes.
Each archive contains `preservation-index.json`; adjacent preservation receipts record
source/member SHA-256 verification. Original paths are archive members without the leading
slash. All 11 registered checkout HEADs and all seven project refs are already reachable
from this checkout; the main native submodule revision is independently archived too.

The old 20671-update checkpoint series, model copies, environments, build outputs and raw
profiles are obsolete historical artifacts, not current training inputs. Their exact
resume/runtime replay is not retained by these compact archives. The original220095 parent
and current corpora remain separately preserved in this checkout. The four old project
directories and obsolete project host/hook settings have been removed. Verified source,
receipts and shared-file preservation checks live under
`artifacts/receipts/final-old-directories-20261112/`; the temporary remote cleanup workspace
is also gone. The unrelated GPU job was left running. Private settings backups remain
ignored and access-restricted in the archive, rather than entering Git.

## Windows artifact archives

The remaining historical Windows raw-output archives have been converted to compact
evidence and retired. Start with the [retirement result](../results/historical-output-retirement/result.md),
its folder catalog and imported observations. Existing topic results remain the scientific
entrance; the conversion does not pool old workloads or certify producer verdicts.

`artifacts/archive/development-history/` contains `windows-support.tar` and
`qualification-support.tar`. Their members are individually gzipped, content-addressed
report, source, configuration and diagnostic records. `windows-measurements.jsonl.gz`
provides numerical/status fields; original report rows remain in support for statistics.
The `*.index.jsonl.gz` files map original source paths to SHA-256 values and disposition.
Nested source paths use `::` to separate archive members; repeated path versions retain
their separate content identities. `windows-metadata.tar.gz` preserves original indexes,
collection manifests and historical link records.

To recover a retained record, find its source path and `sha256` in an index. Open the
appropriate support tar with Python's `tarfile`, read the member named `SHA256.gz`,
decompress it with `gzip.decompress`, and check the decompressed bytes with
`hashlib.sha256` before writing to a separate recovery directory. Preserve link records
as locators rather than following old external paths. A `discard-raw` or `discard-output`
entry without support identifies intentionally unavailable bytes; an index or hash cannot
restore them. Qualification indexes also identify files retained at their development paths.

The original qualification index records disposition at the first retirement.
`current-development-catalog.jsonl.gz` records current retained paths, including small
specifications restored for upcoming checkpoint evaluation. Its support references name
the retained tar and member; the older index's absolute blob paths describe temporary
extraction staging that has been removed. The [result](../results/historical-output-retirement/result.md)
identifies the retired historical checkpoint bindings in the original loss recipe.

`older-source-support.tar.gz` and `older-source-index.jsonl.gz` replace two earlier
metadata/source archives that still contained obsolete model/capture/runtime payloads.
They use the same `SHA256.gz` member convention; use the index's original archive/member
pair to find the preserved record. Other independent source-history and native WIP
archives retain their original recovery procedures.

Authentication, metric recomputation, dependency decisions, exact retirement plans and
journals live in `artifacts/receipts/artifact-closeout-20261113/`. The current6K TRAIN
corpus remains physical data under `artifacts/datasets/`. The original continuation parent,
models, diagnostic DEV data and current comparison exports remain available. Retired old
weights, capture tensors, profiler timelines and raw process/event replay do not.

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

The bundles are verified local preservation, not an off-machine backup. No remote is
configured for this project. The archive lives under ignored `artifacts/archive`; include it when moving the whole
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
implementation snapshots. The 52 replayed commits retain identical trees and messages.
Original current-project revisions are reachable through `archive/pre-bitdraft` in this
checkout and the independent `pre-bitdraft.bundle`/`pre-bitdraft.git` under the closeout
archive. `history-rebase.json` maps old/new IDs; keep original IDs in immutable execution
records. Relocation changes a locator, not a measured result or its source identity.

The canonical checkout is `/home/philip/BitDraft`; the T3 project points there. Both old
checkout paths are absent. The remaining legacy main tree is quarantined under
`/home/philip/BitDraft/artifacts/archive/legacy-checkout-pending-delete` until Philip confirms
its deletion. It contains no required current data input. Delete only that owned
quarantine, without following its compatibility links into other retained evidence/data.
This remains a local-only project with no configured remote or upstream.

## Windows artifact archives

Historical run folders and transfer deliveries formerly scattered under the Windows
user directory are contained in `artifacts/archive/windows-legacy/`. Adjacent
`.index.jsonl` files record each original path, archive member, byte length and SHA-256.
Search the indexes
for a former path before opening a large archive; the receipt collection under
`artifacts/receipts/storage-containment/` records archive identities and verification.
The `binary-eagle-decoding-artifacts` collection includes its main archive and retained
numbered parts; consult its collection manifest and all matching indexes. Its separate
`.links.json` preserves symbolic-link targets, including broken historical locators.

On this host, Python3.14 reads these archives with standard-library
`tarfile`, including `python -m tarfile --list ARCHIVE.tar.zst`. Recover only required
members beneath `artifacts/archive/windows-legacy/unpacked/`, preserving the member's
original directory prefix. Archived checkout links point to that recovery location.
Verify recovered bytes against the adjacent index. Preserve symlink entries as historical
locators rather than following them into former external directories.
The locked numerical environment also supplies `zstandard`: its decompressor's
`stream_reader` can feed `tarfile.open(fileobj=reader, mode="r|")` on Python3.11.

Each source file and its decompressed archive member were checked against the same
SHA-256 before the external copy was removed. Repeated identical large files
may use tar hardlinks; their member identities remain in the index. This is lossless
relocation and compression, not a new scientific result or retirement of those bytes.
The current6K TRAIN corpus uses physical files in `artifacts/datasets/` and does not
depend on extracting this archive.

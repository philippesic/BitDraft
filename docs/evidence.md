# Reading and recovering evidence

Start with [FINDINGS](../FINDINGS.md), then the result's scope, decision and evidence.
[Source catalog](../results/sources/catalog.json) maps imported files to their original
revision, path, length and SHA-256. `.txt` reports are byte-identical archival documents,
not current instructions: their old job ownership, commands and approvals are historical.
A JSON source reference identifies a larger record or implementation in the external
Git archive; it is not a substitute for that record when recomputing a claim.

## Independent archive

The historical project and native-source bundles are stored outside active source at
`/home/philip/binary-eagle-evidence/legacy-20261110/`. Their identities, recovery checks
and exclusions are recorded in the [archive receipt](../results/archive-preservation/result.json).
The new repository tracks compact reports and conclusions. The external archive retains
full committed history, including old diaries, proposed designs, implementation and
supporting records not copied into the ordinary source tree.

Both bundles were restored into independent bare repositories and checked with
`git fsck --full`. They do not borrow objects from the original checkout. A source record
can be read without that checkout or network access:

```sh
git --git-dir /home/philip/binary-eagle-evidence/legacy-20261110/recovered.git show REVISION:PATH
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
independently preserved under `/home/philip/binary-eagle-evidence/legacy-final-20261111/`
with its own `preservation.json` and verified bare reader. Final working reports and
the original run/data now live under
`/home/philip/binary-eagle-artifacts/datasets/retained-he4broad-original/`; its
`closeout-reports-preservation.json` verifies the copied reports. This includes the
220095-update,15647-capture run, distinct from the November8 forty-sequence namesake.

The bundles are verified local preservation, not an off-machine backup. No remote is
configured for this project. Preserve the external archive when moving or deleting a
checkout. Neither archive adoption nor the document cleanup authorizes deleting live
jobs, original model inputs or ignored execution records.

## Compact legacy artifact evidence

The [legacy closeout](../results/legacy-artifact-closeout/result.md) records the later
explicitly authorized artifact retirement. Its exact receipts supersede older statements
that every historical checkpoint or raw run remains available. Historical measurements
and source identities are unchanged; discarded optimizer states cannot be resumed and
raw tensors/events not retained cannot be recomputed from a hash.

`/home/philip/binary-eagle-evidence/legacy-closeout-20261112/` contains deduplicated compact
support, source snapshots, training-curve aggregates, failure diagnostics and exact
worktree WIP patches. Compressed indexes map the original path to `objects/SHA256` in the
corresponding tar archive. The tracked closeout receipt authenticates these files. Look up
the relevant old path in the index, then read just that object, for example:

```sh
tar -xOzf /home/philip/binary-eagle-evidence/legacy-closeout-20261112/compact-support.tar.gz objects/SHA256
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
closure, original220095, the original teacher corpus and frozen6K FP16 corpus remain at
their external artifact homes. Historical absolute locators can be resolved through the
closeout relocation receipts; retained originals are not silently rewritten. The external
storage roots keep their established names to preserve authenticated identities. BitDraft
is the current project name; old names inside immutable receipts identify historical runs.

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

The November 10 investigation was unfinished at the archive boundary. A separately
hashed working-report snapshot is retained externally as interim evidence, not a final
finding. Later changes in that active investigation are outside this snapshot.

The bundles are verified local preservation, not an off-machine backup. No remote is
configured for this project. Preserve the external archive when moving or deleting a
checkout. Neither archive adoption nor the document cleanup authorizes deleting live
jobs, original model inputs or ignored execution records.

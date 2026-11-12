# Historical reports and committed source are recoverable without the old checkout

## Outcome

The adoption preserved all 375 tracked Markdown documents at the recorded archive head
and complete reachable project/native Git histories in independently restored bundles.
The project now contains 30 research findings, including 17 additional historical summaries,
and 67 byte-identical supporting documents. The catalog records every tracked Markdown
path and distinguishes copied reports from history retained only in the external archive.

This makes prior documented knowledge and committed implementation accessible without the
original checkout or GitHub. Old campaign ownership and proposals remain archival context,
not current development instructions. Routine implementation history stays in Git.

## Scope and limitations

This is documentation and source preservation, not experimental reproduction. Editorial
review used report outcomes and consolidated campaign/feature catalogs; it is not a new
audit of every raw request or a reread of every diary entry. The external bundles omit
ignored raw runs, model weights and capture datasets. Some states were intentionally
retired before this adoption. The original November 10 investigation was still unfinished;
two working records were snapshotted separately, and later work is outside this archive.

The archive is local and verified, not backed up to another machine. The original live
checkout and jobs were untouched. It is not safe to delete the original artifact trees
on the strength of this result.

## Decision

Retain compact result pages and local supporting reports as the normal research entrance.
Retain the external evidence bundles for source and deeper historical context. Remove
obsolete migration tasks and routine overhaul receipts from active project navigation;
Git preserves those implementation records. Keep current capability limits visible in
current documentation rather than implying that cleanup completed missing functionality.

## Evidence and reproduction

- [Archive identity and verification receipt](result.json)
- [Source catalog and complete tracked Markdown inventory](../sources/catalog.json)
- [Evidence access and recovery procedure](../../docs/evidence.md)
- [Current supported behavior](../../docs/support.md)

The receipt gives bundle locations and hashes. Restoring each bundle into an independent
bare repository and running `git fsck --full` passed. Selected source bytes were recovered
from that restored repository, not from a live mutable checkout. No historical experiment
was rerun and no original model, capture, source file or run record was deleted.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

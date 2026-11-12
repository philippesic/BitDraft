# Historical results survived cleanup but many exact model states did not

## Outcome

The November 10 record reports 443 retired model artifacts with 543.36 GiB logical size. It preserves current HE4 work, selected controls, HE4/Csmallbroad endpoints and supporting measurements. Earlier optimizer intermediates cannot be assumed resumable merely because reports or native exports survive.

## Scope and limitations

Historical retirement receipt, not a fresh filesystem inventory or physical-space measurement. Hard links make logical bytes differ from reclaimed disk. Current active run ownership remains with its original task.

## Decision

Retain deletion evidence and dependency identities; no new artifact deletion follows.

## Evidence and reproduction

- [storage-cleanup-20261110/report.md](../sources/experiments-storage-cleanup-20261110-report-fc4853c7.txt)
- [storage-cleanup-20261106/report.md](../sources/experiments-storage-cleanup-20261106-report-0a2b5afe.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

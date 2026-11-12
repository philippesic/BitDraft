# Fresh-source HE4broad remained slower than target-only and baseline

## Outcome

It ended at 24.5658% acceptance and 174.113 tok/s, against 185.444 for target-only and
65.3257% and 249.662 tok/s for Baseline. Training ran 226,853 updates over 40 sequences,
3,234 windows and 3,794 unique labels, a mean exposure of 896.889. DEV top-1 peaked at
17.7778% at hour 4 and ended at 13.3333%; final cross-entropy was 8.899076. Formal serving
was scored only at hours 0 and 8, so the best checkpoint was never selected, although
hourly full states were kept.

## Scope and limitations

DEV3, one seed and trajectory. The hour-4 DEV peak does not establish a formal
serving peak. No HE4 win over Baseline.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[experiments/he4broad-20261108/report.md](../sources/experiments-he4broad-20261108-report-4887a5fb.txt) [experiments/he4broad-20261108/audit.json](../sources/experiments-he4broad-20261108-audit-2c53389f.json) [experiments/he4broad-20261108/results.json](../sources/experiments-he4broad-20261108-results-json-3d7147f1.json) [experiments/he4broad-20261108/manifest.json](../sources/experiments-he4broad-20261108-manifest-a44a5efe.json)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. [Evidence access](../../docs/evidence.md) describes archive recovery and remaining
raw-input limits; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

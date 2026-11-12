# Execution speedups have not compensated for model quality

## Outcome

Not so far. The learned-small GPU forward pass improved 17.2% [15.5, 18.8] and the full
request 2.51% [1.37, 3.50], yet it ran 176.21 tok/s against Baseline's 244.39. Some
fusions changed packed bytes and were rejected.

## Scope and limitations

GPU replay, host trace and full HTTP request clocks differ. natural12 is consumed
DEV. Learned-small is not HE4, and the old DSpark head precision differs.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[experiments/execution-campaign-20261105/report.md](../sources/experiments-execution-campaign-20261105-report-b4d86eca.txt) [experiments/execution-campaign-20261105/ledger.json](../sources/experiments-execution-campaign-20261105-ledger-json-717c76e6.json) [experiments/execution-campaign-20261105/historical-latency-context.md](../sources/experiments-execution-campaign-20261105-historical-latency-context-e0f1a6bf.txt)

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

# Retained HE4 and Csmallbroad states offer tradeoffs, not a baseline win

## Outcome

No. On heldout requests HE4 ran 167.06 tok/s, Csmallbroad 168.95 and Baseline 236.67.
Relative to Baseline, HE4 scored 0.706 [0.690, 0.724] and Csmallbroad 0.714
[0.696, 0.734]. HE4's GPU forward pass improved 14.88% against H0, and its private
vocabulary adds 341.02 MiB. Both are tradeoff points, not deployment winners.

## Scope and limitations

24 clusters that were fresh then and are consumed now, 5 process repetitions.
Beating H0 is not beating Baseline.

## Decision

Retain the development tradeoffs and external deployment baseline; no promotion.

## Evidence and reproduction

[experiments/component-campaign-20261105/report.md](../sources/experiments-component-campaign-20261105-report-4a5baa3b.txt) [experiments/component-campaign-20261105/results.json](../sources/experiments-component-campaign-20261105-results-json-8cc5d76d.json)

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

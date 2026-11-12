# Broader data and simple objective additions did not establish serving gains

## Outcome

The 192-group broader-data continuation changed consumed natural24 throughput by -0.32% with interval [-1.61%, +0.80%] at 1,920 additional updates. Root-heavy depth weighting and label smoothing also failed to establish a benefit in their matched controls. The historical catalog contains 98 segments across 34 recipe families; 80 segments used under 300 TRAIN seconds.

## Scope and limitations

Fixed row-W1A8 scale-only model on RTX 5080, one trajectory per treatment and consumed DEV panels. Short exploratory tests are not universal feature rejections, and data breadth does not isolate data quality.

## Decision

Retain controls and uncertainty; do not add these features by default or permanently discard them.

## Evidence and reproduction

- [data-cost-transfer-20261104/README.md](../sources/experiments-data-cost-transfer-20261104-readme-fb52150c.txt)
- [design-readiness-20261104/readiness.md](../sources/experiments-design-readiness-20261104-readiness-6ed200d8.txt)
- [design-readiness-20261104/backlog-audit.md](../sources/experiments-design-readiness-20261104-backlog-audit-88f65969.txt)
- [design-readiness-20261104/decision-feature-map.md](../sources/experiments-design-readiness-20261104-decision-feature-map-917b5c54.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

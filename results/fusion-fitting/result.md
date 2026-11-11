# Better fusion reconstruction did not establish better native acceptance

## Outcome

Real CPU fitting reduced validation relative squared error from 0.074499 for converged scale-only to 0.054397 for fitted signs/scales (26.98% lower). Every validation group improved, but direction/ranking metrics still trailed the original initializer on some measures.

## Scope and limitations

384 native F32 rows from 12 TRAIN groups, split eight fit/four validation; validation has no code groups. Apple CPU fitting and export, not native quality or throughput evidence.

## Decision

Retain the reconstruction finding; leave native benefit unresolved.

## Evidence and reproduction

- [fusion-binary-real-a8-2026-10-03/README.md](../sources/experiments-fusion-binary-real-a8-2026-10-03-readme-5092da5e.txt)
- [fusion-binary-real-a8-2026-10-03/independent-validation.md](../sources/experiments-fusion-binary-real-a8-2026-10-03-independent-validation-d8afa196.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

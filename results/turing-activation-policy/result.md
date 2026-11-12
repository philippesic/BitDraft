# Changing activation precision alone did not close the W1Ax serving gap

## Outcome

All twelve policy cells completed 11,520 measured requests. Every W1Ax mode trailed both matched FP16 and Q4_0 anchors on pooled request and decode throughput. The largest within-cell decode ratios were 0.707 of FP16 and 0.684 of Q4_0.

## Scope and limitations

RTX 2080 Ti; fixed binary signs and row scales, activation-only changes; no QAT. Output divergences and control timing variation remain part of the evidence.

## Decision

Retain this bounded negative result; it does not reject trained activation-precision designs.

## Evidence and reproduction

- [w1ax-activation-precision-results.md](../sources/experiments-w1ax-activation-precision-results-4c7c5fb7.txt)
- [w1ax-policy-grid-results.md](../sources/experiments-w1ax-policy-grid-results-440709aa.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

# Fixed clipping beat learned clipping in one matched held-out comparison

## Outcome

At 1,024 updates, fixed unit bounds achieved 167.51 versus 162.21 request tokens/s for learned bounds: ratio 1.0327, paired 95% interval [1.0180, 1.0484]. DEV gave no clear difference: 0.9999 [0.9817, 1.0198]. Q4 remained faster at 245.61 tokens/s.

## Scope and limitations

RTX 5080 compact G4/D8, matching exposure and initializer, five repetitions and a distinct clip-heldout24 cohort. One training history; changing bounds during training also changes the trajectory.

## Decision

Retain fixed bounds as a control, without declaring a universal advantage or promoting the binary model.

## Evidence and reproduction

- [precision-campaign-20261104/clip-control-report.md](../sources/experiments-precision-campaign-20261104-clip-control-report-77a98b36.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

# Released seven-token DSpark and DFlash beat Q4_0 EAGLE on Turing

## Outcome

DFlash D7 reached 132.37 request tokens/s versus Q4_0 EAGLE 90.80 (+45.8%); DSpark D7 reached 129.49 (+42.6%). FFN-Q4 variants also won their separately paired phase. All four D7 conditions matched baseline output IDs on 144 measured requests each.

## Scope and limitations

RTX 2080 Ti, historical Qwen3-4B F16 target, 24 DEV prompts and six repetitions. These are released and FFN-Q4 models, not W1A8/W1A1; phase-specific anchors must remain separate.

## Decision

Retain a positive architecture reference without transferring its rates to HE4 or modern targets.

## Evidence and reproduction

- [dspark-sm75-20261003/results.md](../sources/experiments-dspark-sm75-20261003-results-82c7d454.txt)
- [dspark-sm75-20261003/reference-results.md](../sources/experiments-dspark-sm75-20261003-reference-results-97e9318d.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

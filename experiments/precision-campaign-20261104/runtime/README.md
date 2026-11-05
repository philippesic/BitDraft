The campaign runtime is an isolated relink, not an unpublished submodule commit.
Reproduce with `scripts/dflash_baseline/build_precision.py NEW_CANONICAL_RUN_DIR`.
It reuses the frozen `runs/w1a8-modern-packed-runtime-20261103-01/build02` objects
and baseline `runs/dflash-modern-native-20261104-03/dflash.cpp`, preserving target
and verifier arithmetic. Apply the vanilla matched-modern.patch before the
DFlash model portion of this patch if reconstructing source manually.

The patch adds per-projection A1/A2/A4/A8, fixed/learned clipping, packed-stream
sums and row midpoints, and positive gains fused into the custom pack kernel.
The fused gain uses explicit F32 RNE multiplication with FTZ disabled. Generic
CUDA gain multiplication in runtime03 failed tiny-value gained-stream checks;
its earlier diagnostic results are retained as a superseded series. Runtime04
passes all36 gained-stream cases byte-exactly, including zero/subnormal inputs,
nonzero midpoints and actual gate/up/down geometries. The CPU custom path rejects
fallback. A1/A2 execute INT8 codes through DP4A; no binary-activation speed claim.

Actual source, compilation/link commands, libraries and hashes are retained in
`runs/precision-native-20261104-04/build.json`. The native model/serving panels,
not compilation alone, determine campaign outcomes.

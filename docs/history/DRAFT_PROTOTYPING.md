# DFlash prototype implementation checkpoint

> Historical record: retained for context, not current instructions or ownership.
> Follow the current STATUS and campaign for live work.

This document preserves the original CPU checkpoint history and the current
qualified native-v2 contract below. See PRECISION_CAMPAIGN.md for measured work
and live ownership; the original CPU-only capability table is historical.

## Original CPU checkpoint

`python -m scripts.dflash_baseline.prototype resolve CONFIG.json` resolves a
version 1 request. Required fields: `version`, `preset`, `source`, `parent`,
`data`, `budget_seconds`. Each pin contains `path` and a 64-character `sha256`.
`prototype_model.authenticate` verifies file contents against pins; the resolve CLI only validates structure.
Presets: `vanilla`, `essential-a8`, `gate-up-a4`, `all-a4`, `a2-stress`,
`protected-example`. Explicit projection overrides win over preset coverage;
feature switches win over preset features. Only W1 with A2/A4/A8 or protected
BF16 with null activation bits is accepted. A2 remains diagnostic.

The standalone Projection implements nonnegative row scales, row midpoints,
positive exponential input gains and symmetric per-token dynamic quantization.
Fixed clipping preserves the supplied bound exactly; learned clipping uses an exponential positive bound. Rounding is RNE. Dynamic quantization preserves the baseline reciprocal/multiply and subnormal fallback. The model installer shares gate/up gains. Both affine terms consume the same quantized stream.

| Capability | Status |
| --- | --- |
| Resolved coverage and standalone affine arithmetic | CPU tested |
| Learned clipping gradients and state persistence | CPU tested |
| Model module replacement and trainable groups | Six-layer small fixture CPU tested; real model untested |
| Exact checkpoint resume | Small fixture optimizer/RNG/cursor CPU tested |
| Explicit parent adaptation | Pending |
| Packed GGUF export/import for new contract | Pending |
| Native affine/A4/A2 loader and operations | Unsupported |
| CUDA build for new contract | Not performed |
| Native GPU qualification | Pending implementation and owner release |
| Manifest benchmarking and matched adaptation runner | Pending |

`native-screen` fails explicitly. It must not invoke the old runtime on these
configs. There is no safe ready-to-run native command until export/runtime work
is complete. Legacy vanilla compatibility is protected by leaving its code
unchanged; A8 projection outputs and codes match the original baseline in the tested finite/zero/subnormal fixture; full-model equivalence remains untested.

CPU check (no pytest dependency):

```
PYTHONPATH=. /home/philip/binary-eagle-decoding/.venv/bin/python - <<'PY'
import runpy
for name, test in runpy.run_path('tests/test_dflash_prototype.py').items():
    if name.startswith('test_'):
        test()
        print(name, 'PASS')
PY
```

Integration: continue in this worktree, keep baseline owner's source/run/state
untouched, and complete the pending components before describing this as the
requested end-to-end pipeline. No new goal or baseline allocation is created.

## Config runner and packed reference IO

The subsequent implementation adds `python -m scripts.dflash_baseline.pipeline`.
See [runner report](../../experiments/dflash-runner-packed-20261104/report.md) for
commands, manifest schema, tested behavior and remaining acceptance gaps.
This supersedes the earlier statement that no runner or packed import exists;
it does not establish full-model or native qualification.

## Native campaign qualification — November4,2026

The earlier CPU-only and fail-closed statuses describe the retained prototype
checkpoint, not the current campaign implementation. `build_precision.py`
relinks isolated matched runtime objects without editing the frozen runtime or
submodule. `export_precision.py` writes actual native GGUF `dflash.w1ax.version=2`
with per-projection A1/A2/A4/A8, gains, midpoints and dynamic/fixed/learned bound
metadata. The PyTorch `packed.pt` container remains reference interchange only.
`native-screen` now checks v2 eligibility; this static check cannot replace
artifact-specific load, CUDA and native evaluation evidence.

36 actual CUDA operation checks,12 CUDA gradient cases, actual full-model
update/exact parameter-and-optimizer resume/adaptation, and native A8/A1 decoded
output equality passed. Pins and exact scopes live in
`runs/precision-campaign-20261104-01/{qualification.json,native-qualification/}`.
Current campaign runtime is `runs/precision-native-20261104-04`. Its fused
gain/pack avoids the tiny-value flushing found in the earlier generic gain op;
all36 gained-stream CUDA cases pass byte-exactly. Runtime03 evidence remains
a separate diagnostic series. CPU fallback for custom affine operations is
explicitly rejected. A1/A2 still execute INT8 codes
with DP4A, not a packed binary activation kernel. Both affine terms use the
same gained quantized stream; native packing computes its sum once.

A1 dynamic levels are ±row-absmax with threshold0, nonnegative positive, no zero
code; all-zero input rows get scale0 and dequantize to0. Fixed/learned A1 uses
±positive bound (zero input maps positive). A2 is the separate symmetric ternary
grid. A4/A8 symmetric signed grids use RNE, saturation and zero point0. Dynamic
input STE is identity through gained input, including zero/subnormal rows;
fixed/learned STE differentiates the clipped normalized stream and scale.
Weight sign STE and nonnegative row-scale projection remain unchanged.

`pipeline.py --device cuda` authenticates source/parent/target/capture files,
filters TRAIN, supports baseline conversion (`--baseline-parent`), v2 adaptation
(`--adapt`, optimizer reset) and exact resume (`--resume`). Adaptation inherits
the parent's data order/cursor; segment updates and supervised positions remain
separate. Signal/pause saves occur at complete update boundaries. Final compact
checkpoints exclude frozen buffers and retain optimizer/RNG/cursor/config/lineage.
`--native-points` preserves native intermediate exports without duplicating the
whole optimizer at each diagnostic point. Lossless `.pt.zst` and versioned
byte-shuffled `.pt.bsz` archives restore exact raw checkpoint bytes; hash checks
precede deletion of an owned uncompressed copy.

Protected component formats are a finite queue (Q8_0/Q4_0 for fusion, attention
Q/K and V/output; private Q4_0 head). Training really quantizes these frozen
weights. Target weights never change; private head bytes are charged. Embedding
reduction and protected BF16 FFN exceptions remain outside native v2 admission.
Component-specific full-model native qualification is required before treating
an arm as measured. See [PRECISION_CAMPAIGN.md](../PRECISION_CAMPAIGN.md) for live
owner state, measured progress, exposure policy and remaining full-chart work.

Lossless native export archival is a storage container, separate from both
GGUF tensor quantization and the PyTorch packed reference container. It uses a
pinned frozen native GGUF dictionary and restores the entire original GGUF byte
stream with SHA256 verification before native loading. Raw GGUF bytes, including
metadata/private copies, remain the model-size metric; compressed archive bytes
are never used as an inference-size or speed claim. Actual full-model archived admission now passes40 responses with same-process
warmups, exact target output signatures and unchanged plain/traced counters in
`runs/precision-campaign-20261104-01/archive-native-qualification`. Every trained parameter/optimizer checkpoint is retained.

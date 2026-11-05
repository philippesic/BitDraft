# DFlash prototype implementation checkpoint

This is an incomplete CPU foundation, not a runnable training/export pipeline.
Legacy baseline source and runtime are unchanged. No GPU work was launched.

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
See [runner report](../experiments/dflash-runner-packed-20261104/report.md) for
commands, manifest schema, tested behavior and remaining acceptance gaps.
This supersedes the earlier statement that no runner or packed import exists;
it does not establish full-model or native qualification.

# Runner and packed reference implementation

Recovered the three requested commits in order. Preserved previous worktree,
vanilla baseline files, GPU owner, and submodule gitlinks. No push.

Implemented `scripts.dflash_baseline.pipeline`: config-resolved real Model/Capture
training on CPU, bounded cumulative updates/windows/update time, atomic compact
parameter checkpoints, optimizer/RNG/order/cursor/token accounting, exact resume,
and explicit v2 parent adaptation with optimizer reset and recorded ancestry.
Frozen model buffers are reloaded from authenticated source/target GGUFs.
The original capture result.json records are accepted directly; TRAIN filtering
and all recorded file hashes are checked. `evaluation.target` supplies the target
path/SHA256 pin. Source, parent and data pins retain the existing config schema.
For from-source initialization, parent must pin the source file; adaptation needs
an actual pipeline v2 checkpoint. All six existing presets are supported.

Packed reference format v1 stores little-bit-order binary signs (positive=1),
row scales, midpoints, log gains, clipping parameters, activation bits, quantizer
contract, norm parameters and protected projection tensors. This is a PyTorch
reference container, NOT a GGUF runtime extension. Canonical projection coverage
uses GGUF naming; extension identifier is `dflash.packed.reference`. Import
reconstructs signs for inference; latent magnitudes and optimizer belong in the
training checkpoint. Source-reloaded frozen tensors are not duplicated. Import
requires an installed model with matching resolved config. Native admission
continues to fail closed.

Runnable commands (use authenticated JSON config pins, absolute artifact paths):

```
PYTHONPATH=.:src /home/philip/binary-eagle-decoding/.venv/bin/python -m scripts.dflash_baseline.pipeline config.json --out /tmp/dflash-check --updates 2 --max-windows 2
# exact restart: same config and larger cumulative limits, new output directory
PYTHONPATH=.:src /home/philip/binary-eagle-decoding/.venv/bin/python -m scripts.dflash_baseline.pipeline config.json --out /tmp/dflash-resume --updates 4 --max-windows 4 --resume /tmp/dflash-check/checkpoint.pt
# adaptation: config.parent pins checkpoint.pt; select new preset
PYTHONPATH=.:src /home/philip/binary-eagle-decoding/.venv/bin/python -m scripts.dflash_baseline.pipeline adapted.json --out /tmp/dflash-adapt --updates 2 --max-windows 2 --adapt
```

Verification: seven inherited test functions pass. New fixture tests execute
updates, exact stochastic restart and packed arithmetic round trips across all
nine A2/A4/A8 × dynamic/fixed/learned combinations, plus protected tensor restore.
Retained actual source GGUF metadata opened successfully: 1,279,872,704 bytes,
69 tensors, 18 FFN matrices, taps 2/6/10/14/18/22/26/30. Authoritative capture
manifest contains 27 records, 24 TRAIN. No actual full-model forward/update or
full CLI fixture was exercised yet; adaptation ancestry/reset still needs an
integration test. These are explicit remaining verification gaps, not completed
milestone acceptance. Full-model CPU loading can consume substantial host memory.

CPU verified: arithmetic, update helper, checkpoint restart, reference packed IO.
Actual model exercised: metadata only. CUDA compiled: no. GPU qualified: no.
Native loader/operations, GGUF extension integration, CUDA build and matched
native throughput/acceptance benchmarks remain outstanding. No latency or
acceptance measurement is claimed. No GPU processes started or stopped.

# Implementation checkpoint

The requested baseline is **not complete**. No zero-update export, optimizer
update, training checkpoint, capture or native evaluation was produced. The
1800-second allocation remains entirely unspent. No GPU job was launched.

The local RTX5080 was verified through `/usr/lib/wsl/lib/nvidia-smi`: 16303MiB
capacity, 1352MiB occupied, zero utilization and no reported compute process.
The local5080 pause flag was false. No host settings were changed.

`source-inventory.json` authenticates the actual local BF16 initializer and all
tensor names/shapes. The reusable inspector in `scripts/dflash_baseline/` ran
successfully against that artifact. This is source inspection, not qualification.
Actual topology: six layers, eight taps, eighteen FFN projections, block size16,
FFN width9216, rope base10000000 and sliding-window4096 in the first five layers.
The artifact does not declare the historical `sample_from_anchor` field.

Existing `block_qat.py::BlockQATConfig`, `block_training.py::load_block_gguf`
and native `dflash.cpp::load_arch_tensors` enforce historical geometry. A simple
layer-count relaxation is insufficient: modern mask/position layout, shared
head/embedding, sliding-window behavior, objective and native capture contract
must be implemented and qualified together. The released inventory supplied as
an entry point identifies historical Qwen3 weights; it must not replace this
modern source. Existing EAGLE captures were not opened or admitted.

Next work: recover and pin the compatible modern DFlash author training graph
and objective; implement dedicated modern capture/QAT/export/native adapters;
qualify update/sign persistence/resume/native outputs; then run the authorized
1800 measured update seconds and matched native panels. This checkpoint does
not resolve these engineering dependencies or claim a finished baseline.

No runtime changes, gitlink changes, push, shared-file edits or owned background
processes. GPU ownership is released. Frozen external baseline remains unchanged.

## Packed zero-update serialization produced

Implementation now includes `scripts/dflash_baseline/export_zero.py`. Actual
462MiB artifact and audit: `/home/philip/binary-eagle-decoding/runs/dflash-modern-20261104-02/{zero.gguf,zero-audit.json}`.
All18 FFN matrices use nonnegative-is-positive packed signs and F32 row mean
absolute scales; original BF16 attention/fusion and F32 norms remain byte-identical.
Packed signs/scales pass byte-hash roundtrip. Shared head/embedding remain external.
Native admission is **not qualified**; no throughput or correctness claim.
Author graph downloaded from https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py
into the run source directory; this moving-source download is not yet a qualified
training-source pin. No training/capture/evaluation performed;1800seconds unspent.
No GPU process launched. Training/resume, modern capture, native adapters and
matched evaluation still require implementation. This is partial work, not completion.

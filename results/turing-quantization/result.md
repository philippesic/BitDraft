# Packed low-bit paths did not beat ordinary EAGLE in the Turing suite

## Outcome

All five native W1A1 coverage settings were slower than ordinary EAGLE. W8A8 was effectively tied; W4A4 lost acceptance. Q4_0 and Q8_0 GGUF controls were faster, but they use different arithmetic. Specialized matrix instructions did not improve whole-request speed in this single-sequence workload.

## Scope and limitations

RTX 2080 Ti, Qwen3-4B F16 target, 12 frozen prompts, five measured repetitions, 128-token cap. Each ratio uses its own paired anchor; this is not a modern HE4 result.

## Decision

Retain the measured negative result; do not generalize it to larger batches or different kernels.

## Evidence and reproduction

- [rtx2080ti-synthesis.md](../sources/experiments-rtx2080ti-synthesis-4cae21cb.txt)
- [rtx2080ti-quantization-suite.md](../sources/experiments-rtx2080ti-quantization-suite-4c56c3bc.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

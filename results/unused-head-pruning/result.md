# Skipping unused catch-up head work gave a small matched throughput gain

## Outcome

Two opposite-order blocks found zero raw-output/counter mismatches across 480 paired requests each. Order-balanced request-throughput ratios were 1.0084 for Q4_0, 1.0499 for fitted D and 1.0110 for FP16, versus 1.0012 for target-only. The patch did not close the binary model quality gap.

## Scope and limitations

RTX 5080, historical Qwen3-4B F16 target, 24 reused DEV prompts and five repetitions. Geometric means of order-block ratios are descriptive, not new confidence intervals. Exact cache-byte and stage-attribution gates remained separate.

## Decision

Retain the opt-in runtime finding; do not treat it as universal serving capacity or a new default.

## Evidence and reproduction

- [eagle-prune-timing-5080.md](../sources/experiments-eagle-prune-timing-5080-ad64ee36.txt)
- [eagle-prune-quality-5080.md](../sources/experiments-eagle-prune-quality-5080-71d8305b.txt)

Imported reported measurements, not rerun. [Evidence access](../../docs/evidence.md)
describes independent source recovery and the remaining raw-artifact limits.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

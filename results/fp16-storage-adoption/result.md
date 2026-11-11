# FP16 feature storage passed the scoped HE4 qualification

FP16 can be used for the authorized supplemental captures from the Qwen3.5-4B Q4KM
target. It halves feature-array storage while retaining FP32 materialization and training.
The matched qualification passed every margin frozen before acquisition.

## Outcome

Both arms continued the same final original HE4broad optimizer state for 1,024 updates
on the RTX 5080, with identical task/position sampling over 64 diverse TRAIN captures.
The loss check used eight fixed windows from each of 32 separate DEV captures, always
with the same original FP32 teacher features. Every category passed its loss margin.

| Measurement | FP32 storage arm | FP16 storage arm |
| --- | ---: | ---: |
| Task-mean DEV loss | 3.44779 | 3.43955 |
| Native accepted/proposed | 17,826 / 37,901 | 17,821 / 37,912 |
| Native acceptance | 47.0331% | 47.0062% |
| Requests matching target tokens and stopping | 120 / 120 | 120 / 120 |
| Native request tokens/s | 180.978 | 176.848 |

The paired source-group acceptance difference was −0.027 percentage points, with
a 95% bootstrap interval from −0.277 to +0.235 points. The throughput ratio was 0.9772,
with interval 0.9688–0.9845. Both passed the predeclared margins. All 512 converted arrays
passed the rounding bound; token labels and data roles stayed unchanged.

## Scope and limitations

This is one seed and 1,024 updates, not lifetime convergence equivalence. The public
training path, checkpoint retention, GGUF exports and native request path all executed.
The native timing used one serial block; it does not establish a storage-caused speed
effect. FP16 here describes teacher-feature files. Both deployed drafters use HE4 formats.
The NVFP4 target and different architectures require separate qualification.

The first DEV-loss attempt retained all measurements but refused certification because
source files changed during execution. A rerun from a frozen snapshot reproduced every
loss and completed successfully. Benchmark commands now execute from frozen snapshots.

## Decision

Retain FP16 for this target and recipe. Preserve the FP32 qualification references and
the original recovery state. Resume production from original update 220095; the probe
descendants are experimental controls and must not replace that starting point.
Earlier bounded evidence remains in [capture qualification](../fp16-capture-qualification/result.md).

## Evidence and reproduction

[Measurements, gates and authenticated raw-result references](result.json) identify the
executed inputs and source snapshots. Reproduce with the pinned capture conversion,
matched public training intents, `benchmarks evaluate` loss specification and
`benchmarks compare` short-panel specification. Each raw output retains its commands,
source snapshot, runtime identities, responses and checkpoint/export manifests.

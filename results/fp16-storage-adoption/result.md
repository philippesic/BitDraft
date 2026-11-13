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

## Canonical export followup, 2026-11-11

Export-time exponentials now run in CPU FP32, reproducing the original export exactly.
Re-exporting the same two 1,024-update states and rerunning the frozen 120 requests gave
17,827 accepted drafts for each arm: 37,901 proposed for FP32 and 37,900 for FP16.
All outputs/stops matched. FP16/FP32 throughput was 1.0053, with paired 95% interval
1.0011–1.0095; acceptance difference was +0.0012 percentage points, interval
−0.2750–+0.2526. Every original qualification gate still passed. Earlier executions above
remain unchanged; serial timing does not establish a storage-caused speed effect.
Raw results and stratified bootstrap are in
`/home/philip/BitDraft/artifacts/qualification/precision-canonical-paired.json`.

## Artifact disposition

Retain the compact measurements and executed source linked above. Reusable saved
controls and active data/model inputs retain their dependency closure outside source.
Qualification-only descendants are disposable after their measurements and failure
receipts are verified; exact retired states cannot be resumed.

November13,2026 disposition: [historical-output retirement](../historical-output-retirement/result.md)
keeps the recorded comparison arms, per-request measurements, diagnostics and executed
source while retiring qualification-only optimizer states, unused exports and capture
fixtures. Current original/6K corpora and the separate DEV corpus used by training
diagnostics remain physical inputs. Exact discarded qualification-state resumes are
unavailable; historical measurements and their scope are unchanged.

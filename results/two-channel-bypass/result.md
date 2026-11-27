# Bypassing two channels leaves a small, uncertain gain after matched training

At 21,000 updates the gate/up bypass scored 53.49% simulated acceptance against
52.90% from the unchanged run on the same seed. The +0.58-point difference
(95% source-group interval −0.07 to +1.42) fell below the one-point threshold
for pursuing the treatment. The large measured advantage at initialization did
not persist at the same magnitude after training.

## Outcome

| One-bit arm | Updates | Offline acceptance, 32 DEV requests | Native acceptance, 120 DEV requests |
| --- | ---: | ---: | ---: |
| Unchanged, seed 8101 | 21,000 | 52.90% | 46.75% |
| Unchanged, seed 8102 | 21,000 | 52.81% | 46.61% |
| Two-channel bypass, seed 8101 | 21,000 | 53.49% | — |

The unchanged seeds differed by −0.09 points offline (95% interval −1.06 to +0.75)
and −0.14 points on the served panel (−0.47 to +0.17), relative to seed 8101.
Every native panel request had exact target output. The bypass was +0.68 points
against seed 8102, a different data order (95% interval +0.01 to +1.47).

The pilot finished 64 actual optimizer updates and restored its checkpoint in
the same CUDA replay used for the matched runs. It scored 47.19% against 10.31%
for the historical unchanged untrained start. This qualified the path through
real updates, checkpoint and replay, with different training lengths for the two
pilot scores. The [start analysis](../claim-comparators/result.md) measured the
bypass at equal zero updates and found a +20.17-point difference.

## Scope and limitations

Every full arm used 21,000 updates on train-final-02-fused with four training slots,
cosine decay with a 420-update linear warmup, authenticated Baseline attention and
fusion tables, and the same initialization inputs. The two seed-8101 runs consumed identical trajectories
and 3,376,456 labels. The bypass changes gate/up by masking input channels 0 and 850
in the one-bit computation and carrying them through exact F32 source columns. It
also masks the two channels during rounding calibration. The trainer took roughly
88 minutes per arm. It used the same RTX 5080 and Qwen3.5-4B Q4KM features.

The bypass was trained on one seed. The two unchanged seeds estimate the observed
spread only. The offline simulation used depth 3 on 32 reused DEV captures. Its
acceptance percentages and differences have a separate denominator from the served
DEV panel. There is no bypass throughput result, native bypass kernel or protected
confirmation result here.

## Decision

Keep the current release recipe. The matched point estimate misses the one-point
criterion and the same-seed interval includes zero. The experiment remains available
if a later architecture change creates a reason to revisit these channels.

## Evidence and reproduction

The unchanged train source is
`8c3d9db551d78293b5e9ec5fc928d49042a74bcd80086c4d53393d86c595b53b`
at Git revision `812ff0ead`. The bypass and all four offline replays use
`997d280c024845620f0d5d35b341d1ea259930710b94316bb3a8c4cc003a9253`
from `f2f23dcf` on `experiments/two-channel-bypass`. The source patch passed
the CPU gate and the GPU pilot. Run intents, chained events, train reports,
replay specifications and restore receipts live under
`artifacts/one-bit-attribution-01/arms/{w1-21000,w1-21000-s8102,w1-bypass-pilot,w1-bypass-21000}/`.
The campaign's `scripts/train.sh` and `scripts/offline-score.sh` use the frozen source,
the normal training controller and the maintained round replay. The unchanged panel
uses runtime-v8. Both bypass exports use the diagnostic-only architecture marker.

[Compact evidence](result.json) records the exact recipes, data and source identities,
all comparison arms and the data needed to recreate the statistics.
Use `w1a1_eagle.drafting.paired_interval` with 2,000 draws and seed 20261121
for the trained comparisons. Align the offline rows by request ID, then resample
the source groups, preserving accepted and proposed counts.

## Artifact disposition

Keep the full bypass state and its inputs through the release recipe decision so the
owner can reproduce the comparison or revisit serving without retraining. Keep the
pilot state and diagnostic export through that same review as the qualification fixture.
Keep seed 8102's state and native export as the independent data-order control while
the 21,000-update comparison is used. The project owns these stores and retires them
through checkpoint retention once those uses end. Preserve the compact evidence,
frozen sources, and exact reports. The pilot state was retired through Store retention on
2026-11-27 with owner approval after the release recipe decision (receipt
`artifacts/receipts/control-disk-retirement-20261127/`); its diagnostic export remains. The full
bypass state and seed 8102's state remain.

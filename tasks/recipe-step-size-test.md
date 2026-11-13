# Four-arm step-size, schedule and latent-scale test

## Goal

Measure whether a decaying schedule, a bounded row latent scale with clamping, and a larger
sign step reduce noise-driven sign flips and improve learning on grouped HE4 updates, and
keep the best arm's state and export for later continuation.

## Authorized

Philip, 2026-11-12: build the recipe settings and run one four-arm test. GPU scope: four
fresh 7,800-update runs (seed 8101, frozen 6K TRAIN corpus, default grouping, plain shuffled
groups), their exports and 60-request Q4KM candidate-only screens, one screen of the shared
update-0 export, and at most one retry of a failed arm. Stop at 5 GPU hours in total.
Never use heldout24 or its aliases. Take the GPU only through the project lease, release it
between arms and wait if another task holds it. Delete non-selected arms' large states and
exports after their measurements are extracted and verified; keep the best arm's final
state and export in a new folder under the main checkout's `artifacts/training/`.

| Arm | Latent scale | Sign step | Other steps | Schedule |
| --- | --- | ---: | ---: | --- |
| A | raw, no clamp | 1e-3 | 1e-5 | constant |
| B | raw, no clamp | 1e-3 | 1e-5 | cosine to zero at 7,800 |
| C | row, multiple 3, clamp | 1e-3 | 1e-5 | 156-update warm-up, cosine to zero |
| D | as C | 4e-3 | 1e-5 | as C |

## Done when

All four arms and the update-0 export are screened (or the budget stop is reported), DEV
curves, screen comparisons and weight-change statistics are in a result page with JSON,
FINDINGS links it, the selected state/export is copied and verified loadable from the main
checkout, and the other raw outputs are retired.

## State

Owner: recipe-step-size agent, branch `t3/recipe-step-schedule-latent-scale`.
Recipe settings are committed (`d26a2853`). Raw outputs go to this worktree's
`artifacts/training/recipe-test/`: per arm a supervisor output (initial, update-7,790 and
final states, final export, journal) and a screen output. Purpose: measurements for this
result; disposal at closeout after extraction. Update-0 export (CPU, byte-identical under both latent scales) and its screen are done (1.5 GPU minutes). Arms run on code `ddac6d70…` without the data-path speedup, which had not reached main.

## Next

Run arms A–D in order through the lease, screening each export,
then extract measurements and close out.

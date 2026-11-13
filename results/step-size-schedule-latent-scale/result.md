# Decaying the step size improves fresh HE4 training; a bounded latent scale needs a larger sign step

A review of the 220,095-update HE4 state found that its 1-bit feed-forward signs had
flipped mostly on noise: 42–45% of signs changed, latent movement matched a random walk,
and some latents had drifted outside [-1, 1], where they stop learning. Four fresh
7,800-update runs (about 3.3M labelled tokens each, the same as that 8-hour run) compared
a cosine decay of the step size, a bounded row latent scale with clamping, and a 4× larger
sign step on that scale.

Decaying the step size to zero raised 60-request screen acceptance by 1.9 points over a
constant step. The bounded scale at the old sign step barely let signs move and was 1.0
point worse than the control. With a 4× sign step it matched the decay-only arm. All four
arms beat the old 8-hour state on the same screen by 4–7 points, after about an hour each.

## Outcome

Fresh start: the DFlash BF16 drafter binarized through the authenticated initialization.
All arms used the frozen 6K TRAIN corpus, seed 8101 and default grouping, so every arm
trained on the same 7,800 groups in the same order (222,825 windows, 3,280,230 labels). DEV numbers
are teacher-forced on the 120 fixed DEV windows at update 7,800. Screen numbers are native
Q4KM acceptance on the 60-request candidate-only training screen against saved controls.
Paired differences are source-group bootstraps (2,000 draws). The B, C and D rows against
arm A also hold under adjustment for three comparisons, except C, whose adjusted interval
is −2.09 to +0.07 points.

| Arm | Latent scale | Sign step | Schedule | DEV loss | DEV top-1 | Positions 1–3 top-1 | 3-token prefix survival | Screen acceptance | Accepted per round | Difference vs A (95%) |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| update 0 | — | — | — | 7.346 | 2.0% | 5 / 7 / 3% | 0% | 3.5% | 0.07 | −47.9 (−51.3, −44.8) |
| A | raw | 1e-3 | constant | 3.306 | 36.0% | 79 / 67 / 63% | 47.5% | 51.4% | 1.02 | — |
| B | raw | 1e-3 | cosine to 0 | 3.097 | 38.5% | 82 / 73 / 65% | 50.8% | 53.3% | 1.06 | +1.85 (+0.80, +2.91) |
| C | row ×3, clamp | 1e-3 | 2% warm-up, cosine | 3.260 | 37.0% | 79 / 69 / 63% | 45.8% | 50.4% | 1.00 | −0.97 (−1.91, −0.08) |
| D | row ×3, clamp | 4e-3 | 2% warm-up, cosine | 3.093 | 39.7% | 82 / 72 / 65% | 52.5% | 53.3% | 1.06 | +1.95 (+0.84, +3.09) |

Other paired screen differences, in points:

| Comparison | Difference (95% interval) |
| --- | --- |
| D vs B | +0.09 (−0.60, +0.74) |
| C vs B | −2.83 (−3.62, −2.09) |
| D vs C | +2.92 (+2.17, +3.72) |

The old 220,095-update state's saved reference on the same 60 requests scores 46.4%. Against
it A is +5.0 (+3.5, +6.6) points, B +6.9 (+5.3, +8.5), C +4.1 (+2.4, +5.7) and D +7.0
(+5.3, +8.7). Baseline scores 69.6% on this screen, so no arm approaches it. Screen tokens/s
were 126.1 (A), 124.8 (B), 120.7 (C) and 127.1 (D): 0.94–0.99× target-only, descriptive only.

DEV curves are in [result.json](result.json) (13 readouts per arm, every 650 updates and at
the end). With a constant step, A's DEV loss stays noisy late in training: 3.27, 3.41, then
3.31 over its last three readouts. The decaying arms improve steadily to the end; D leads
from about update 3,250 onward.

Weight change from update 0 to 7,800, over all 425M sign latents:

| Arm | Signs flipped | Smaller / larger half of start magnitude | Mean movement / mean start magnitude | Movement / random-walk prediction | Start magnitude in random-walk σ | At or beyond ±1 | Signs flipping per update at the end |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A | 24.2% | 36.1% / 12.2% | 1.05 | 0.97 | 0.73 | 0.002% | 215,400 (0.05%) |
| B | 16.8% | 29.4% / 4.1% | 0.65 | 0.98 | 1.20 | 0.0001% | 0.4 |
| C | 4.2% | 8.4% / 0.002% | 0.15 | 1.15 | 6.17 | 0.001% | 0.2 |
| D | 13.5% | 25.2% / 1.7% | 0.50 | 0.96 | 1.54 | 0.002% | 0.5 |

The random-walk prediction is the expected mean |movement| of independent ±step moves,
√(2/π) × √(Σ step²) over the run. "Start magnitude in random-walk σ" divides the mean
starting |latent| by √(Σ step²); it roughly counts how many standard deviations of
consistent evidence a typical sign needs before it flips. End-of-run flips compare the
final state with the update-7,790 state. The decaying arms' step is about 1e-10 by then, so
their near-zero rate reflects the schedule, not settled weights. Down projections flip
most in every arm, rising toward the last layer (A: 24% → 28%; D: 13% → 16%). The
per-matrix values are in the JSON.

Latent movement stays close to the random-walk prediction in every arm, including the
two that improved. That statistic does not separate useful from useless flips. What
changed outcomes is how much movement remained at the end (decay) and whether the step
matched the latent scale (C against D). In this short run few raw latents left [-1, 1]
(0.002% for A, against 4–10% in the old run), so the clamp's protection against frozen
latents is preventive here, not a measured gain.

Throughput: about 1,800 labels/s within updates and 860–890 end to end, before the
sequence-prefetch change. Peak memory was 12.55 GiB allocated and 13.88 GiB reserved
in every arm. The update-0 export is byte-identical under the raw and row latent scales.

## Scope and limitations

One seed, one corpus, one short budget (7,800 updates) and one GPU. DEV windows and
screen prompts are reused DEV cohorts, not independent confirmation; the screen sees
only answer openings (47 of 60 requests hit the 128-token cap). The DEV top-1 differences between
B, C and D are within the sampling noise of about 1,800 labels. B and D are
indistinguishable on acceptance. The old-state comparison holds labelled tokens equal
but not data: that state trained on the earlier 15,647-capture corpus with one window per
update, and its screen is a saved reference from separate execution. Speed ratios are
descriptive. Larger sign steps on the bounded scale, other warm-up lengths, longer
budgets and other seeds were not tried. Arms ran on the code before sequence prefetch.
That change is qualified as bit-identical, but these runs did not use it.

## Decision

Decided by this run, within its scope: carry arm D forward as the starting state and recipe
for the next HE4 training. It ties B on acceptance, has the best DEV loss, top-1 and
three-token prefix survival (B is one window ahead at position 2), keeps all latents in the trainable range, and its sign step is a working lever (C to D
+2.9 points). B is an equally supported recipe on this evidence. A constant step is no longer
the preferred default for fresh grouped training. The product defaults stay at today's
behaviour so existing recipes keep their meaning; intents select these settings
explicitly ([execution](../../docs/execution.md#optimization)). No deployment or leader
promotion follows.

## Evidence and reproduction

Executed source: runtime source identity `ddac6d70…` (branch commit `2e46f8e7`), frozen in
each run's output. Inputs: source `18cefa06…`, vocabulary `a68f55d9…`, initialization
`2f852a41…`, corpus `3fbc0135…`, DEV specification `qualification/training-diagnostics-spec.json`,
screen workload `feaeeae0…`, saved controls under `artifacts/benchmarks/references/`, and
the old state's screen reference `7b3e89ab…json`. Recipe identities: A `8eb60e75…`,
B, C and D in [result.json](result.json), which also holds every DEV readout, training loss
every 100 updates, per-matrix weight statistics, per-request screen rows for all five
screens and the old reference, and every paired comparison. Retained scripts in
`artifacts/training/recipe-test-arm-d-01/scripts/`:

- `intents.py`, `run_arm.sh`: generate the intents, then train and screen each arm.
- `export_zero.py`, `screen_spec.py`: build the update-0 export and the screen specifications.
- `analyze.py`: extract journals and weight statistics.
- `build_result.py`: regenerate result.json from the retained screens, journals and measurements.
- `verify_load.py`, `retire.py`: verify the kept state and retire the other outputs.

GPU use was 4.43 hours of wall time, including model load and screens.

## Artifact disposition

Retained in `artifacts/training/recipe-test-arm-d-01/` of the main checkout (5.4 GB):

- Arm D's final state `53a35c58…` (7,800 updates, 5.1 GB) and its GGUF export `6a07cd5f…`
  (0.54 GB), each in a store folder with its manifest and `latest.json`.
- Arm D's intent, sealed intent, journal and report, and its frozen executed source.
- All four arms' journals, intents and launcher logs, and all five screen outputs with
  specifications.
- Measurements, scripts and the 18 retirement receipts.

On the merged branch code, the kept state restores in branch mode from that folder. Its
re-export is byte-identical to the kept export (`measurements/verify-load-branch.json`).
Continue mode needs CUDA, like the original device, and was not exercised. The code on
main before this branch refuses to continue or branch the state, because its restore
required default learning rates.

Retired through `Store.retire_closed`, with receipts naming the kept state as recovery:

- Every arm's update-0 and update-7,790 states.
- The final states and exports of A, B and C.
- The worktree copies of D's final state and export.
- The update-0 exports, which `export_zero.py` rebuilds from the initialization in about
  30 s on CPU.

Exact resume from any arm's update-7,790 state, and continuation of A, B or C, is no longer
available. Models, corpus, initialization and saved controls were read only.

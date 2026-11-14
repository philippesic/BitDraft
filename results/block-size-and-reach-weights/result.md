# Training HE4 on short blocks raises acceptance 5–7 points at depths 2–4; random block sizes gain slightly less

HE4 was trained with sixteen draft slots and a fixed loss decay over all fifteen positions,
but servers build only depth+1 slots, and the slots attend to each other, so a block of
three or four slots gives the drafter a context it never trained on. Two recipes were
compared with the 16-slot control at about equal optimizer time from the same fresh start:
every update on 4-slot blocks (S4), and a random block of 3–16 slots per update weighted
toward short blocks, with loss weights proportional to how often each position is reached
(SR). The 16-slot control (S16) is the arm D run of the
[step-size test](../step-size-schedule-latent-scale/result.md).

Both short-block recipes beat S16 at every tested depth, by 4–7 acceptance points. S4 is
best at depths 2, 3 and 4. SR does not match S4 at shallow depth: it trails S4 by 1.6
points at depths 2 and 3 and by 1.0 at depth 4. Its advantage, better positions 5–10 in a
full sixteen-slot block, lies beyond the depths that exact serving uses today.

## Outcome

Native Q4KM acceptance on the 60-request DEV screen. Accepted per round counts accepted
drafts per drafting round. Paired differences are source-group bootstraps (2,000 draws) of
acceptance points; every interval below also excludes zero under the two-comparison
adjustment in [result.json](result.json).

| Depth | S16 | S4 | SR | Baseline | S4 − S16 (95%) | SR − S16 (95%) | SR − S4 (95%) |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2 | 53.3% / 1.06 | 60.0% / 1.19 | 58.4% / 1.16 | 69.6% / 1.38 | +6.6 (+5.6, +7.8) | +5.0 (+3.9, +6.2) | −1.6 (−2.5, −0.9) |
| 3 | 43.1% / 1.27 | 49.1% / 1.45 | 47.4% / 1.40 | 60.6% / 1.80 | +6.0 (+5.1, +7.0) | +4.4 (+3.3, +5.4) | −1.6 (−2.4, −0.9) |
| 4 | 35.6% / 1.40 | 40.7% / 1.60 | 39.7% / 1.56 | 51.2% / 2.01 | +5.1 (+4.1, +6.1) | +4.1 (+3.3, +4.9) | −1.0 (−1.8, −0.3) |

Cells are acceptance / accepted per round. Baseline leads every HE4 arm by 9.6 to 17.6
points (S4 against Baseline: −9.6 at depth 2, −11.6 at depth 3, −10.5 at depth 4). Screen
throughput was 0.91–1.06× target-only for the HE4 arms and is descriptive only.

Training at about equal optimizer time:

| Arm | Blocks | Position weights | Updates | Optimizer s | Windows | Labels | DEV cadence |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| S16 | 16 slots | decay exp(−i/7) | 7,800 | 1,814.5 | 222,825 | 3,280,230 | 650 |
| S4 | 4 slots | decay | 8,209 | 1,713.2 | 821,502 | 2,458,062 | 684 |
| SR | 3–16 slots, P(k) ∝ 1/(k−1) | reach, refreshed every 351 updates | 8,416 | 1,749.0 | 600,213 | 2,693,161 | 351 |

Smaller blocks let each update hold proportionally more windows: S4 averaged 100 windows
per update against S16's 29, at the same peak allocation (13.4–13.5 GB). Per-update time was
nearly constant (0.21–0.23 s), so the arms ran similar update counts. SR's updates drew
3 slots 22% of the time and 16 slots 2.8%.

DEV teacher-forced top-1 on the 120 fixed windows, final states. Short blocks are what a
depth-2, -3 or -4 server builds (3, 4, 5 slots); with 120 windows one point is one window
and the sampling noise is about ±4 points.

| Block | Arm | Position 1 | 2 | 3 | 4 | Prefix survival through the last position |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 slots | S16 / S4 / SR | 82 / 85 / 85 | 68 / 78 / 73 | | | 63 / 74 / 68 |
| 4 slots | S16 / S4 / SR | 83 / 85 / 83 | 72 / 79 / 73 | 64 / 68 / 68 | | 52 / 57 / 53 |
| 5 slots | S16 / S4 / SR | 82 / 85 / 84 | 71 / 79 / 73 | 65 / 68 / 68 | 53 / 53 / 53 | 37 / 41 / 41 |

In one sixteen-slot block, positions 1–15 (percent):

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S16 top-1 | 82 | 72 | 65 | 53 | 45 | 46 | 37 | 32 | 31 | 30 | 26 | 20 | 17 | 18 | 14 |
| S4 top-1 | 84 | 74 | 65 | 53 | 38 | 35 | 20 | 23 | 22 | 14 | 9 | 8 | 9 | 7 | 8 |
| SR top-1 | 84 | 73 | 68 | 54 | 45 | 44 | 36 | 30 | 26 | 29 | 19 | 12 | 14 | 16 | 10 |
| S16 survival | 82 | 64 | 53 | 38 | 27 | 22 | 18 | 12 | 10 | 5 | 5 | 4 | 4 | 4 | 2 |
| S4 survival | 84 | 70 | 54 | 35 | 19 | 16 | 9 | 7 | 6 | 2 | 0 | 0 | 0 | 0 | 0 |
| SR survival | 84 | 68 | 53 | 41 | 30 | 25 | 19 | 15 | 12 | 7 | 4 | 1 | 0 | 0 | 0 |

The window counts per position (120 to 111) and all hits and survivors are in
`dev_blocks` of [result.json](result.json), along with every training-time readout.
S4 never trained on positions beyond 3, yet holds up through position 4 even in sixteen
slots; past position 5 it falls well behind. SR keeps S16's deep-position accuracy while
matching S4 on positions 1 and 3. SR's DEV loss over all positions (3.049) is the lowest of
the three (S16 3.093, S4 3.411), but loss and top-1 are not acceptance.

SR's reach weights started almost entirely on position 1 (update 0: 1, 0.06, 0.01, …) and
reached 1, 0.84, 0.68, 0.54, 0.42, 0.31, 0.27, 0.21, 0.16, 0.14, 0.09, 0.05, 0.02, 0.01, 0.01
at the last refresh (update 8,073). Every refresh is journaled.

## Scope and limitations

- One seed, one corpus, one fresh start, one GPU, about 30 minutes of optimizer time per
  arm. The 60-request screen sees only answer openings and is a reused DEV cohort.
- Equal time is approximate: update budgets were set from 400-update pilots, and the full
  runs ran slightly faster, so S4 had 94.4% and SR 96.4% of S16's optimizer time. Both
  shortfalls favour S16; S4 also had 2% less time than SR and still led.
- S16 at depth 2 is arm D's screen from the earlier test, on the same runtime, workload
  and saved controls; acceptance counters are deterministic under the exact-verification
  setting. Depths 3 and 4 ran all five arms, target and Baseline included, in one session.
- Depths above 4 were not screened. SR's DEV advantage at positions 5–10 suggests it may
  overtake S4 at deeper drafts, but this is untested natively, and exact verification
  currently makes depths above 3 costly ([verification cost](../cuda-decode-consistency-cost/result.md)).
- Only one random-size distribution and one reach estimate were tried. The reach weights
  come from the sixteen-slot diagnostic, use smoothed measured prefix survival, and start
  near position 1 alone, which may waste SR's first few hundred updates. Random sizes and
  reach weights were not separated, so which of the two cost SR its 1–1.6 points is unknown.
- S4 and SR saw 3.7× and 2.7× as many windows as S16 but fewer labels per window; no
  window was repeated in any arm.

## Decision

Decided by this run, within its scope: carry S4's recipe (every update on 4-slot blocks,
fixed decay) and its state forward for HE4 at depths 2–4. It is best at each tested depth
and costs nothing in memory or time. SR is not preferred: it does not match S4 at shallow
depth. Random block sizes and reach weights stay available as recipe settings for depth
calibration work, with today's fixed 16 slots as the product default so existing recipes
keep their meaning ([execution](../../docs/execution.md#block-sizes-and-position-weights)).
No deployment or leader promotion follows. Proposed, not decided: measure S4 and SR at
deeper depths from the kept exports (the depth calibration on main, `benchmarks
calibrate-depth`, measures every depth in a few GPU minutes), and separate random sizes
from reach weights (for example random 3–5 slots with decay) before choosing a
depth-general recipe.

## Evidence and reproduction

Executed source: runtime source identity `179771139f…` (branch commit `e10976f4`), frozen in
each run's output; S16 ran `ddac6d70…`, and a 100-update pilot on the new code reproduced
its groups, update-0 diagnostic, loss (5.811744689941406) and gradient norm at update 100
exactly. Inputs, read only: source `18cefa06…`, vocabulary `a68f55d9…`, initialization
`2f852a41…`, corpus `3fbc0135…` (2,283,089 windows), DEV specification
`qualification/training-diagnostics-spec.json`, screen workload `feaeeae0…`, saved Q4KM
controls, runtime v5. Recipe identities: S16 `7b736d6a…`, S4 `5012dbf2…`, SR `93caf21f…`.
Other arms' recipes repeat arm D's except for blocks, update budget and schedule length
(2% warm-up then cosine to zero at the budget).

[result.json](result.json) holds every arm's budgets, block-size counts, DEV readouts with
per-position top-1, survival and reach weights, training loss every 100 updates, the
final-state DEV diagnostics at 3, 4, 5 and 16 slots, every screen summary, per-request
accepted/proposed/tokens and all paired comparisons. Scripts, kept with the evidence in
`artifacts/training/block-test-s4-01/run/scripts/` of the main checkout:

- `intents.py`, `run.sh`: generate the intents and train with lease-retrying launches.
- `pilot_check.py`: arm D reproduction and update-time calibration.
- `dev_blocks.py`: DEV diagnostics of final states at several block sizes.
- `screen_spec.py`, `screens.sh`, `post.sh`: screen specifications and execution.
- `build_result.py`: regenerates result.json from the retained journals and screens.
- `verify_load.py`, `retire.py`: verify the kept state and retire the others.

GPU use was 1.69 hours of wall time, including model loads, diagnostics, screens and three
startups refused by the lease before training.

## Artifact disposition

Retained in `artifacts/training/block-test-s4-01/` of the main checkout (5.9 GB):

- S4's final state `6977443f…` (8,209 updates, 5.1 GB) and its export `10ade33a…`
  (0.54 GB), each in a store folder with manifest and `latest.json`; S4's intent, journal,
  report and frozen executed source.
- SR's export `dc4e30f6…` (0.54 GB) under `other-arms/sr/exports/`, so deeper screens can
  run without retraining.
- `run/`: every arm's and pilot's journal, intent and log, all screen outputs and
  specifications, DEV block measurements, the retirement receipts and scripts.

On this branch's code the kept state restores in branch mode and re-exports byte-identically
to the kept export (`run/verify-load-branch.json`). Continue mode needs CUDA and was not run.

Retired through `Store.retire_closed`, with receipts naming the kept S4 state as recovery:
SR's final and update-0 states and its export copy in the worktree, S4's update-0 state
and worktree copies, and all three pilots' states and exports. Exact resume of SR, and of
any pilot, is no longer available; SR's export remains. Arm D's kept state was read only.

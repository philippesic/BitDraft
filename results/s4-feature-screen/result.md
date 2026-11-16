# In short S4 runs no quantizer or teacher feature clearly helps; two-value-plus weights lead on DEV, the teacher is rejected, and S4's learning curve stopped for lack of new data

A 2026-11-15 screen ranked candidate features for the one-bit drafter S4 with matched
30-optimizer-minute runs on the S4 recipe. Every run used the same data and order and
changed one thing. It also tested whether the
[learning curve](../s4-data-curve/result.md) flattened because of its data or because its
step size reached zero.

- No feature beat the control by a margin that decides anything.
- Two weight formats that give each feed-forward weight more than two values led on DEV prefix survival:
  - three values (−1, 0, +1): +4.2 points;
  - a second one-bit plane (four values): +3.3 points.

  Neither can be served yet, so neither has an acceptance measurement.
- Full-precision feed-forward weights gained only about 2 points, so weight precision is not
  what limits S4 at this length.
- Matching a full-precision teacher gave no acceptance gain once the teacher fit at the
  normal memory setting on native Linux: +0.2 points (−0.4, +1.0). The owner rejected it on
  2026-11-16.
- One more optimizer-hour at a fresh step size on the curve's own data gained nothing
  (+0.4 points, −0.7 to +1.4), so the flattening came from the data.

## Outcome

DEV figures use the 120 fixed DEV windows in one 16-slot block, at the final update:
- top-1 is pooled over positions 1–3;
- survival is the share of windows whose positions 1–3 are all correct.

The screen is native acceptance on the 60-request Q4KM DEV screen at depth 3. All screen
differences are paired source-group bootstraps (2,000 draws, stratified by category).

| Arm | What changed | DEV top-1 | DEV survival | Screen acceptance | Δ top-1 | Δ survival | Δ screen (95%) | Beyond noise (top-1 / survival / screen) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| a1 control | S4 recipe with `baseline-k` frozen tables | 73.6% | 53.3% | 49.70% | — | — | — | — |
| a2 | control, data order seed 8102 | 74.4% | 54.2% | 49.64% | +0.8 | +0.8 | −0.1 (−0.8, +0.8) | the noise |
| b1 | feed-forward weights −1 / 0 / +1 | 75.6% | 57.5% | not servable | +1.9 | +4.2 | — | borderline / yes / — |
| d1 | second one-bit plane (residual signs) | 75.6% | 56.7% | not servable | +1.9 | +3.3 | — | borderline / yes / — |
| b4 | feed-forward weights full precision | 75.8% | 55.0% | not servable | +2.2 | +1.7 | — | borderline / borderline / — |
| b2 | feed-forward weights 2-bit | 75.0% | 53.3% | not servable | +1.4 | 0.0 | — | borderline / no / — |
| d2 | one scale per 128 inputs per row | 74.7% | 54.2% | not servable | +1.1 | +0.8 | — | no / no / — |
| L2 (native Linux) | hidden-value teacher, default memory setting | 76.1% | 55.0% | 49.94% (control 49.70% in the same panel) | +2.5 | +1.7 | +0.2 (−0.4, +1.0) | borderline / borderline / no |
| c0 (WSL) | control at 2 GiB activations | 73.9% | 54.2% | 48.36% | +0.3 | +0.8 | −1.3 (−2.2, −0.4) | no / no / yes |
| c1 (WSL) | c0 plus output-score teacher | 75.6% | 53.3% | 49.32% | +1.7 | −0.8 | +1.0 (+0.0, +1.9) vs c0 | borderline / no / borderline |
| c2 (WSL) | c0 plus hidden-value teacher | 75.6% | 55.0% | 49.52% | +1.7 | +0.8 | +1.2 (+0.3, +2.0) vs c0 | borderline / no / yes |
| a3 restart | the curve's final state, one more optimizer-hour, fresh cosine at 1/3 step | 78.3% (start 79.2%) | 58.3% (start 60.8%) | 55.58% (start 55.23%) | −0.8 | −2.5 | +0.4 (−0.7, +1.4) vs start | no / borderline / no |

**Noise rule.** The reshuffled control differed by +0.8 top-1, +0.8 survival (one window)
and −0.06 screen points.
- "Yes" means the difference exceeds twice that, with floors of 2.5 DEV points and 1.0 screen point.
- On the screen, the paired interval must also exclude zero.
- "Borderline" means the difference exceeds the observed noise and half the floor.

All fresh arms share one initialization, so this is data-order noise only.

**Weight precision at initialization.** Every weight format, full precision included, starts at
4–7% DEV top-1 at update 0. The initial damage comes from rounding the feed-forward inputs to 4 and 8 bits,
not from rounding the weights.

**Teacher.**
- The teacher is Baseline with the source drafter's BF16 feed-forward tables and no input rounding.
- It shares the student's frozen 4-bit attention, fusion and vocabulary tables.
- On DEV it scores 75.8% top-1 and 59.2% survival, against the control's 73.6% and 53.3%, so there is
  little to transfer on positions 1–3.
- On WSL it fit only at 2 GiB activations. That smaller setting alone cost the control 1.3 screen
  points, so the WSL gains over c0 left both teacher arms level with the normal control: −0.4 (−1.3, +0.7)
  and −0.2 (−1.1, +0.8).
- On native Linux it fit at the default setting, and with the confound removed it shows no
  acceptance gain.

**Restart.** The curve's final state `769e184e…` was branched into `baseline-k` and trained for 17,700
updates (3,600 optimizer-seconds). It used the same 6K+6K mixture, 2% warm-up, and cosine to zero
at one third of S4's step sizes. DEV at the five readouts (top-1 / survival):

| Readout | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-1 | 79.2% | 78.3% | 78.1% | 77.2% | 78.3% |
| Survival | 60.8% | 58.3% | 58.3% | 56.7% | 58.3% |

Training loss stayed at 0.92–0.95 for the whole hour.

**Cost estimates** (not measured):

| Feature | Feed-forward bytes (one bit: 53 MB) | Export (now 542 MB) | Serving work |
| --- | ---: | ---: | --- |
| Three values (−1, 0, +1) | 85 MB at 1.6 bits/weight, 106 MB at 2 bits | +32 to +53 MB | two bit-planes or a 2-bit kernel, about twice the one-bit work |
| Second one-bit plane | 106 MB | +53 MB | two passes of the existing one-bit kernel plus a second row scale |
| One scale per 128 inputs | +6.6 MB | +7 MB | a partial sum per block |
| Teacher | 0 | 0 | none; training needs 0.8 GiB more GPU memory |

## Scope and limitations

- One 30-minute run and one seed per arm, matched on 8,209 updates rather than equal time.
- Short runs can rank features differently from long ones. Finer formats used S4's sign step
  size (0.004), which was not retuned.
- DEV has 120 windows (one window is 0.8 survival points). The screen is a reused 60-request DEV cohort.
- Unservable arms have no acceptance measurement.
- The teacher was tried at one weight (1.0), with hidden matching on Linux and both variants on WSL.
- The restart is one run at one restarted step size, on the third pass over those prompts.
- WSL screens used saved runtime-v7 controls. The Linux screen ran the target live, because the
  saved references are WSL hardware records.
- The control screens 49.70% on both hosts. The Linux parity run reproduced the control's
  update-100 loss, gradient norm and data groups exactly.
- No speed figure is used.

**Host failures.** With the RAM overclock (EXPO) on, six runs failed in ways the model code cannot explain:
- three "F16 cache storage overflow" errors, which the worst-case bound rules out and which did not repeat
  on rerun: 4-bit weights at update 3,255, second plane at 2,990, output-score teacher at 3,351;
- two data-checksum mismatches: one as the WSL instance went down at about 06:50, one at the same second
  as a kernel-logged general protection fault;
- one process killed by another general protection fault.

Windows also bugchecked at 12:21 (0x7F, Vid.sys). After EXPO was turned off, about 3.75 GPU
hours ran without a fault on WSL and native Linux, including both teacher arms and the second plane. This
implicates the overclock but does not prove it. The teacher arms also ran out of memory at the
default setting on WSL.

Not run: 4-bit feed-forward weights, which crashed and were then dropped by the owner as off-goal;
4-bit down tables in layers 2 and 5, dropped by the owner; and best-of-weights with the
teacher (budget). GPU time was about 11.8 hours.

## Decision

Decided by the owner on 2026-11-16:
- **Teacher training: rejected.** No acceptance gain at the normal memory setting, and the
  teacher is only modestly better than the student.
- **4-bit feed-forward weights and 4-bit down tables: dropped** as contrary to the low-precision goal.

Proposed, not decided:
- **Three-valued weights and a second one-bit plane** are the only features worth a longer
  look. Each first needs a serving kernel and an acceptance measurement.
- **2-bit weights and one scale per 128 inputs** showed nothing worth pursuing.
- **Weight precision** is not the limit at this length; input rounding at initialization is the
  larger loss.
- **Restart:** S4's curve flattened because of its data. More passes or a restarted schedule on
  the same prompts will not help; new prompts might.

The prototype code stays unmerged on branch `t3/overnight-feature-screen` as executed source.

## Evidence and reproduction

- **Code.** Prototype formats and teacher losses are on branch `t3/overnight-feature-screen`:
  - `06297fb6`: source `5f7d62d7…`;
  - `f6d86e3f`: per-update finiteness checks, source `d5465afd…`;
  - `e0e03ffa`: native-Linux backing capacity, source `d3320ffe…`.
- **Block A** ran unchanged code from `11982ee2` (source `c51fbc6d…`).
- **Frozen executed sources** are under `artifacts/training/overnight-feature-screen-01/executed-sources/`.
- **Inputs:**
  - Baseline `83586a63…`
  - source `18cefa06…`
  - vocabulary `a68f55d9…`
  - initialization `2f852a41…`
  - corpus `3fbc0135…` (the restart used mixture `34a955a8…`)
  - DEV specification `54afbd49…`
  - screen specification `q4km-screen-leader.json` (runtime-v7); on Linux the same specification without references
- **[result.json](result.json)** holds every arm's final DEV figures, states, exports, screen
  totals and paired intervals, the Linux parity record, and per-group accepted and proposed
  counts for every screen.
- **Full report** with failure details and per-position tables:
  `artifacts/training/overnight-feature-screen-01/report.md`. The Linux preflight is
  `linux-teacher-preflight.md` in the same folder.

## Artifact disposition

Retained in `artifacts/training/overnight-feature-screen-01/` (62 GB):
- every arm's intent, journal, run report and error log;
- every completed arm's final state, and the export of each servable arm;
- all screens and their specifications;
- `summary.json`, `parity-check.json` and `parity-check-linux.json`;
- the report, the preflight, the frozen executed sources and the scripts.

The final states of the three-valued and second-plane arms are kept as starting points and fixtures for serving
kernels. The rest are kept because the disk has room (255 GB free) and they are the only trained
examples of each format.

Removed earlier (logged in `logs/cleanup.log`), all created by this screen:
- update-0 states, regenerable from the initialization in about a minute;
- the two parity runs' states and exports;
- the zero-update restart state;
- the update-0 states of failed arms;
- one incomplete state written during a kill.

Failed arms have no resumable state; none is needed. Nothing created by others was changed.

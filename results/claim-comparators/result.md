# The size claim survives stock low-bit files, the target's built-in head beats Baseline's acceptance with no file, and two input channels explain most of the one-bit start

The research sweep of 2026-11-20 named three measurements that decide what the one-bit
leader `ef068eba…` may claim. Test-list items 13, 14 and 15 ran them on the owner's go of
that day. Acceptance is the share of drafted tokens the target keeps.

What they show:
- **Stock low-bit files do not match the leader.** The publisher's 2-bit file of Baseline
  (244 MB) accepts 39.3% on the panel against Baseline's 53.4% and the leader's 50.8%, and
  runs 15.5% slower. The 3-bit file (314 MB) accepts 52.0% and runs 3.9% slower.
- **The target's built-in draft head is a comparator every claim must name.** It needs no
  drafter file, accepts 57.9% (4.6 points more than Baseline) and runs at 0.964 of
  Baseline's speed.
- **With the same shortlist, the leader's small file is 1.8% faster than the 4-bit
  control.** In the same file form (full table plus shortlist) the control is 1.1% faster.
  The control accepts 3.8 points more. It trained for 21,000 updates and the leader for
  335,812, so the arms are not matched on training length.
- **Sign weights collapse the untrained start, and two input channels carry most of it.**
  Sending those two channels through Baseline's own columns lifts the one-bit start from
  10.3% to 30.5% simulated acceptance. 8-bit inputs alone do not.

## Outcome

### Stock low-bit files and the built-in head (item 13)

Five arms in one run per tier on runtime-v10, depth 3, greedy, no shortlist: target,
Baseline, the publisher's Q3_K_M and Q2_K files, and the head. Ratios are paired
source-group bootstraps against Baseline, 95%.

| Arm | Drafter file | Panel acceptance | Speed against Baseline (95%) | Drafting ms per round | Screen acceptance |
| --- | ---: | ---: | --- | ---: | ---: |
| Baseline (Q4_K_M) | 381 MB | 53.35% | 1 | 1.59 | 60.64% |
| Baseline Q3_K_M | 314 MB | 52.00% | 0.961 (0.957, 0.966) | 1.84 | 58.38% |
| Baseline Q2_K | 244 MB | 39.31% | 0.844 (0.837, 0.852) | 1.76 | 42.61% |
| Target's built-in head, 3 steps | none | 57.92% | 0.964 (0.956, 0.974) | 2.53 | 64.54% |

Acceptance differences from Baseline on the panel: Q3_K_M −1.35 points (−1.76, −0.94),
Q2_K −14.03 (−15.10, −13.20), head +4.57 (+3.70, +5.55). Every output was exact.

Against the leader, from separate runs on the same runtime, host and day (Baseline's
panel speed repeated at 276.6 and 276.8 tokens per second):

| | File | Panel acceptance | Tokens per second |
| --- | ---: | ---: | ---: |
| Leader, small file | 284 MB (185 MB without rows) | 50.80% | 297.5 |
| Baseline Q3_K_M | 314 MB | 52.00% | 266.0 |
| Baseline Q2_K | 244 MB | 39.31% | 233.8 |
| Built-in head | none | 57.92% | 266.9 |
| Baseline | 381 MB | 53.35% | 276.8 |

So among stock files, nothing at or below the leader's size comes near its acceptance, and
the 3-bit file is larger and about 11% slower than the small file. The head accepts 7.1
points more than the leader with no file at all, and is about 10% slower than the small
file. The leader's small file uses the shortlist; the head and the stock files do not.

The low-bit files are slower because this runtime's Q2_K and Q3_K kernels are slower than
Q4_K, as test-list item 5 found for the vocabulary table. Their speed is a property of
these kernels, not of the formats.

### The 4-bit control with the shipping shortlist (item 14)

Six arms in one run per tier on runtime-v10, depth 3, greedy. Baseline and the control use
the leader's 68,728-token shortlist as a sidecar. The control is the 21,000-update export
`3d006c22…`.

| Arm | Panel acceptance | Tokens per second | Drafting ms per round |
| --- | ---: | ---: | ---: |
| Baseline with shortlist | 53.33% | 289.4 | 1.18 |
| 4-bit control | 54.58% | 280.2 | 1.58 |
| 4-bit control with shortlist | 54.57% | 292.2 | 1.19 |
| Leader, full table with shortlist | 50.80% | 289.0 | 0.97 |
| Leader, small file | 50.80% | 297.5 | 0.92 |

| Comparison (control with shortlist against) | Panel speed (95%) | Screen speed (95%) | Panel acceptance, points |
| --- | --- | --- | --- |
| Leader, small file | 0.982 (0.974, 0.991) | 0.983 (0.973, 0.993) | +3.77 (2.90, 4.64) |
| Leader, full table with shortlist | 1.011 (1.003, 1.020) | 1.004 (0.994, 1.014) | +3.77 |
| Baseline with shortlist | 1.010 (1.006, 1.014) | 1.018 (1.007, 1.031) | +1.24 (0.79, 1.77) |
| The control without shortlist | 1.043 (1.040, 1.045) | 1.035 (1.029, 1.041) | −0.02 |

The leader's lead over the control comes entirely from the small file's unexplained 2.9%
over its own full-table form (1.029; 1.027, 1.031), which
[the shortlist-only result](../shortlist-only-table/result.md) already recorded. The
control's file layout borrows the target's table and has no small-file form, so whether a
4-bit drafter could get the same 2.9% is untested. Against Baseline with the shortlist the
leader's full-table form is 0.999 (0.989, 1.008) and its small file 1.028 (1.019, 1.038).

### Start quality with no training (item 15)

Each row builds the drafter from the public BF16 tables with no update, calibrates it on
the leader's own sample (16 TRAIN sequences, 16 windows each) and replays the round
simulation on 32 DEV requests (about 12,000 to 26,000 proposed drafts per row). "Exact"
keeps Baseline's feed-forward weights; "sign" is the one-bit start with fitted row scale
and midpoint. Inputs to gate and up are unrounded, 4-bit, or 8-bit; down inputs are 8-bit
except in unrounded rows. "Bypass" sends input channels 0 and 850 of gate and up through
their two exact source columns instead of the clipped, rounded path.

| Weights | Gate/up inputs | Simulated acceptance | First choice, slots 1–3 | In top 16, slots 1–3 |
| --- | --- | ---: | --- | --- |
| Baseline itself | as served | 60.85% | 79.8 / 65.2 / 54.1 | 97.7 / 90.1 / 82.6 |
| Exact | unrounded | 60.78% | 79.7 / 64.9 / 54.1 | 97.7 / 89.9 / 82.8 |
| Exact | 4-bit | 41.33% | 68.8 / 43.7 / 30.3 | 94.8 / 78.7 / 68.6 |
| Exact | 4-bit, bypass | 60.10% | 79.0 / 64.8 / 53.9 | 97.6 / 89.8 / 82.2 |
| Exact | 8-bit | 60.14% | 78.7 / 64.7 / 53.3 | 97.4 / 89.5 / 82.6 |
| Sign | unrounded | 12.75% | 28.3 / 24.9 / 19.4 | 62.5 / 57.0 / 51.0 |
| Sign | 4-bit (the leader's start) | 10.31% | 24.6 / 19.1 / 12.9 | 59.7 / 52.2 / 46.9 |
| Sign | 4-bit, bypass | 30.49% | 50.7 / 43.1 / 36.0 | 85.4 / 75.3 / 67.8 |
| Sign | 8-bit | 12.98% | 28.9 / 25.2 / 19.8 | 63.1 / 57.5 / 51.3 |
| Sign | 8-bit, bound at the largest input | 12.24% | 27.9 / 24.1 / 18.7 | 61.4 / 55.9 / 50.4 |
| Sign | 8-bit, bypass | 31.17% | 51.3 / 43.4 / 36.9 | 85.6 / 75.4 / 68.4 |
| Sign | unrounded, bypass | 31.49% | 51.9 / 43.2 / 36.8 | 85.5 / 75.7 / 68.5 |

Paired differences over the 32 source groups, in points (95%):

| Change | Difference |
| --- | --- |
| Exact weights: unrounded to 4-bit inputs | −19.45 (−23.38, −16.50) |
| Exact weights: unrounded to 4-bit inputs with bypass | −0.68 (−1.19, +0.03) |
| Exact weights: unrounded to 8-bit inputs | −0.63 (−1.03, −0.24) |
| Unrounded inputs: exact to sign weights | −48.03 (−57.48, −41.24) |
| Sign weights, 4-bit inputs: add the bypass | +20.17 (+15.60, +26.43) |
| Sign weights: 4-bit to 8-bit inputs | +2.67 (+1.35, +4.42) |
| Sign weights with bypass: 4-bit to 8-bit inputs | +0.69 (+0.38, +1.16) |

Reading:
- The simulation is sound: Baseline's tables in our graph give 60.78% against Baseline's
  own 60.85% (−0.07; −0.49, +0.40).
- With exact weights, 4-bit input rounding costs 19 points, and all of it is the two
  channels. Carrying them exactly, or using 8-bit inputs, leaves under 1 point.
- Sign weights cause the collapse, with or without input rounding (12.75% unrounded).
- The two channels are also most of the sign-weight damage. With them carried exactly the
  sign start triples to 30–31%, and input precision then barely matters.
- 8-bit gate/up inputs without the bypass gain only 2.7 points. Research gap (b) is
  therefore not the fix on its own.

The channels are close to constants in the early layers, which is why a row-scaled sign
cannot carry them. Channel 0 is +35.2 ± 0.2 and channel 850 is −35.3 ± 0.2 at layer 0,
against an rms of 0.17 for the other 2,558 channels. By layer 5 they are +16.1 ± 5.9 and
−15.4 ± 6.8, against 0.88.

## Scope and limitations

- **Start quality is not trained quality.** The start table has no training. The leader
  recovers from 10% to 50.8% served acceptance over 335,812 updates, and nothing here shows
  how much of the 20-point start gain survives training.
- **Simulated, not served.** Start-table rows are round simulations on 32 DEV captures
  (`broad-precision-dev-01`), not the screen or the panel. The 60-request screen captures
  the earlier diagnostics used were retired. The bypass has no serving kernel.
- **The control is not matched to the leader on training length** (21,000 against 335,812
  updates) and starts from Baseline. Item 14 compares speed at release settings only.
- **Greedy screen and panel only, one run per tier.** Arm order rotates across request
  groups. Thinking-on, sampled and agentic scenarios were not run for the new arms.
- **Leader against low-bit files and head is across runs**, same runtime, host and day.
  The acceptance figures are exact-mode counts and do not depend on the run.
- **Stock conversions only.** The publisher's card states no calibration. A calibrated
  low-bit mix of Baseline near 184 MB was not built.
- **The head was served by naming the target file as the drafter**, so the server loaded
  the target twice (7,465 MiB after start). A manual check without a drafter file gave the
  same tokens, counts and speed on two prompts at 4,530 MiB, against 4,098 MiB for the
  target alone (`logs/head-check.txt`). The shortlist does not apply to the head in this
  runtime; the head scores the full vocabulary at each of its 3 steps.
- **Reused DEV data**, one RTX 5080, Qwen3.5-4B Q4KM, batch 1, native Linux, runtime-v10,
  depth 3. The protected confirmation sets and heldout24 were not touched.

## Decision

- **Retain** the `"draft-mtp"` speculation value of the comparison specification
  ([benchmarks](../../docs/benchmarks.md)).
- **Claims, proposed for the owner:**
  - "Smaller than any stock file of Baseline at comparable acceptance" is supported.
  - Every size or speed claim names the built-in head: no file, 4.6 points more acceptance
    than Baseline, 3.6% slower than Baseline.
  - "As fast as a same-data 4-bit drafter at release settings" is supported for the small
    file (1.8% faster) and not for the matched file form (1.1% slower); both carry the
    3.8-point acceptance gap and the unmatched training length.
- **Proposed, not decided:** one 21,000-update one-bit run with the two-channel bypass
  against `w1-21000` (46.75% on the panel), read first in the round simulation. It needs a
  training-graph change, and serving it needs a small kernel side-term. Not authorized.
- **Unresolved:** a calibrated low-bit Baseline near 184 MB; the head with a shortlist;
  the small file's 2.9%.
- **No promotion.** `ef068eba…` stays the provisional acceptance leader.

## Evidence and reproduction

All paths are under `artifacts/claim-measurements-01/`. Compact measurements, paired
statistics, per-group totals and per-request simulation counts are in
[result.json](result.json), written by `scripts/assemble.py`.

**Served runs** (`runs/NAME/`, specifications `specs/NAME.json` from
`scripts/make_specs.py`, summaries `logs/summary-NAME.txt`):

| Run | Requests | Executed source | Result |
| --- | ---: | --- | --- |
| `control-screen`, `control-panel` | 60, 120 | frozen `29115117…` (as the shortlist-only result) | all arms complete and correct |
| `lowbit-screen`, `lowbit-panel` | 60, 120 | frozen `dbcfd691…` in `runs/executed-sources/`, commit `645a745a` | all arms complete and correct |

Runtime-v10 is `artifacts/shortlist-only-table-01/builds/runtime-v10` (`result.json`
`b7e096f2…`). The low-bit files are `models/dflash-4b-q2k.gguf` (`d4da5ebc…`) and
`models/dflash-4b-q3km.gguf` (`973e4eef…`), downloaded from the publisher's repository
`Anbeeld/Qwen3.5-4B-DFlash-GGUF`; both hashes equal the repository's listed ones, and its
Q4_K_M file is Baseline (`83586a63…`).

**Start table:** `runs/start-table/` (one JSON per row, `calibration-sample.json`), run by
`scripts/start.sh` with `scripts/start_table.py` (`90d87306…`) on the tree at commit
`645a745a`. It uses the intent of `artifacts/one-bit-attribution-01/arms/w1-21000` for
source, vocabulary table, frozen tables and calibration sample, and the round settings of
`artifacts/feature-validation-01/runs/round-diagnostics/calibrated-dev-spec.json`. The
script is an untracked experiment, not a maintained tool. Its first test stopped before
measuring because it named the retired screen captures; the recorded rows use the DEV
captures above.

**Commands:** `scripts/bench.sh`, `scripts/bench_tree.sh`, `scripts/start.sh`,
`scripts/head_check.sh`; status in `logs/status.log`.

## Artifact disposition

**Retained** (604 MB in all), owner Philip: the two publisher files (557 MB), the four
served runs, the start table, the specifications, scripts and logs. They are kept until
the owner decides the release claim and the bypass run, then the runs follow the
retirement command. The publisher files can be downloaded again from the hashes above.

**Dependencies:** the leader's serving files and runtime-v10 under
`artifacts/shortlist-only-table-01/`, the control export `3d006c22…`, Baseline, the
target, the public BF16 drafter, `train-final-02-fused` and `broad-precision-dev-01`. None
was changed.

**Retired:** a two-row test of the start table (`runs/start-debug`), whose rows the full
table repeats.

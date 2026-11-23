# More distinct data helps one-bit; trained tables, trained vocabulary rows and soft targets do not

Release-retrain Stage 2, 2026-11-22 to 2026-11-23, local RTX 5080, Q4KM target, runtime-v10,
depth 3. At about one pass, four times more distinct new-mix data raised one-bit DEV-screen
acceptance by 1.6 and 2.4 points; repeating data cost up to 21 points. No added feature showed a
resolved served advantage in the configurations tested: trained attention tables (with and
without rounding during training), trained shortlist vocabulary rows, and top-32 soft targets.
Batched answer writing is as close to the target's greedy choices as one-at-a-time writing.
One-bit attention tables could save at most about 2.6% of round time.

## Outcome

Acceptance on the 96-request retrain DEV screen (greedy unless noted). Paired request bootstrap
(2,000 draws) of feature arm minus its matched control, same data, seed, grouping and start.

| Test | Updates | Seed | Feature | Control | Difference (95%) |
| --- | ---: | --- | ---: | ---: | --- |
| Trained attention (F32, quantizer export) | 2,370 | 8101 | 48.22% | 48.30% | −0.08 (−0.73, +0.61) |
| Trained attention (F32, quantizer export) | 2,370 | 8102 | 48.93% | 49.40% | −0.47 (−1.14, +0.21) |
| Trained vocabulary rows | 2,370 | 8101 | 47.12% | 47.42% | −0.30 (−0.85, +0.25) |
| Trained vocabulary rows | 2,370 | 8102 | 47.29% | 47.42% | −0.13 (−0.79, +0.49) |
| Soft targets, weight 0.5 | 2,370 | 8101 | 48.08% | 48.30% | −0.22 (−0.90, +0.50) |
| Soft targets, weight 0.5 | 2,370 | 8102 | 49.58% | 49.40% | +0.19 (−0.58, +0.94) |
| Soft targets, sampled check | 2,370 | 8101 | 46.23% | 45.99% | +0.24 (−0.18, +0.69) |
| Soft targets, sampled check | 2,370 | 8102 | 47.01% | 46.92% | +0.10 (−0.37, +0.61) |
| Rounded attention, exact export | 14,000 | 8101 | 50.73% | 50.82% | −0.08 (−0.80, +0.56) |
| Trained vocabulary rows | 14,000 | 8101 | 48.12% | 48.76% | −0.64 (−1.34, +0.02) |

Seed-to-seed spread of unchanged controls: +1.10 (+0.28, +1.97) at 2 GiB grouping and 0.00
(−0.86, +0.89) at 1.75 GiB. Long controls: 50.82% and 51.07% greedy, 48.42% and 48.19% sampled.

Data: 768 against 192 distinct records at 2,370 updates (+1.64 (+0.60, +2.66) and +2.45
(+1.61, +3.24); 1.41 and 0.39 passes by realized groups; calibrations fitted different values);
256 against 64 records 47.1–47.5% against 26.5% for one-bit, 58.0% against 57.5% for the 4-bit
control. Writing: replay's exact choice differed at 1.71% of positions for 16-slot batched answers
and 1.68% for one-slot server answers. Soft-target probe (192 prompts, 41,049 positions): top-32
holds 99.07% of target mass on average (98.2% at the 10th percentile); 4.6% of greedy choices lie
outside the old 68,728-row shortlist (multilingual 34% of mass). Round profile (runtime-v8,
native Linux, leader, depth 3, short prompts, existing record): drafter attention projections
0.200 ms of a 7.706 ms round (2.6%; 2.5–2.6% at depths 1 and 7).

## Scope and limitations

- One completed seed at 14,000 updates for each long feature arm; production is 1.25 million
  updates, so small long-run gains are not excluded. Soft targets ran only at 2,370 updates.
- Trained vocabulary rows trained in F32 and lost about 0.5 DEV top-1 points at Q4_K export; a
  rounding-aware rows arm was not run.
- Invalid and excluded arms: `long/arms/attn-s1-stale-rounding-quantizer-export` and
  `attnx-s1-stale-rounding` never trained through rounding (a cache keyed on the parameter version,
  which fused AdamW does not bump; served −0.87 and −1.14); `rows-s1-oom-175` ran out of memory;
  `tables/failed` state-budget and refused attempts. The long rounded attention arm fitted
  different calibration values from its control.
- Rounded attention trains at about 1.4 updates/s against 5.3. Its final DEV split (36.1% trained,
  35.0% exported) is partly explained by a diagnostic that re-rounded decoded export values; that
  diagnostic is fixed, and the served screen is unaffected.
- DEV screen only; no gate or held-out data.

## Decision

The release recipe uses frozen Baseline tables, hard labels, batched writing and about one pass
over 100K distinct records. `attention_tables` (`trained`, `rounded`), `vocabulary_rows` and
`soft_targets` stay as tested opt-in settings. One-bit attention tables are not built for speed.

## Evidence and reproduction

- Code: commits `3de216a4` (trained attention), `d1f3d559` (vocabulary rows, plain resume),
  `2fa5ecbb` (soft targets), `073e18bd`/`3f9c8dfa`/`d0f19bef`/`ff9507ae` (rounding, exact export,
  cache refresh, diagnostic fix).
- Runs: `artifacts/datasets/retrain-01/{scaling,spending,tables,soft,long,soft-probe}/` with each
  arm's intent, journal (`output/events.jsonl`), export and `screen*/result.json`; scripts `arms.py`,
  `run.py`, `long.py`, `probe.py`. Profile: `artifacts/benchmarks/runtime-v8-linux-01/profile-v8-leader/round-profile.json`.
- Statistics recomputed from screen rows (`native.accepted`/`native.proposed`, paired by task).

## Artifact disposition

Retained: every arm's intent, journal, logs, exports, shortlist variants and screen results; the
soft-probe summary; the 1,000-record long fused corpus (11 GB) until the owner decides on rounded
attention seed 2, then retire. Discarded earlier with their decisions recorded: training state
binaries of finished arms (no resume needed; exports retained), raw replay taps after fusing.
The 768-record fused inputs (`spending/fused-768`, `soft/fused`, about 11 GB) are retired when the
rental starts; regeneration needs their streams, which are retained.

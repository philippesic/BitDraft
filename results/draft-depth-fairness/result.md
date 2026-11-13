# Fixed depth 2 overstates HE4 against Baseline by about 8%; fixed depth 3 is fair under exact verification

Every benchmark runs drafters at draft depth 2 with exact verification
(`GGML_CUDA_DECODE_CONSISTENT=32`). The [consistency study](../cuda-decode-consistency-cost/result.md)
found that this mode makes depths above 3 expensive, and that Baseline gains 6.6% from
depth 3. HE4's drafts are rarely accepted beyond the first positions, so a shallow fixed
depth might cost Baseline more of its potential than HE4's. This study measured each
drafter's best depth on the 120-request Q4KM panel and compared HE4/Baseline speed ratios
under four protocols.

The current depth-2 protocol favours HE4. With each drafter at its own best exact depth,
HE4 and arm D reach 0.80 and 0.82 of Baseline's complete-request speed, against 0.87 and
0.88 at fixed depth 2. Fixed depth 3 gives 0.80 and 0.82, within 1% of the best-depth
protocol. Exact mode does not change acceptance on identical text, and it does not
systematically favour HE4.

## Outcome

Q4KM target, 120-request DEV panel (longer caps), exact mode unless marked "off". Complete
is emitted tokens over summed HTTP request time. Decode uses the server's token-prediction
time only. Cells with two runs show their mean, and each run is in [result.json](result.json).

| Drafter, depth, mode | Runs | Complete tok/s | Decode tok/s | Accepted per round | Decode ms per round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, 2, exact | 2 | 200.8 (204.3, 197.3) | 256.9 | 1.28 | 8.86 |
| Baseline, 3, exact | 2 | 217.3 (217.7, 216.9) | 281.1 | 1.60 | 9.24 |
| Baseline, 4, off | 1 | 224.0 | 289.9 | 1.78 | 9.60 |
| HE4 220,095, 2, exact | 1 | 174.2 | 213.2 | 0.82 | 8.52 |
| HE4 220,095, 3, exact | 1 | 172.7 | 210.4 | 0.94 | 9.21 |
| HE4 220,095, 3, off | 1 | 167.8 | 213.5 | 0.93 | 9.02 |
| Arm D, 2, exact | 2 | 175.9 (174.5, 177.2) | 220.7 | 0.95 | 8.85 |
| Arm D, 3, exact | 2 | 178.1 (176.8, 179.4) | 220.2 | 1.12 | 9.63 |
| Arm D, 3, off | 1 | 187.2 | 234.7 | 1.11 | 8.98 |

Baseline's depth-2 first run and its depth-3 and depth-4-off first runs come from the
consistency study (same runtime, models and workload, earlier the same day). The rest are new.
Every exact run matched target-only output on all 120 requests. Each mode-off run differed in
70–71 of them.

**Best depths.** With exact mode, Baseline is fastest at depth 3. Depth 4 is ruled out by
the consistency sweep: its rounds cost 28% more for 8% more tokens per round. HE4 is
fastest at depth 2, though depth 3 is within 1%. Arm D ties: depth 3 is 1.3% faster
complete-request and 0.2% slower in decoding, inside the 1.5% run-to-run spread of its
repeats. Neither HE4 state gains from depth 3 because each extra round costs 7–9% more
while adding only 0.12–0.17 accepted drafts. Baseline adds 0.32 accepted drafts for 4%.
With the mode off, Baseline's best is depth 4–8 (the consistency study's screen sweep is
flat over that range). Both HE4 states were measured at depth 3. The screen sweep shows
HE4 decoding equally fast at depths 2–4 with the mode off.

| HE4/Baseline speed ratio | Cells (Baseline / HE4 / arm D) | HE4 220,095: complete | decode | Arm D: complete | decode |
| --- | --- | ---: | ---: | ---: | ---: |
| (a) Fixed depth 2, exact (today) | 2 / 2 / 2 | 0.868 | 0.830 | 0.876 | 0.859 |
| (b) Fixed depth 3, exact | 3 / 3 / 3 | 0.795 | 0.748 | 0.820 | 0.783 |
| (c) Each at its best exact depth | 3 / 2 / 3 | 0.802 | 0.758 | 0.820 | 0.783 |
| (d) Each at its best depth, mode off | 4 / 3 / 3 | 0.749 | 0.736 | 0.836 | 0.809 |

- **(c) is the fair reference.** Exact output is the project's correctness contract, so
  exact mode is the setting in which drafters are compared. Within it, each drafter should
  run at the depth that serves it best.
- **Today's protocol (a) overstates HE4.** Against (c), HE4's ratio is 8.2% too high on the
  complete-request clock and 9.4% on the decode clock. Arm D's is 6.9% and 9.7% too high.
  The cause is that depth 2 caps Baseline below its best depth, while HE4 loses nothing there.
  The bias exceeds the run-to-run variation (0.4–3.5% between repeats).
- **(b) does not flatter HE4.** Fixed depth 3 is within 0.8% of (c) for the 220,095 state,
  slightly against it, and equals (c) for arm D.
- **Exact mode is not a systematic HE4 advantage.** Moving from (c) to (d) lowers the
  220,095 state's ratio by 6.6% and raises arm D's by 2.0% (complete). Baseline gains 3.1%
  without the mode. On the decode clock the moves are −2.9% and +3.3%.
- The 220,095 state loses with the mode off because it repeats the consistency study's
  unexplained stalls. Its requests spent 19.2 s outside the server's prompt and decode
  timers, against 13–15 s for every other drafter run, with single-request gaps of up to
  1.65 s. Arm D does not stall.

**Acceptance is unchanged by the mode.** Mode-off runs produce different text after
near-ties, so aggregate acceptance compares different continuations. On requests whose
output was identical in both modes, accepted drafts agree within 0.25% on the panel and
0.5% at every screen depth:

| Requests with identical output in both modes | Requests | Accepted, exact | Accepted, off |
| --- | ---: | ---: | ---: |
| Panel, HE4 220,095, depth 3 | 50 of 120 | 6,005 | 6,020 |
| Panel, arm D, depth 3 | 50 of 120 | 6,249 | 6,253 |
| Screen, Baseline, depths 2/3/4/6/8/15 | 28–36 of 60 | 2,002 / 2,378 / 2,529 / 1,901 / 2,163 / 2,230 | 2,008 / 2,378 / 2,526 / 1,907 / 2,161 / 2,228 |
| Screen, HE4 220,095, depths 2/3/4/6/8/15 | 28–36 of 60 | 1,763 / 1,925 / 2,135 / 1,657 / 1,880 / 1,914 | 1,771 / 1,932 / 2,129 / 1,660 / 1,881 / 1,912 |

The panel's per-position survival at depth 3 (first one, two, three drafts all accepted)
is 0.74, 0.50, 0.35 for Baseline, 0.51, 0.27, 0.16 for HE4 220,095 and 0.58, 0.33, 0.21
for arm D.

**Acceptance rates depend on depth.** At depth 3 the share of proposed drafts accepted
falls to 53% for Baseline, 31% for HE4 and 37% for arm D, from 64%, 41% and 48% at depth
2, because the third position is rarely accepted. Screen acceptance at the two depths is
not comparable, and saved references bind depth for this reason. Accepted per round is the
depth-robust quantity.

## Scope and limitations

- One RTX 5080, runtime v5, Qwen3.5-4B Q4KM target, F16 KV, context 32,768, greedy,
  concurrency one. NVFP4 and the 60-request screen were not measured here, except for the
  screen acceptance check above, which reuses the consistency study's runs.
- The 120 requests are a reused DEV cohort. Most cells are single runs. Baseline's two
  depth-2 runs, from different sessions, differ by 3.5% complete-request and 2.5% decode.
  Its depth-3 runs differ by 0.4%, and arm D's repeats by 1.5%. Ratios mix this study's
  runs with the consistency study's, so they are descriptive. The 7–10% depth-2 bias is
  larger than this variation. The 1% differences between (b) and (c) are not resolved.
- Protocol (d) uses one Baseline run at depth 4 and one HE4/arm D run at depth 3. Their
  neighbouring depths are inferred from the consistency study's screen sweep, not measured
  on the panel.
- Arm D's exact-mode rounds cost 4–5% more than the 220,095 state's at the same depth
  (9.63 vs 9.21 ms at depth 3), although both have the same architecture and equal round
  cost with the mode off (8.98 vs 9.02). The drafter's own traced time explains 0.12 ms of
  the 0.42 ms gap. The rest is unexplained; the 220,095 state has one run per cell.
- The conclusion holds for today's runtime. The cap at 3 comes from exact mode's cost
  jump at depth 4. A runtime change that removes it, such as the four-warp wide groups
  proposed in the consistency study, would move Baseline's best depth deeper. Fixed depth 3
  would then flatter HE4 by roughly Baseline's mode-off depth-4 gain over exact depth 3,
  3% on this panel. A drafter whose later positions improve would also gain from depth.

## Decision

Proposed, not decided:

- Switch the benchmark default to depth 3 for every drafter, with exact mode kept. On
  today's runtime it matches the best-depth protocol (c) within about 1% for every
  measured drafter, keeps one shared protocol and set of saved references, and removes the
  7–10% overstatement of HE4 in today's depth-2 ratios.
- Report historical depth-2 HE4/Baseline ratios as favourable to HE4 by about 8%.
- Re-establish depth fairness with a panel sweep (Baseline at depths 2–8, the current
  candidate at 2–4) after any runtime change to verification cost. Set the shared depth to
  Baseline's best exact depth, and check that the candidate's own best is within run
  variation of it.

No benchmark default, saved reference or runtime was changed. The saved controls are to
be re-recorded separately after the runtime fixes.

## Evidence and reproduction

[result.json](result.json) holds every panel run used (new and reused): totals, per-position
survival, drafter time, time outside the server timers, divergence from target-only in both
modes, and per-request rows (task, group, category, input/emitted tokens, stop, wall,
prompt/prediction ms, proposed, accepted). It also holds cell means, the protocol ratios,
the paired acceptance check, executed plans and input/script hashes.

The harness and analysis are byte-identical to the consistency study's executed
[harness](../cuda-decode-consistency-cost/executed-harness.txt) (`f9d4ce64…`) and
[analysis](../cuda-decode-consistency-cost/executed-analysis.txt) (`0dd10d7a…`). This
study's [result builder](executed-result-builder.txt) regenerates result.json from the raw
outputs, with the analysis beside it as `analyze.py`. Run from the repository root with the
small check environment:

```sh
PYTHONPATH=. uv run --locked --no-sync --offline python -B harness.py PLAN OUTPUT
PYTHONPATH=. uv run --locked --no-sync --offline python -B build_result.py RUN_DIR EARLIER_RUN_DIR result.json
```

Inputs (read-only, main checkout): target `3874209…`, Baseline `83586a63…`, HE4 220,095
export `99d2be79…` (`artifacts/qualification/original-zero-export-02`), arm D export
`6a07cd5f…` (`artifacts/training/recipe-test-arm-d-01/exports`), runtime v5 server
`c83a7965…`, panel workload `f25c7504…` (DEV role). Target-only references and the reused
Baseline runs are the consistency study's raw outputs.

GPU use was 42.9 minutes of arm time (ten panel runs, including model load), executed
2026-11-13 under the project lease in three batches. Batch 2 waited for another agent's
lease and retried without interrupting it.

## Artifact disposition

- **Raw outputs, disposable after review.** This study's ten runs (per-request tokens, server
  logs; 5 MB) are in this worktree's ignored `artifacts/depth-fairness/`, with plans,
  logs and scripts. result.json holds every value reported here except token IDs, and
  rerunning the plans reproduces them. Exact tokens and server logs are lost when the
  worktree is removed.
- **Reused raw outputs.** Target-only and the reused Baseline runs remain in the consistency
  study's worktree (`t3-f8f8df23/artifacts/consistency/`) under that study's disposition.
  Rebuilding result.json needs them.
- **Read-only inputs.** No models, saved references or runtimes were created, changed or
  deleted.

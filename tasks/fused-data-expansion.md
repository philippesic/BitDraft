# Fused storage retest, then data expansion or the two-pass run

## Goal

Decide whether fused F16 storage is good enough to train on. If it is, shrink the
TRAIN set 8×, roughly double it with data aimed at our weakest scenarios, and train
longer. If not, run the two-pass release recipe on the existing FP16 data.

## Authorized

Owner go 2026-11-18 for the whole chain in this file:
- the retest;
- on fail, the two-pass FP16 run;
- on pass, fusing, the expansion captures and the long fused run.

The narrow deletion is authorized too: the FP16 tap files of the 6K bank and
`train-expansion-01` may go through authenticated retirement plans if fused passes.
`long-sessions-01` taps, manifests, tokens, labels, sources, helpers and DEV captures stay.

Owner decision 2026-11-18: the pass bar is a pooled lower bound of −1.1 points or better.
The earlier result (−0.36, −1.07 to +0.32) counts as a pass; "I just don't want it to get
worse."

## Done when

- The fused verdict is recorded under `results/` with the rule below.
- Pass: the expanded fused corpus is admitted and a two-pass run on it is screened
  against leader `ae6569f6…` and on the release scenarios.
- Fail: the two-pass FP16 run is screened the same way.

## State

### Fused retest (passed 2026-11-18 02:35)

Verdict: fused minus FP16, pooled over three seeds on the 120-request panel, −0.18 points
(−0.39, +0.03): **pass**, clearing both the owner's −1.1 bar and the original −1.0.

| Seed | Difference (95%) |
| --- | --- |
| 8101 | −0.14 (−0.50, +0.20) |
| 8102 | −0.24 (−0.57, +0.06) |
| 8103 | −0.17 (−0.52, +0.18) |

- Panel acceptance: 47.43% fused against 47.62% FP16.
- Supplementary 60-request screen: −0.43 (−1.02, +0.14).
- Evidence: `fused-retest/verdict.json`.
- The panel interval came out much narrower than the ±0.45 guess.

### Retest design

Question: does training on fused F16 storage cost at most 1 acceptance point against
FP16 storage? One point of acceptance is about 1.1% request speed at depth 3. The
exported drafter and runtime are identical either way; only training inputs change.

- Arms: a third seed (8103) of the earlier pair, each a fresh 8,209-update calibrated run
  on the 8,633-record `train-expansion-01` mixture the earlier arms used (not
  `train-final-01`): fused (`fuse-full-train-r20.json`, corpus rebuilt in about 15 GPU
  minutes, 56 GB) and FP16 control. Both new arms run one frozen source, executed source
  `8c3d9db5…`; the earlier controls ran `4bb823…`, which differs by fused calibration and
  disabled optional paths (Astra review 2026-11-18). Transfer to `train-final-01` is
  extrapolated.
- Evaluation: all six exports (three seeds × two storages; the four earlier exports are
  retained) on the 120-request Q4KM panel, candidate-only against saved runtime-v8
  references. The 60-request screen is 59/60 inside this panel.
- Rule, fixed before running: pooled paired source-group bootstrap of fused minus FP16
  acceptance over three seeds on the 120-request panel (`compare_screens.py` with matched
  `--control` lists); pass if the 95% lower bound is ≥ −1.10 points (owner, 2026-11-18). A fail means "not
  demonstrated", and the two-pass FP16 run follows. Each seed is reported. Fresh 60-request
  screens of the two new arms give the supplementary screen result; it does not decide.
- Detectability: two seeds on 60 requests gave ±0.70. About ±0.45 on the panel is a scaling
  guess, not measured: the bootstrap resamples requests, not training seeds.
- Cost: about 1.5–2 GPU hours (one fusion, two 35-minute runs, six panels).
- Preflight: the mechanism shows as `corpus_format: fused-mixture-v1` with the fused-4
  identity in the fused arm's intent. The control differs only in storage: same intent
  apart from corpus and format, same executed source. CPU admission passed for the control.
  The early gate is the verdict itself, reached within 2 GPU hours.

### Fail branch: two-pass FP16 run (`s15-twopass`, queued)

- Recipe: the 3-hour run's settings on `train-final-01`, cosine over 126,000 updates (2 ×
  63,011 groups per pass), warm-up 2,520, data-reuse cap 12 (mean-visit bound 10.8).
  CPU admission passed; peak new disk 41 GB.
- Execution: nine 14,000-update segments resumed exactly from each previous state (the
  route the step 11 pilot qualified). Each segment's export is screened; the previous
  segment's states are retired through `Store.retire_closed`. Fused-4 is retired first
  through its authenticated plan.
- Expectations: about 1.1 hours per segment, about 10.5 hours in all. Mid-run screens
  are not annealed and sit below a run stopped there.
- Kill criteria: the screen at 28,000 updates below 52.0% (the 3-hour run's 14,000-update
  point), two consecutive falling screens, or any failed segment or screen.

### Expansion if fused passes (queued as `queue21.sh`)

Current gaps, ours minus Baseline acceptance, final export, step 14 runs: writing −7.0 to
−7.5, thinking-on −6.7 (maths −7.1, code −5.8), conversation −5.5 to −5.9, reasoning
−4.3 to −5.9, documents −3.8 to −4.4 (−6.1 at 8–16K tokens), tool-call harness code −6.3,
sampled −4.3, greedy code about even, structured +27, agentic suite +17.

Current mix (4.49M supervised positions): reasoning 27%, conversation 21%, writing 21%,
single-turn code 18%, sessions 7%, documents 5%, structured 1%.

| Area | Records | Positions | Source |
| --- | ---: | ---: | --- |
| Thinking-on maths and code | 1,000 (600 / 400) | about 1.5M | MATH train (7.2K pool), CommitPack edit pool |
| Writing | 900 | about 1.05M | UltraChat writing (33K pool) |
| Conversation | 700 | about 0.6M | UltraChat multi-turn and oasst2 (97K pool) |
| Reasoning, thinking off | 300 | about 0.33M | MATH |
| Long documents | 300 | about 0.07M | QASPER (350 left) |
| Agentic long projects (11–20K start) | 152 sure, up to about 400 | 0.06–0.17M | `plan-long-projects` pool: 152 prepared and uncaptured (23 generated, 129 not), about 250 more if the 848 unexamined planned families keep the 30% preparation yield |
| Agentic short sessions | 381 | 0.12M | `train-expansion-01/sessions-2`, already generated |
| Single-turn code and tool calls | 600 | about 0.17M | CommitPack edits |
| Structured | 0 | 0 | already +27 over Baseline |
| **Total** | **about 4,600** | **about 4.0M** | |

Selection A (CPU, done 2026-11-18, `datasets/expansion-03/bank-a5`) found the pools
smaller than planned:
- Commit-edit prompts are nearly used up. 400 plain ones fill, and none are left for
  thinking-on code; thinking-on is 640 maths and 360 general reasoning.
- OpenAssistant writing gave 24 of 60.
- QASPER has no 8K+ papers left, so all 300 papers come from the 2–8K band.

Selection A holds 3,564 prompts: 2.68M input tokens, at most 409 GB of FP16 features
(about 260 GB expected). QASPER is 63% of its input tokens.

Selection B follows the session generation: about 412 long-project sessions (389 from
`skip=171` of the 560 prepared tasks plus 23 generated leftovers) and 381 short sessions.

Refused selection attempts `bank-a`–`bank-a4` are small leftovers, to be retired at
closeout.

All selection goes through the existing exclusion checks: heldout24, confirmation-v2
protection, DEV banks and workloads, DEV identities and token prefixes, agentic project
families and code content. Pools are ample historically; their exclusion counts are checked
fresh at selection. Sampling stays proportional to positions. Data and training length both
change, so gains are not attributed to the targeting itself.

Sequence and cost:
- Retire fused-4, the seed-8103 states and the long run's 14K/28K branch states (fresh run
  chosen), then fuse `train-final-01` (about 71 GB).
- On the owner's OK, retire FP16 tap files only where token replay is qualified: the 6K
  bank and `train-expansion-01` (replay bit-identical on 40 records, including 22 from the
  6K bank and 9 Windows captures). That frees about 440 GB.
  - Keep `long-sessions-01` taps (120 GB) until replay is qualified on its records.
  - Keep every manifest, token stream, label, source record and helper, and all eight-tap
    DEV captures (DEV diagnostics read them).
  - Use authenticated retirement plans.
- Capture in FP16 chunks, fusing and retiring each chunk: about 12–14 GPU hours, about
  77 GB fused. Budget the peak (chunk plus fused output plus checkpoints plus reserve) per
  chunk.
- Two capture components (A: prompts, B: sessions), each fused and its taps retired. The
  combined `train-final-02-fused` mixture has exactly eight components, the limit.
- Two passes, with updates derived from the admitted `groups_per_pass` (about 240,000,
  18–20 GPU hours). Screen every 14,000 updates.
- Stop early on any inexact output, nonfinite loss, or two consecutive screens that fall.

Limits: a gap can reflect one-bit capacity rather than missing data. Every learning-curve
slice rose through 42,000 updates, but gains per category are not predicted. Only
the long-project pipeline yields new agentic sessions; more needs a re-plan beyond 1,850 families.

Expansion progress (2026-11-18):
- Done:
  - fused-4, the seed-8103 states and the long run's 14K/28K states are retired;
  - `train-final-01-fused` is built (9,047 records, 71 GB, within its bound);
  - FP16 taps are retired for he4broad (110 GB), bank-v8 (137 GB) and
    train-expansion-01 (195 GB);
  - 389 new long-project sessions are generated.
- Incident: two bugs of mine let capture A fill the disk to about 1 GB free at 07:48. It
  stopped cleanly after 1,901 of 3,564 records ("insufficient capture space").
  - The bank-v8 and expansion-01 plans listed manifests inside their own roots as retained
    evidence, so the retirement tool refused them.
  - The queue logged that refusal and continued.
  - Fix: manifests are now protected rather than retained, and a failed retirement stops
    the queue. Both plans were reapplied (311 GB free), and capture A resumed its verified
    records at 08:04.

Paused 2026-11-18 16:12 for an owner system shutdown:
- Done: capture A (3,564 records, 35 GB fused); selection B (405 long-project and 378
  short sessions); capture B; the combined `train-final-02-fused` set; the long fused run
  `s16-fused-long`, segment 1 (14,000 updates, screen 48.72%).
- Segment 2 was stopped with SIGTERM. It published a recoverable state at 17,311 updates
  (`seg-2/output/states/76c89905…`, `latest.json`).
- Resumed 2026-11-18 17:31 after the reboot. The interrupted output moved to
  `seg-2/interrupted-17311/` (its states are retired once segment 2 completes). Segment 2
  now resumes from that state with the same 28,000 budget.

Kill criterion fired 2026-11-18 18:17: the screen read 50.59% at 28,000 updates, below 52.0%
(48.72% at 14,000). The run stopped, and its 28,000-update state is kept
(`seg-2/output/states/ada84a02…`), so it can resume exactly.
- The bar was miscalibrated. It came from the 3-hour run at 14,000 of 42,000 updates, where
  the cosine step was down to about 77%. This run is still at about 99%, and mid-run screens
  at full step understate the result.
- Not only schedule: at a matched step size, the 3-hour run read 49.8% after 7,000 updates,
  while this one needed 28,000 updates to reach 50.6%. The new data is 2.7× larger and has
  less screen-like data per update; the 60-request screen is noisy (about ±0.7).
Owner go 2026-11-18 (evening):
- Resume the long run from 28,000 updates, without the 52% bar.
- Rebuild the 60-request training screen to cover every release scenario, re-measure the
  leader and the 14K and 28K exports on it, and keep training.

Resumed 18:38 (`queue21.sh`, jobwatch `fv-queue21g`; `queue21-v1.sh` is the earlier copy):
- Each segment now runs the old screen plus the 120-request panel.
- The stop rule is two consecutive falling panels; any failure still stops the run.
- A `HOLD` file in the campaign folder pauses the queue between segments.
- Once `scripts/screen2_spec.py` exists, the queue also runs the new screen.

New screen (`artifacts/benchmarks/screen-v2/`): 60 requests reused from the DEV panels with
known ancestry, 60 distinct groups.

| Scenario | Requests | Cap |
| --- | ---: | ---: |
| Thinking-on (8 code, 6 reasoning, from the thinking suite) | 14 | 256 |
| Agentic coding (6 project histories, 6 tools-harness requests) | 12 | 128 |
| Chat and writing (8 writing, 8 conversation) | 16 | 128 |
| Reasoning | 6 | 128 |
| Coding requests | 6 | 128 |
| Long documents (3 at 8K+, 3 at 2–8K) | 6 | 128 |

The `screen-v2` tier is merged (`38311841`). It is frozen as source `060915bf…`, and
`verify-inputs` passes.

`screen2/setup.sh` (jobwatch `fv-screen2-setup`) does the rest:
1. Set `HOLD`, wait for the boundary after segment 3, and stop the queue.
2. Establish target, Baseline and leader on screen-v2 and record the three references.
3. Install `screen2_spec.py` and restart the queue as v3 (jobwatch `fv-queue21h`).

Restarted, the queue re-measures the 14K, 28K and 42K exports on the new screen, then
measures every new segment.

Done 19:53. The three-arm establish took about 2 minutes. Queue v3 runs under jobwatch
`fv-queue21h`. The "state retirement after segment 2 failed" line on the re-run is expected:
segment 1's states were already retired.

Acceptance in %:

| Updates | Old screen | Panel | New screen (screen-v2) |
| ---: | ---: | ---: | ---: |
| 14,000 | 48.72 | 40.98 | 49.13 |
| 28,000 | 50.59 | 42.73 | 51.70 |
| 42,000 | 51.40 | 43.83 | 52.06 |
| 56,000 | 52.69 | 44.48 | 52.87 |
| 70,000 | 53.01 | 45.01 | 53.69 |
| 84,000 | 53.37 | 45.38 | 53.68 |
| 98,000 | 53.80 | 45.97 | 54.92 |
| 112,000 | 54.11 | 46.32 | 55.66 |
| 126,000 | 54.35 | 46.44 | 55.02 |
| 140,000 | 55.15 | 46.91 | 56.32 |
| 154,000 | 55.02 | 47.50 | 56.08 |
| 168,000 | 56.07 | 47.85 | 57.53 |
| 182,000 | 56.98 | 48.70 | 57.77 |
| 196,000 | 57.69 | 49.14 | 58.08 |
| 210,000 | 58.26 | 49.12 | 58.17 |
| 224,000 | 58.63 | 49.69 | 58.94 |
| 238,000 | 58.83 | 49.98 | 59.35 |
| 252,000 | 57.91 | 50.04 | 59.07 |
| 266,000 | 59.19 | 50.40 | 60.03 |
| 280,000 | 58.23 | 50.39 | 59.60 |

Owner decision 2026-11-19 (14:30): only panel falls of more than 0.3 points count toward the
stop rule. The 0.01 fall at 280,000 updates would otherwise have let noise end the run
before its wind-down. `screen2/swap-v4.sh` (jobwatch `fv-swap-v4`) installs it at the
boundary after segment 21 and restarts as v4 (`fv-queue21i`). The release queue is
re-armed as `fv-queue22d`.

Earlier false alarm: queue22's first trigger matched the old "queue21 failed" line from the
disk incident, and the fix was off by one. Two stale failure wakes followed; training was
not affected.
| Leader `ae6569f6…` | 57.6 | 48.7 | 57.38 |
| Baseline | 60.6 | 53.4 | 62.70 |

Queued behind the run is `scripts/queue22.sh` (jobwatch `fv-queue22`). It starts on
"queue21 done" or "stopped" and repeats the step 14 release comparison on the last export:
- fresh three-arm timing against Baseline on all seven panels at depth 3, under the step 14
  source `8180f208…`;
- reversed-order repeats of requests, thinking and sampled.

Outputs go to `step16/`, taking about 2.2 GPU hours. Selection against the leader then uses
candidate-only screens (screen, screen-v2, panel) with `compare_screens.py`, followed by
the policy and scenario reports.

## Next

- Retest verdict, about 02:30 on 2026-11-18.
- Fail: the queue continues into the two-pass run with no agent step.
- Pass: `queue21.sh` (unit `fv-queue21`, jobwatch `fv-queue21`) runs the whole
  expansion and the segmented long fused run. Its kill criteria match the two-pass run.

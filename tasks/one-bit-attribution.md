# What does one bit buy? A 4-bit control for the leader

## Goal

Say how much of leader `ef068eba…`'s standing against Baseline comes from one-bit
feed-forward weights and how much from our training data, by training Baseline's own
4-bit architecture on the same data and comparing it with the one-bit drafter.

## Authorized

Owner go on 2026-11-20, about 00:00 PST: free rein until 08:00 PST the same day for this
comparison and other work that is helpful right now; no brand-new feature sets; builds go
to an Opus subagent. The 4-bit feed-forward arm runs as a control only. The owner's
2026-11-16 decision to drop 4-bit weights as a candidate format stands.

Lead's limits inside that go:
- the protected confirmation sets and heldout24 stay unspent and untouched;
- no selection, promotion or publication follows from these runs;
- test-list item 11 and a second tool-call convention are not started: the GPU schedule
  below fills the night.

## Done when

The matched pair and the control-against-leader comparison are measured or reported as
not achievable by the deadline, a result entry and FINDINGS line record them, raw outputs
have a disposition, and this file is deleted.

## State

Owner: Philip. Lead: the thread on branch `t3/one-bit-attribution`
(`/home/philip/.t3/worktrees/BitDraft/one-bit-attribution`). Campaign folder:
`artifacts/one-bit-attribution-01/`. GPU jobs go through pueue group `gpu`, watched by
jobwatch job `oba-chain1`. Timing marker: `artifacts/one-bit-attribution-01/TIMING-ACTIVE`.

Main was merged with `t3/candidate-screen` (commit `23f8b14e`, full gate passed, pushed)
so the build starts from current code.

### Preflight (2026-11-20)

**Question.** Would Baseline's 4-bit architecture, fine-tuned on `train-final-02-fused`,
do as well against Baseline as the one-bit leader?

**Decision-changing results.**
- Control against leader: the control's speed ratio to Baseline matches or beats the
  leader's on at least four of the five release scenarios. Then data, not one bit, carries
  the result, and the product framing and any further one-bit training change.
- Matched pair at equal data, order and updates: the control leads the one-bit arm by more
  than 4 acceptance points on the 120-request panel (one bit costs more than it buys), or
  by less than 2 (one bit pays for itself).

**Where the 2 and 4 come from** (round profile and the item 1 kernel result): one-bit
feed-forward takes 136 µs per round against 292 for 4-bit tables, and our private Q4_K
vocabulary table scores in 0.40 ms against 0.57 for the target's table that Baseline
borrows. Together 0.33 ms of a 7.7 ms round, about 4.3% of speed; the kernel alone is
about 2%, and it is the only part specific to one bit. One acceptance point is worth about
1.1% of speed on short prompts.

**Can the test detect the effect?** Yes by existing data. The panel resolves about 0.5
points and timed ratios about 1%. Fresh calibrated one-bit runs of 8,209 updates score
about 47.6% on the panel; Baseline, where the control starts, scores 53.4%. The expected
gap is therefore 5 points or more unless fine-tuning hurts the control, which is itself a
result.

**Preconditions and their checks.**
- The control differs only in the treatment: same data, seed 8101, grouping, blocks,
  schedule shape and update count as the one-bit arm. The step size cannot match (different
  parameters), so two values are piloted.
- The control's mechanism ran: its feed-forward tensors differ from Baseline's, and its
  update-0 export, screened on the GPU, shows what our quantization route costs before
  training.
- One end-to-end pilot of 10 minutes or less: about 2,000 updates, export, native screen
  with exact outputs.

**Expectations and kill criteria.**
- One-bit arm, 21,000 updates: about 89 minutes at 14,200 updates an hour; no export by
  110 minutes ends it.
- Control pilot: DEV loss must not rise above its update-0 value, and the screen must
  complete with exact outputs. A failing step size is dropped; if both fail, stop and
  report.
- Early gate before the 21,000-update control: the better pilot's screen acceptance is at
  least Baseline's minus 1 point.
- The timed release chain for the control starts by 05:15 PST or is cut to the scenarios
  that fit.

**Unverified.** Short runs can rank differently from long ones: the leader trained 16
times longer than the matched pair. The control-against-leader comparison favours the
leader on length and the control on starting point. One seed.

### Queue

| Task | Content | Status |
| --- | --- | --- |
| 0, 1, 6 | One-bit arm `w1-21000` (recipe `9dc9621b…`) with screen, panel and screen-v2 | Done 02:33 |
| 2–4 | Control pilots `plain-2000-{3e-5,1e-5,1e-4}`, screen and panel | Done |
| 5 | One-bit arm `w1-2000`, screen and panel | Done |
| 7 | Control `plain-21000-3e-5`, then screen, panel and screen-v2 | Running since 02:34, about 70 minutes |
| 8 | Timed release chain, control as candidate: requests, thinking, sampled, agentic, screen | Queued after 7 |
| later | Reversed-order repeats or a 1e-5 control, decided on task 8's result; leader on NVFP4 if time remains | Not queued |

Jobwatch job `oba-control` watches tasks 7 and 8. Task 1 failed only at screen-v2, which
needs its own frozen source `060915bf…`; task 6 reran it.

### Measured so far (candidate-only, exact outputs, acceptance in %)

| Arm | Updates | Screen (60) | Panel (120) | Screen-v2 (60) |
| --- | ---: | ---: | ---: | ---: |
| Baseline, where the control starts | 0 | 60.64 | 53.35 | 62.70 |
| Control, step 1e-5 | 2,000 | 63.12 | 54.43 | |
| Control, step 3e-5 | 2,000 | 63.27 | 54.55 | |
| Control, step 1e-4 | 2,000 | 61.86 | 53.28 | |
| One-bit | 2,000 | 50.67 | 43.71 | |
| One-bit | 21,000 | 55.04 | 46.75 | 55.66 |
| Leader `ef068eba…`, one-bit | 335,812 | 59.75 | 50.81 | 60.30 |

At 2,000 updates the control leads the one-bit arm by 10.84 panel points (10.24, 11.41)
and 12.59 screen points (10.85, 14.29), paired source-group bootstrap. Step 3e-5 leads
1e-4 by 1.27 panel points (0.94, 1.60) and ties 1e-5 (+0.11; −0.11, +0.34); its DEV loss is
the lowest (2.428 against 2.464 and 2.440, from 2.490). Step 3e-5 is used for the 21,000-update
control (lead's choice; early gate passed: 63.27 against a bar of 59.6). Peak device memory
was 12.0 GiB for the control and 12.9 for one-bit; the control trains about 17% faster per update.

Build (Opus, delivered 00:37, merged as `1bd92b47`, full gate passed): intent setting
`"feed_forward": "plain"` trains the 18 feed-forward tables as full-precision tables in
Baseline's graph and exports Baseline's layout with Q4_K/Q6_K tables. Executed source
`94852e01…` in `artifacts/one-bit-attribution-01/executed-sources/`. Its update-0 export is
byte-identical to the Baseline file, so the control starts exactly at Baseline and the
update-0 screen is Baseline's saved reference. The control scores against the target's own
vocabulary table, as Baseline is served; the one-bit arm uses our Q4_K table. Fresh runs
only; campaign checks and round diagnostics are not wired for it.

Step sizes: the three pilots span movement budgets of about 0.01, 0.03 and 0.1 per weight
(Q4_K rounding error is about 0.005; weights have RMS 0.08). The 21,000-update control at
1e-5 has a budget of about 0.1.

### Artifacts

Purpose: the matched pair and the control comparison. Inputs: `train-final-02-fused`, the
public BF16 drafter, Baseline, executed sources. Disposal at closeout: keep each arm's
final export, state, journals and check results; retire nothing of other campaigns.

## Next

Wait for the build report. Freeze its source, screen the update-0 export, pilot two step
sizes, train the 21,000-update control, then run the release chain with the control as
candidate and compare with the leader's 2026-11-19 chain.

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
| 0 | One-bit arm `w1-21000`, release recipe, source `8c3d9db5…`, recipe `9dc9621b…` | Running since 00:05 |
| 1 | Its screen, panel and screen-v2, candidate-only | Queued |
| 2 | Leader `ef068eba…` on the NVFP4 screen and panel, candidate-only | Queued |
| 3, 4 | One-bit arm `w1-8209` and its checks: fallback match if the control must be short | Queued |

Build: [Build 4-bit feed-forward control] (Opus, delegated 00:07), brief
`artifacts/one-bit-attribution-01/briefs/build-plain.md`.

### Artifacts

Purpose: the matched pair and the control comparison. Inputs: `train-final-02-fused`, the
public BF16 drafter, Baseline, executed sources. Disposal at closeout: keep each arm's
final export, state, journals and check results; retire nothing of other campaigns.

## Next

Wait for the build report. Freeze its source, screen the update-0 export, pilot two step
sizes, train the 21,000-update control, then run the release chain with the control as
candidate and compare with the leader's 2026-11-19 chain.

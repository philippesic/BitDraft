# Feature validation before the release candidate

## Goal

The next candidate should be the final candidate for a first release: a single drafter
that is faster than Baseline. This screen must tell us everything needed to construct that
candidate and train it once. It does not train it.

Starting position on native Linux: the candidate accepts 55.2% on the 60-request screen
against Baseline's 60.6%, at 0.96–1.00 of Baseline's speed. Its drafter is 22% faster than
Baseline's; acceptance is the limit.

**The speed arithmetic that judges every item** (from the native round profile; an
independent review worked it out and the lead agent agrees):

- one point of acceptance is worth about 1.1% of speed;
- 0.1 ms off a 7.7 ms round is worth about 1.3%;
- from the weak end of the range (0.96), parity needs about 3.7 points or 0.31 ms;
- recovering the whole 5.4-point gap alone gives about 1.02, so a real win needs
  acceptance and drafter time together;
- the prizes on the drafter side are small: vocabulary scoring 0.41 ms, token selection
  0.11, attention tables 0.20, feed-forward 0.17.

## Authorized

Nothing may run yet. This version, written 2026-11-16, folds in the owner's decisions of
that day, a 157-idea research sweep
(`artifacts/research/field-sweep-2026-11-16/ideas.md`) and an independent review by a
second model. It is a proposal. Each group starts only on the owner's explicit go, and GPU
or CPU-heavy work waits while another thread holds the GPU.

## Done when

Every question under "What the screen must hand over" has an answer backed by a check, or
the checks show no plausible speed margin and that is reported. Training the candidate is
a separate decision. Every check run has a recorded outcome under `results/`; then delete
this file.

## State

Owner: Philip. Proposal by the lead agent; no check has run.

### Rules for every check

- Every result is translated into expected speed with the arithmetic above; only measured
  native speed decides.
- One change per run against a matched control, on fresh 30-minute runs (about 45 minutes
  with checks) unless stated, compared at equal updates and equal data.
- The real screen resolves about 1.1–1.4 points. The 120 DEV windows resolve far less: 2.5
  points is three windows. DEV results nominate; they do not confirm.
- A winner inside two points gets a second seed before it is adopted. Finalists get a
  one-hour run.
- Storage and simplification changes are judged by a stated allowed loss (proposed: at
  most one screen point, to be confirmed by the owner), not by "no difference seen".
- One-bit weights are evaluated as hard signs at every decision point.
- heldout24 is never used in this screen.

### Group 0: where the gap is (no training; under an hour of GPU)

These locate the problem. They do not prove causes.

| # | Check | Use |
| --- | --- | --- |
| 0a | Fix the release workloads and the current native gap on each | The target every later number is read against |
| 0b | Candidate and Baseline at real four-slot round states: where the first wrong slot falls, first-choice and top-16 recall, round starts against random positions | Whether the deficit sits at round starts or at later slots. The candidate beats the original on random DEV positions (79% against 76%) yet trails Baseline by 5 points when served |
| 0c | Training labels by scenario and thinking setting beside the per-scenario gap | Whether the 0.88 on thinking-on requests is a data-mix gap |
| 0d | Candidate and Baseline on prompts from the training set | Localises the gap to seen or unseen prompts; it does not separate capacity from generalisation by itself |
| 0e | Clipping frequency and spike positions in the drafter's down inputs and the stored target values | Whether rare positions set the clipping bound |
| 0f | Input ranges per layer, split by anchor and draft slot | Whether per-slot rounding settings are worth a run |
| 0g | Gains set to 1 and midpoints to 0 in the trained candidate | Whether the trained drafter depends on them at all |
| 0h | Calibrated settings on an untrained drafter | Whether calibration lifts the 4–7% start |
| 0i | Trained candidate with feed-forward inputs unrounded, and with down inputs at 4 bits | Where input rounding costs acceptance |

### Group 1: training speed (about 1–1.5 GPU hours plus a small build)

The trainer multiplies each feed-forward table twice per pass (verified in
`Projection.forward`), runs every large multiplication in plain 32-bit arithmetic, and
synchronises with the host inside each projection. Expected gain is between 1.2–1.8 times
(reviewer) and 2–4 times (lead agent); the profile decides.

| # | Step | Note |
| --- | --- | --- |
| 1a | Profile a representative slice, including long contexts, loading and checkpoints | The "100 ms of fixed work per update" is a hypothesis until this exists |
| 1b | Apply the largest low-risk change the profile shows: one multiplication per table, fewer synchronising checks, or tensor-core arithmetic | Tensor-core arithmetic is exact only for the whole-number forward sum; gradients, attention and vocabulary scoring change slightly |
| 1c | Qualify at identical updates and examples: forward values and every gradient family, then a matched 30-minute run on the real screen | A faster run that sees more data is not a like-for-like check |
| 1d | Larger updates, as a separate change | Alters update count and schedule, so it is a recipe change |

New arithmetic enters the recipe identity and branches explicitly; old states do not
resume into it. Stop if the saving cannot repay its engineering cost in this campaign.

### Group 2: training data (about 3–5 GPU hours plus the largest build)

Today the target writes answers at 132 tokens per second, one conversation at a time. Its
internal values are stored at 41 KB per token. The owner's requirement: the project must
generate a fresh bank for any target model. A larger bank is authorised in measured
increments, each with a cost per point of acceptance, not from an extrapolation.

| # | Check | Note |
| --- | --- | --- |
| 2a | Data response on the bank we hold: stratified, source-disjoint subsets against the full bank, two seeds, equal updates; report distinct examples, label visits and scenario results | Shows whether more variety helps at this budget. It cannot price a tenfold bank |
| 2b | Store exact token IDs with template, thinking mode and target identity, and recompute the target's internal values in batches with the served 4-bit target; compare with stored captures | "Text only" is not enough: retokenising can change the sequence |
| 2c | Agreement of batch-recomputed values with one-at-a-time capture, and each row matched to its own request | Supervision already ends at the first token that differs from the target's own choice ([data](../docs/data.md)); a changed token also changes everything after it |
| 2d | Write answers for 8, 16 and 32 conversations at once: speed, and how much text still follows the target's own greedy path | Text that diverges is a different data recipe and is judged as one |
| 2e | Store the 2,560 values after fusion and before the trainable normalising step, at 16 bits, bound to the exact fusion table | Eight times smaller with no new rounding. Invalid once fusion trains |
| 2f | 8-bit values (FP8 against scaled whole numbers, chosen from the real distributions) against 16-bit, on the real screen | Only after 2e, and only within the allowed loss |
| 2g | One loop end to end, with representative contexts: write, rebuild, train, discard | The cost is unmeasured; writing and training in turn on one GPU could be far slower than stored data, and both must fit in 16 GB |

Dropped by the owner: re-proving that data is the limit on the unused 2,633 prompts, and
reusing public text. Later option: logging real use during serving.

### Group 3: which features matter (about 3–5 GPU hours)

Purpose, set by the owner: filter impactful features from wasteful ones. The current
recipe stays the main control.

1. **Free:** 0g and 0h above.
2. **One run: blank** (signs only, row scale fixed at the original average, fixed clipping
   bound, no gains, no midpoints) against the full-recipe control already held (49.70%).
   Within a point: the extras are jointly wasteful and blank becomes the base.
3. **Only if blank is clearly worse:** remove one feature at a time from the full recipe,
   learned row scale first, then learned clipping bound. If neither reproduces the loss,
   remove them as a pair.
4. **Four runs on the chosen base:** base, calibrated start (input-weighted fit), gradual
   precision, both. Gradual precision follows the published form with the scale kept,
   `S·F(W/S, t)` with `F = tanh(t·x) / tanh(t)` ([Tu et al.](https://arxiv.org/abs/2508.06974)),
   and ends with a period at hard signs.
5. **Second seed** for the best alternative and its control.

Add-ons, one at a time on the result, only where group 0 points to them: training weighted
toward real round states (the published method also uses rollout outcomes, so sampling
alone may give less); position weights derived from expected accepted length
([D-PACE](https://arxiv.org/abs/2605.18810), reported to help block drafters under greedy
checking); a second one-bit plane used only in training; per-slot rounding settings; an
offset on input rounding; a larger step for the clipping bound.

Flip counts are recorded as description only. Dropped: the output teacher, scales mixed
per token, the small correction beside each table, training only normalising weights.

### Group 4: vocabulary scoring and tables (about 2–3 GPU hours)

The vocabulary path is the largest drafter-side prize. The vocabulary table also supplies
the drafter's input tokens, so any replacement is judged by running the whole drafter, not
by rescoring saved outputs.

| # | Check | Note |
| --- | --- | --- |
| 4a | No training: each table converted to 2-bit and 1-bit, one at a time | Describes fragility; opens no training branch by itself |
| 4b | A shortlist or cheap first pass with exact rescoring, at real round states: frequent tokens, tokens in the prompt, a one-bit coarse pass, a conventional low-rank first pass at 4 or 16 bits | Errors on early slots cost more than overall agreement suggests; needs a few hundred independent cases to support a 99% claim |
| 4c | Timing of the proposed scoring shape in the runtime, with token selection merged, before any training | A smaller table is not automatically faster |
| 4d | The target's own table in place of our 341 MiB copy, whole drafter | File 517 to 176 MiB if tokens agree; costs about 0.17 ms unless 4b pays |
| 4e | Binary thin tables for vocabulary, against the conventional low-rank control | High risk: the cited paper keeps its vocabulary table at full precision |

Acceptance side: item 10 ([test list](speedup-test-list.md)), training the existing 4-bit
attention and fusion tables, is running in another thread and belongs here if it pays.

Cut from the release path: one-bit attention. Halving attention's time is worth about
1.3% of speed and risks more than that in acceptance. If it is ever reopened, value tables
stay at 4 bits.

### Group 5: reference check (about 30 minutes)

DSpark on the current target, and the public Qwen3.5-4B DFlash 2 checkpoint if it runs in
our runtime. The 2026-11-06 DFlash 2 result (73–79% of DFlash) was a three-prompt smoke
test with a precision workaround. One screen rejects a slow reference; it does not crown
the strongest.

### Group 6: serving work, bounded before training

Before training any format the runtime cannot serve, bound its serving cost with a timing
prototype. Runtime improvements that apply to both drafters are applied to both in the
final comparison.

### What the screen must hand over

The screen ends here. Training the candidate and release qualification are outside it
(owner, 2026-11-16). It is complete only when every line below has an answer backed by a
check above, so that the candidate can be built and trained without further experiments:

| Question | Answered by |
| --- | --- |
| Where the acceptance gap to Baseline sits, by scenario and by slot | Group 0 |
| The trainer's arithmetic and update size, and its measured speed | Group 1 |
| The data plan: how many conversations, which mix, how written, how stored, and the expected gain per added batch | Groups 0 and 2 |
| The feed-forward feature set and weight format | Group 3 |
| How training starts and proceeds: start state, calibration, precision schedule, step sizes, length, stopping rule | Groups 1 to 3 |
| The vocabulary scoring shape and each table's precision | Group 4 |
| The reference to beat | Group 5 |
| That every chosen piece can be served, and its measured cost per round | Group 6 |
| A predicted speed against Baseline for the assembled candidate, from the arithmetic above, with its uncertainty | All groups |
| The release test the candidate will later face, fixed before training | Group 0 and [benchmarks](../docs/benchmarks.md) |

### Decisions still with the owner

- The go per group, and who builds.
- Must the release be one-bit? If a trained 4-bit drafter met the speed goal, is that
  acceptable? This also decides whether the 4-bit trained control is worth a run.
- Three-valued weights: the reviewer would reopen them only after their serving cost is
  bounded; an extra 0.17–0.20 ms needs about two points just to break even.
- The allowed loss for storage and simplification changes (proposed: one point).

### Limits

- Estimates of trainer speed-up, loop cost and batched writing are unmeasured.
- Short runs can rank start-speed features too kindly; the second seed and the one-hour
  run are the guard.
- About twenty exploratory comparisons on one screen will throw up a false winner now and
  then; second seeds reduce that and release qualification, outside this screen, is the
  final check.
- The research sweep's four gap branches were not independently checked.

## Next

The owner answers the decisions above and gives the go per group. Group 0 and step 1a
need little or no new code and can go first once the GPU is free.

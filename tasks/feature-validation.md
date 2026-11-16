# Feature validation before the release candidate

## Goal

The next candidate should be the final candidate for a first release: a single drafter
that is faster than Baseline. This screen must tell us everything needed to construct that
candidate and train it once. It does not train the release candidate.

**What "faster" means** (owner, 2026-11-16): faster than the fixed, authenticated public
Baseline on every release scenario: ordinary greedy requests short and long, thinking-on
requests, sampled requests and agentic coding. Some flexibility is allowed: a few scenarios may be
up to about 3% slower if the rest are faster (owner, 2026-11-16; a guide, not an exact
figure). Thinking-on requests stay a required scenario. Working rule until the owner says
otherwise: at most two scenarios may fall short, each with the lower end of its speed range
at 0.97 or better; every other scenario's lower end is at 1.00 or better; no loss of
correctness or task completion anywhere. This is reported as the owner's policy beside the
existing stricter verdict in `benchmarks/scenarios.py`, which is left unchanged. The claim is a trained candidate against a fixed public
release; it does not isolate what one bit contributes.

**Where we stand** on native Linux: 55.2% acceptance on the 60-request screen against
Baseline's 60.6%, at 0.96–1.00 of Baseline's speed. Older measurements, to be redone:
thinking-on 0.88, sampled 0.95, agentic coding level.

**The arithmetic that judges every item** (from the native round profile):

- one point of acceptance is worth about 1.1% of speed on short prompts, less on long ones;
- 0.1 ms off a 7.7 ms round is worth about 1.3%;
- drafter-side prizes are small: vocabulary scoring 0.41 ms, token selection 0.11,
  attention tables 0.20, feed-forward 0.17, and most runtime gains help Baseline too.

**An independent forecast** (a reviewing model's estimates, not measurements): summing
realistic gains from every group gives about 0.98 of Baseline in the expected case and
1.02–1.04 in a favourable one; about 1.03 on short prompts, 1.00 on long ones, and about
0.95 on thinking-on requests even with +7 points, because the older figure implies a
12-point need there. The screen therefore allows the outcome "this candidate family has
not earned the final run".

## Authorized

The owner gave the go for the whole plan up front on 2026-11-16: groups 0 to 7 run
continuously without a stop for approval between them, and the owner checks in
periodically. Execution starts once the final independent hole review of this version is
folded in. Stop and ask only where a result changes the scope, or to report that the
candidate family has not earned the final run. Training the release candidate itself is
not authorized.

GPU and CPU-heavy work waits while another thread holds the GPU. Builds go to delegated
agents; proposed: Opus for product-code builds and Astra for reviews and closeouts, unless
the owner names others. Long runs follow the preflight and job-watch procedures.

## Done when

Every field of the candidate specification below is filled, each either a deliberate
"keep" or backed by a named comparison; the assembled recipe has been rehearsed and
measured on every release scenario; and a predicted margin with its uncertainty is
recorded. Or the checks show no plausible margin and that is reported. The decision record
and open limits go under `results/` before this file is deleted.

## State

Owner: Philip. Proposal by the lead agent; no check has run.

### Decisions already made

- **Fresh start.** Every screening run and the candidate start from the public drafter,
  not from the continued best state. Calibration and gradual precision only exist at the
  start of training, and a state whose step size has decayed to zero ranks short runs
  badly. The continued best state (55.2%) is kept as the yardstick a fresh recipe must
  overtake in the long rehearsal.
- **Frozen attention and fusion tables.** Training them gained +1.3 points
  ([result](../results/trained-attention-tables/result.md)) but forces the slower table
  format, needs 2.3 GiB more memory and breaks fused storage. Held as a reserve lever.
- **This target only.** The checks qualify Qwen3.5-4B. Generating a bank for another
  target needs its own capture adapter, outside this screen.
- Dropped by the owner: the output teacher, one-bit attention on the release path,
  re-proving the data limit on old prompts, public text. Three-valued and 2-bit weights
  are skipped for now.

### Rules for every check

- Only measured native speed decides; every result is first translated with the
  arithmetic above, crediting shared runtime gains to both drafters.
- One change per run against a matched control at equal updates and equal data.
- The real screen resolves about 1.1–1.4 points. The 120 DEV windows only nominate.
- Nothing is called harmless because no difference was seen. A simplification or a
  storage saving is adopted only if its loss is shown to be inside a budget: all such
  losses together may cost at most one point.
- Every adopted short-run finalist is repeated with a second data-order seed and checked
  on the 120-request panel and on the thinking-on, sampled and agentic slices. The panel is
  reused development data, not independent confirmation; the protected confirmation sets
  stay untouched. The single long run in group 7 is exempt from the second seed by the
  owner's choice.
- Two depth tracks are kept apart: a fixed depth of 3 for every diagnostic and comparison
  of acceptance, and a deployment depth per drafter and scenario for final speed.
- One-bit weights are evaluated as hard signs at every decision point.
- heldout24 is never used in this screen.

### Group 0: where the gap is, per scenario (about 2 GPU hours)

| # | Check | Use |
| --- | --- | --- |
| 0a | Fresh native measurement of candidate and Baseline on every release scenario, with draft depth calibrated for each drafter and scenario | The real gaps to close; replaces the older WSL figures |
| 0b | Candidate and Baseline at real four-slot round states: first wrong slot, first-choice and top-16 recall, round starts against random positions | Where the deficit sits |
| 0c | Training labels by scenario, thinking setting and context length, counted in conversations, labels and sources, beside each scenario's gap | The release agentic tasks start at 11.5–19.9K tokens; our tool sessions are 1.6–6.8K |
| 0d | Candidate and Baseline on prompts from the training set | Seen against unseen prompts |
| 0e | Clipping frequency and spike positions; input ranges per layer split by anchor and slot | Whether rare positions set the clipping bound; whether per-slot rounding is worth a run |
| 0f | Gains set to 1 and midpoints to 0 in the trained candidate; calibrated settings on an untrained drafter; feed-forward inputs unrounded | Dependence only. These do not rule a feature in or out by themselves |
| 0g | How the runtime selects tokens (it takes the top ten and keeps the first) and what the CPU-side 0.47 ms is spent on | Whether a plain top-choice path or less waiting is available, for both drafters |

### Group 1: training speed (about 1.5 GPU hours plus a small build)

| # | Step | Note |
| --- | --- | --- |
| 1a | Profile a representative slice, including long contexts, loading and checkpoints | Expected gain is between 1.2–1.8 and 2–4 times; the profile decides |
| 1b | The largest low-risk change: one multiplication per feed-forward table, fewer synchronising checks, or tensor-core arithmetic | Exact only for the whole-number forward sum |
| 1c | Qualify at identical updates: forward values and every gradient family, then a matched run on the real screen | Repeated later on the selected feature set |
| 1d | Larger updates as a separate recipe change | Changes update count and schedule |

New arithmetic enters the recipe identity and branches explicitly.

### Group 2: training data (about 5–6 GPU hours plus the largest build)

Sizes are counted in conversations, captured tokens, supervised labels and independent
sources, which are different things. At today's 132 tokens per second, 100,000
conversations would take about 108 GPU hours to write and about 580 GB even in the
smallest storage, so the bank grows in measured steps.

| # | Check | Note |
| --- | --- | --- |
| 2a | Store exact token IDs with template, thinking mode and target identity; recompute internal values in batches with the served 4-bit target | Retokenising text can change the sequence |
| 2b | Agreement with one-at-a-time capture by scenario and context length; each row matched to its request; counts of usable windows by length | A changed token changes everything after it; shortened windows may remove the hard examples |
| 2c | Write 8, 16 and 32 answers at once: speed, and how much text still follows the target's own greedy path | Diverged text is a different data recipe |
| 2d | Fused storage: the 2,560 values after fusion and before normalising, at 16 bits, bound to the fusion table and the feature identity. Qualified by a matched training pair, because it adds one rounding step | Eight times smaller; valid because fusion stays frozen. Token streams are kept so values can be rebuilt |
| 2e | 8-bit storage against 16-bit, only after 2d and inside the loss budget | |
| 2f | One loop end to end, cold: load, write, rebuild, train, checkpoint, discard, with the longest contexts and within 16 GB | Cost unmeasured |
| 2g | One representative new batch: long agentic sessions and thinking-on requests of the release kind. Record usable labels, capture cost and disk per useful label | Its learning value is measured in group 7. A source for long projects must be found; the current one is exhausted |

### Group 3: which features matter (about 6–7 GPU hours)

Purpose, set by the owner: filter impactful features from wasteful ones, all from a fresh
start.

1. **One run: blank** (signs only, fixed row scale, fixed clipping bound, no gains, no
   midpoints) against the full recipe. Blank becomes the base only if its loss is shown to
   be inside the budget. The extras cost almost nothing at serving, so they are not given
   up for a result that merely looks level.
2. **If blank is worse:** remove learned row scale, then learned clipping bound, from the
   full recipe; as a pair if neither explains the loss.
3. **Four runs on the base:** base, calibrated start, gradual precision, both. Gradual
   precision is written out first: the forward and backward rule on our stored sign
   values, `S·F(W/S, t)` with `F = tanh(t·x) / tanh(t)`
   ([Tu et al.](https://arxiv.org/abs/2508.06974)), checked to reproduce the public drafter
   at its soft start, and ending with a period at hard signs.
4. **Second seed** for the best alternative and its control.

Add-ons, one at a time, only where group 0 points to them: training weighted toward real
round states, position weights from expected accepted length, a training-only second
plane, per-slot rounding settings, an input-rounding offset, a larger step for the
clipping bound. Each needs its settings, saved-state and export support built before it
can be chosen.

### Group 4: vocabulary scoring (about 2–3 GPU hours)

The vocabulary table also supplies the drafter's input tokens, so a replacement is judged
by running the whole drafter, on both drafters.

| # | Check | Note |
| --- | --- | --- |
| 4a | A shortlist or cheap first pass with exact rescoring: frequent tokens, tokens in the prompt, a one-bit coarse pass, an ordinary low-rank first pass | Judged by accepted tokens lost per round, measured directly, by scenario and slot; a 1% miss rate can cost about 2% of speed |
| 4b | Timing of the scoring shape with token selection merged, before any training | |
| 4c | The target's own table in place of our 341 MiB copy | File 517 to 176 MiB if tokens agree |
| 4d | Binary thin tables against the low-rank control | High risk |
| 4e | No training: each table converted to 2-bit and 1-bit | Description only |

A shortlist fitted now is rechecked on the trained body from 7b.

### Group 5: the reference (about 1 GPU hour)

Pin Baseline's identity and a cutoff date. Check DSpark, and the public Qwen3.5-4B
DFlash 2 checkpoint if its origin can be confirmed, each in its intended settings. Later
public releases are separate comparisons.

### Group 6: serving and integration (about 1 GPU hour plus builds)

- Bound the serving cost of any new format with a timing prototype before training it.
- Every chosen piece gets its settings, saved-state, export and runtime support through
  the maintained entry points; nothing unsupported reaches the handover.
- Memory check of the assembled recipe at the longest context.

### Group 7: the assembled rehearsal, light version (owner's choice)

| # | Step | Note |
| --- | --- | --- |
| 7a | Integration pilot, about 10 minutes: train, interrupt, restore, continue, export, load and serve with the chosen vocabulary path and depths; exact outputs checked; memory at the longest context | Catches shape, memory, resume, export and dispatch failures |
| 7b | One long run of the assembled recipe from a fresh start, two to three training hours, on everything we hold including the new batch from 2g | Measured against the continued best state (55.2%) as yardstick. Short runs can mislead: the data continuation gained 0.18 points at 30 minutes and 6.1 at three hours |
| 7c | The learning curve from 7b | The run's own schedule is frozen beforehand; the curve recommends length, schedule and stopping rule for the release run, with its uncertainty, and does not prove an optimum |
| 7d | The 7b result on all release scenarios with fresh paired timing, both drafters at their best depth per scenario | The selection, against the definition of "faster" above |

Given up by choosing the light version, and stated in the hand-over: no matched control
at this length, no second seed of the long run, and no separate measurement of what the
new data batch is worth. The long run shows that the assembled recipe works end to end
and where it lands; it does not attribute the gain between features and data.

If the long run clears every scenario, the owner may promote it instead of training again.

### What the screen must hand over: the candidate specification

Every line is either "keep" with its reason or a named comparison.

| Field | Source |
| --- | --- |
| Start state and transition | Decided: fresh from the public drafter |
| Weight format and feature set | Group 3 |
| Initial values: calibration, sign values, scales | Group 3 |
| Precision schedule and hard-sign period | Groups 3 and 7 |
| Step size per parameter family, warm-up and decay | Keep, except those group 3 changes |
| Optimizer settings and gradient clipping | Keep; clipping statistics logged in 7b |
| Update size, block size (4 slots, stated explicitly: the default is 16) | Keep, unless 1d changes it |
| Position weights | Keep, unless a group 3 add-on wins |
| Which tables train | Decided: none |
| Data: bank size, mix by scenario, order, passes | Groups 0, 2 and 7; bank size beyond what we hold is an extrapolation in the light version |
| Data storage and rebuild route | Group 2 |
| Trainer arithmetic | Group 1 |
| Training length, stopping rule, checkpoint choice | Group 7 |
| Export format and runtime kernel of every tensor | Group 6, traced in 7a |
| Vocabulary scoring path | Groups 4 and 7 |
| Draft depth per scenario | Groups 0 and 7 |
| The reference and the release test | Group 5 and [benchmarks](../docs/benchmarks.md) |
| Predicted margin per scenario, with uncertainty | Group 7, one run and one seed |

### Time

Rough expectations, not budgets: about 17–26 GPU hours in total, about a day and a half
of wall time running back to back with builds overlapping. Groups should finish faster
than this where they can, and nothing is padded to fill an allowance. A group that needs
materially more than expected is reported, not silently extended.

### Decisions still with the owner

- Who builds, if not the proposed models.

Three-valued weights are not reopened for now (owner, 2026-11-16).

### Limits

- Estimates of trainer speed-up, loop cost and batched writing are unmeasured.
- Two to three hours is still shorter than a final run; the learning curve in 7c is the
  guard.
- Many comparisons reuse the same development prompts; second seeds and the panel reduce
  the risk of a false winner, and release qualification, outside this screen, is the last
  check.
- The research sweep's gap branches and the forecast above are estimates.

### Decision rules for unattended running

Mechanical rules, so the run does not stop for judgement calls. Where a rule cannot be
applied, the original setting is kept and the gap is reported.

| Judgement in the plan | Rule |
| --- | --- |
| References | Baseline, target, public initializer, vocabulary, the continued yardstick state and the runtime are pinned by hash before group 0. Other public drafters are reported separately and never move the finish line |
| "Inside the loss budget" | Loss is acceptance points at depth 3. A simplification or storage change is adopted only when the one-sided 95% bound on its loss, against the otherwise identical recipe, is within the budget, and the combined bound for all adopted ones is within one point. Inconclusive means keep the original |
| "Blank becomes the base" | Only under the rule above. Otherwise the full recipe stays, and removals are run only if they change an implementation decision |
| "Largest low-risk change" (trainer) | The largest measured end-to-end gain among changes that pass the value, gradient and resume checks. None passing means keep the current trainer |
| Larger updates | One predeclared grouping change, at equal labels and data, with the schedule restated |
| Capture agreement | Exact row, request and token identities and valid labels are required. Any path whose values are not bit-identical also needs a matched training and serving check |
| Batched writing that diverges | Not counted as a faster equivalent. Fall back to lower concurrency or one at a time |
| 8-bit storage | Only if 16-bit fused storage leaves a measured storage or loading problem |
| "A representative new batch" | Quotas fixed before writing: long agentic sessions and thinking-on requests both required, with caps per source and project, chosen by origin and coverage, never by DEV results, and with the confirmation and agentic-project exclusions applied |
| No qualified long-project source | Keep the admitted data, mark long agentic coverage unresolved, continue |
| Add-ons "where group 0 points" | Round-state weighting: a paired gap between round-start and random positions. Position weights: a deficit that grows with slot, on enough reached slots. Second training plane: a fixed removal schedule with hard-sign updates after it. Per-slot rounding: repeatable range differences between slots. Input offset: a repeatable one-sided rounding error. Larger clipping step: logged clipping updates that lag while gradients survive the shared limit. None met means skip |
| Shortlist | Built from TRAIN or calibration inputs only; adopted on whole-round acceptance and clean end-to-end timing with the same change applied to Baseline |
| The target's own table (4c) | Needs matching token IDs, sizes and mask and end-of-turn handling, then whole-drafter acceptance |
| A public comparator's origin | Needs an author or repository chain, a fixed revision or hash, target compatibility and stated settings. Otherwise recorded as unverified and skipped |
| "No plausible margin" | A measured dead end or a defensible upper bound below the success rule. A weak short screen alone is "not demonstrated" |
| "Materially more than expected" | Reported at 1.5 times the expectation; work continues while its scientific stop rule is unmet |
| The long run's early stop | Only for correctness, data, non-finite, state or memory failures. Not for a weak first half hour. Its length is not extended after seeing results |
| Checkpoint choice | One checkpoint for all scenarios: the final one, unless a rule fixed beforehand on development data picks an earlier one |
| Promotion after the long run | Recommended to the owner only; never published or confirmed on protected data by this screen |

### What is being kept, written out

"Keep" in the specification means these values, set explicitly in every intent, never left
to product defaults: sign step 0.004; scale, midpoint, gain, clip and norm steps 0.00001;
row latent scale times 3; about 2% warm-up and cosine decay to zero at the declared end;
fused AdamW with no weight decay, betas 0.9 and 0.999, epsilon 1e-8; one shared gradient
norm limit of 1.0; blocks `{"4": 1.0}`; grouping of at most 32 windows with a 3 GiB
activation budget; position weights `decay`; `frozen_format: baseline-k`;
`trained_tables` off. Normalising weights still train. This supersedes the older note in
the [test list](speedup-test-list.md) that acceptance work continues from the best state.

### Execution order

1. Pin references and write the explicit control intent, data manifests and guards.
2. Start builds in parallel: per-drafter and per-scenario depth with compatible reports;
   round-state and clipping diagnostics; data replay, batched capture and fused storage;
   feature controls, calibration, gradual precision and training telemetry; and the search
   for a long-project data source.
3. Qualify the benchmark and sandbox path on this host and the sampler for this runtime;
   establish native controls for target, Baseline and the current candidate on every
   scenario; run group 0 at the fixed depth, then at calibrated depths once supported.
4. Profile the trainer on short and long inputs; qualify the best supported change; decide
   grouping; freeze screening update counts, schedule and data order.
5. A small sample of the new data first; qualify token replay, batched writing and
   recomputed values separately, against current serving values; fused storage; the cold
   loop on the longest samples; then write the batch and admit the final TRAIN set.
6. Time the vocabulary scoring shape and settle the table choice before the feature runs.
7. Witness calibration and gradual precision; run blank, any removals and the four-run
   square with matched controls; then only the add-ons that meet their rule.
8. Second seeds; the panel and the three scenario slices for short finalists.
9. Requalify trainer arithmetic and storage on the assembled features; one short matched
   pair, assembled against the same recipe without its simplifications, for the combined
   loss.
10. Reference checks on other public drafters, without delaying the rest.
11. Freeze the assembled recipe; the integration pilot with an exact resume check
    (uninterrupted against interrupted, including the next update), export, dispatch and
    the longest context.
12. The single long run with a fixed schedule: exports and diagnostics every 30 training
    minutes; per-family gradient norms, the shared clipping factor, sign and rounding
    statistics, memory and throughput logged throughout; full states kept at the start,
    around any precision switch, at one and two hours and at the end.
13. After training, every saved readout on the screen, the panel and the three scenario
    slices; one checkpoint chosen by rule; its shortlist and deployment depths frozen.
14. Fresh paired timing on every release scenario; the report, with the single-run limits
    and any claim not earned; evidence, retention and closeout.

## Next

The final hole review is folded in above. Follow the execution order from step 1.

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

Owner: Philip. Executing since 2026-11-16 17:00 (lead agent; plan text below unchanged from
`7f621ed9`). Campaign folder: `/home/philip/BitDraft/artifacts/feature-validation-01/`.

Progress:
- Step 1 done: references pinned in `pins/references.json` (target Q4KM `38742092…`, Baseline
  `83586a63…`, initializer `18cefa06…`/`2f852a41…`, vocabulary `a68f55d9…`, yardstick
  `769e184e…`/`5749a05f…`, runtime-v8 `d613abbe…`). Release target for this screen is Q4KM only.
  Control intent generator `scripts/intents.py` writes every kept value explicitly; the full
  TRAIN set (`mixture.json`, 8,633 records) is the default data, the 6K corpus only to match
  earlier runs. Warm-up is `round(0.02 × updates)`. The control admits on CPU.
- Step 2 done: builds A (owner-policy report, per-arm depth), B (round-state, label, clipping and
  dependence diagnostics, Baseline evaluation model), C (token replay, replay agreement, batched
  writing, fused storage), D (feature switches, calibrated start, gradual precision, telemetry)
  and G (long-project TRAIN source, thinking-on selection) are merged into this branch with the
  full gate (213 tests). Build E (exact faster trainer arithmetic) is running. Briefs in `briefs/`.
- Step 3 done at depth 3 (notes `group0/0a-0g-notes.md`): sampler qualified for runtime-v8;
  target, Baseline and yardstick references registered for the thinking, sampled and agentic
  suites and slices (`step3/`). Yardstick against Baseline: thinking-on 0.915 (0.906, 0.924),
  acceptance 53.4% vs 66.6%; sampled 0.974, 43.5% vs 50.0%; agentic task rate 1.027 (1.019,
  1.037), 65.5% vs 63.4%, completion 37/60 for every arm; greedy panel 0.965 (item 9). Greedy depth
  calibration: both drafters best at 3. The deficit sits in thinking-on and ordinary greedy.
  0g answered from the runtime source and round profile: drafts come from a top-10 sampler whose
  first entry is the argmax; a plain argmax and fewer per-slot synchronisations are generic gains.
  The agentic sandbox's process limit was raised to 4096 on this branch (1024 was reachable).
  Group 0b–0f wait for build B; the greedy short/long split for build A.
- Step 4: profile in `step4/` (matmuls 55% of training GPU time as FP32 SIMT; one product per
  feed-forward table and no in-update synchronisation estimated 1.13–1.31× per update, measured
  under CPU load; global TF32 1.44–1.58× but inexact). Build E implements the exact part. Screening
  runs frozen at 8,209 updates, warm-up 164, cosine to zero, full TRAIN, seed 8101 (8102 for second
  seeds). Current-trainer control `arms/s4-c0-current` (source `3cd2db82…`) is training.
- Step 4 qualification of build E passed (`builds/trainer-speed/run-01`): forward values
  bit-identical, every gradient family within 1e-5, resume exact, TF32 feed-forward products
  bit-identical. Against this morning's trainer about 1.18–1.40× per update; build E's exact
  fixes also sped up the reference path. Matched run `s4-c0-fast` (`fast-exact-tf32ff`) against
  `s4-c0-current`: screen 48.72% vs 48.95%, −0.23 (−0.91, +0.42), inside the loss budget; update
  time 1,520 vs 1,767 s for the same 8,209 updates and 2,493,623 labels (1.16×; 1.14× with data
  loading). Step 9 then requalified it on the assembled recipe: calibrated + fast arithmetic vs
  calibrated: 56.15% vs 56.56%, −0.41 (−1.45, +0.58). The lower end is below −1.0, so under the
  step 4 preflight's rule (counted against the one-point budget) it is not adopted: the long run
  and the recommended recipe keep `reference` arithmetic (likely noise; the arithmetic passed
  exact value, gradient and resume checks; cost: about 14% fewer updates per training hour).
- Fused storage (step 9, rerun after the fix): calibrated on fused vs eight-tap storage: 56.36%
  vs 56.56%, −0.20 (−1.11, +0.72): lower end below −1.0, inconclusive, original storage kept;
  fused storage is not qualified, also not for the release run.
- Control `s4-c0-current`: screen 48.95% (Baseline 60.6%), all outputs exact, 35 training minutes.
- Step 7 (`group3/preflight.md`, thresholds fixed): gradual start checks passed for both intents.
  Update-0 probes: calibration moves the untrained drafter little (DEV first-slot top-1 5.8% vs
  5.0%, screen 2.7% vs 2.1%), but one warm-up update then collapses the calibrated drafter (DEV
  loss 7.4 → 12.9, screen 0%) while the uncalibrated one improves (7.1, 4.7%): a calibration bug,
  returned to build D. Both calibrated arms are skipped until it is fixed.
- Incident: the host froze hard at 19:15, 30 s after build B's first diagnostics step hit a GPU
  out-of-memory; no kernel message; the owner power-cycled it at 23:05 (about four GPU hours
  lost). Every queued job now runs in a systemd scope capped at 24 GB RAM; the diagnostics are
  held behind skip markers and returned to build B for an explicit memory budget.
- Group 3 results so far (screen, paired against the control, 60 requests):
  - Blank: 48.38% vs 48.95%, −0.57 points (−1.24, +0.09). Its interval's lower end is below −1.0,
    so blank is not admitted; the full recipe stays the base. Removals (fixed scale, fixed clip)
    would change no implementation decision (the extras cost almost nothing at serving): skipped.
  - Gradual precision: 49.11% vs 48.95%, +0.16 (−0.46, +0.83): not adopted (tie keeps control).
  - Calibrated probe (fixed): untrained DEV first-slot top-1 17.5% vs 5%, screen 9.0% vs 2.1%;
    after one update 52.5% / 21.7% vs 9% / 4.7%.
  - Calibrated start (`g3v2-calibrated`, source `4bb8231d…`): 56.56% vs 48.95%, +7.61 points
    (+6.65, +8.60); DEV prefix survival 57.5% vs 55.8%, thinking-on windows 60% vs 50%, training
    loss 1.78 vs 2.31 at update 8,000. Second seed 8102: 56.42% vs 49.20%, +7.21 (+6.10, +8.41);
    pooled over both seeds (paired bootstrap over both screens' rows): +7.41 (+6.53, +8.30).
    Adopted against its control. Level 2 (`level2/g3v2-calibrated/`, candidate-only against
    the saved runtime-v8 references, all outputs exact, agentic completion 5/12 on both):
    acceptance vs Baseline 47.7% vs 53.4% (120 requests; yardstick 46.2%), prefix 70.6% vs 74.1%,
    live 81.6% vs 70.1%, thinking-on slice 56.3% vs 66.2%, sampled slice 45.7% vs 51.5%, agentic
    slice 77.2% vs 61.8%; descriptive speed vs saved Baseline timing 0.962 (120 requests), 1.011
    (prefix), 0.952 (thinking), 0.983 (sampled). The Level 2 report returns `level3_required`
    (acceptance below Baseline beyond the interval in five scenarios; direction disagreement in
    three). Level 3 completed (candidate-only, saved references; outputs exact; agentic
    completion 37/60 for both): thinking-on suite 57.1% vs 66.6% (yardstick 53.4%), descriptive
    speed 0.944; sampled suite 44.2% vs 50.0%, 0.982; agentic suite 79.4% vs 63.4%, task rate
    1.104 (descriptive). The Level 3 report says `not-established` (no paired Baseline timing in
    candidate-only runs, as expected); the plan's precondition for adoption is met and calibrated
    start stays adopted on its two-seed result against the control. Step 14 measures paired
    timing on the long run's checkpoint. Calibration becomes the base for the remaining
    add-ons: clip step and round-state weighting are rerun on top of it (`scripts/queue4b.sh`).
  - Calibrated + gradual vs calibrated alone: 56.17% vs 56.56%, −0.39 (−1.16, +0.38): not adopted.
    The four-run square ends with calibration alone.
  - Clip step 1e-3 on calibrated vs calibrated: 56.35% vs 56.56%, −0.21 (−1.25, +0.77): not
    adopted. Calibration already removes clipping (final clipped input shares 0.0–0.1% in every
    projection, against up to 57.5% in the uncalibrated control's last down projection).
  - Per-slot rounding (triggered on the uncalibrated yardstick, needs runtime support first):
    ceiling check on the calibrated drafter (`runs/round-diagnostics/calibrated-dev/`): removing
    input rounding entirely leaves DEV unchanged (prefix survival through 3: 60.0% both ways;
    first-slot top-1 88.3% vs 89.2%, one window), as on the yardstick, so finer per-slot rounding
    cannot help: skipped. Gains 1 / midpoints 0 also stay within one window there.
  - Round-state weighting (triggered by 0b; build F merged): the map pilot measured 323 simulated
    proposals per second (120 rows, 55,306 proposals; round starts 36.7% of windows), so the full
    TRAIN map would take 3.4 GPU hours and one run's restricted map about 2.6; the build's
    13–25-minute forecast was 8–15× too low. Over the preflight's 60-minute kill limit, and the
    measured excess round-start penalty is 0.07 accepted tokens per round: skipped; recorded as
    triggered but not demonstrated.
  - Group 0 (`group0/0a-0g-notes.md`): the candidate's deficit is concentrated at real round starts
    (4.9 points there vs 2.3 over all positions); on TRAIN prompts it beats Baseline by 10.7 points
    (generalisation, not capacity, limits it); learned clip bounds stay near 1.0 while late-layer
    down inputs reach 5–27× the bound; gains and midpoints barely matter. Add-ons triggered:
    round-state weighting (build F running), larger clipping step (`g3-clipstep` queued), per-slot
    rounding (deferred: needs runtime support; decided after the clipping arms).
  - Calibration bug fixed by build D (clip bounds now minimise the tables' output error; row-fit
    trust limit); the calibrated probe and arms rerun on source `4bb8231d…`, whose default path is
    pinned bit-identical to the control's.
- Step 5 checks: token replay at 32 tokens per batch is bit-identical to the stored captures
  (40 TRAIN records, 26,736 labels; at 512 values differ); batched writing at 8/16/32 slots is
  2.2–2.5× faster but only 8 of 32 answers match one-at-a-time writing, so new data is written one
  request at a time. Build A's per-arm depth qualified on the GPU (`OVERALL PASS`).
- Step 5: long-session pilot passed (11/12 sessions completed, initial contexts 12–20K tokens, all
  30 records validate; session 446 labels and 0.74 GB per record, thinking-on 1,502 labels and
  69 MB). Batch sized by the pilot rule at a 120 GB ceiling (disk: 182 GB free before retirement):
  138 long sessions and 246 thinking-on requests, about 1.5 GPU hours and 430K labels.
  Closed, unselected short-run states of this screen retired through the store (receipts beside
  each; `logs/state-retirement-20261117.json`), freeing 49 GB; exports, journals and screens kept.
- Builds merged: H (per-arm speculation mode; public drafters verified: DSpark
  `shanjiaz/qwen3_5_4b_perfectblend_regen_dspark@3842177e`, DFlash2 `nchapman/Qwen3.5-4B-DFlash2@8d8bb628`),
  V (vocabulary scoring measurements) and C's fused-calibration fix (calibration's sample loader
  ignored the corpus's fusion identity and refused fused records).
- Fused storage: full TRAIN converted (8,633 arrays, max relative L2 2.1e-4, 55.2 GB, 18 min);
  the first matched run failed at initialisation on that bug; the rerun's verdict is above.
- Group 5 (`logs/group5.log`, 60-request screen, runtime-v8, the authors' stated settings, BF16
  GGUF converted without loss, outputs exact for both): DSpark at depth 8: 1.71 accepted tokens per
  round, 0.71× target-only and 0.50× Baseline (saved timing, descriptive); DFlash 2 at depth 15:
  2.07 per round, 0.63× target-only, 0.45× Baseline. The runner's "acceptance below 15%" kill flag
  fired on DFlash 2 only because per-proposal acceptance is low at depth 15; the screen completed.
  Reported separately; neither moves the finish line. A Q4 conversion was not tried: the authors
  publish none (decision rule: stated settings only).
- Group 4 (`logs/group4.log`, build V; screen round states, candidate = yardstick): scoring +
  top-10 selection 451 µs per round (ours) and 621 µs (Baseline's table); merged argmax is slower
  (483 / 658), correcting the 0g note. Frequent-TRAIN-token shortlists with the request's tokens
  lose almost nothing (64K: 0 accepted tokens per round; 32K: −0.006) and save 0.33–0.40 ms for us
  but 0.46–0.55 ms for Baseline: applied to both, as the rule requires, our ratio falls by about
  1.5%. One-bit or low-rank first passes with exact rescoring are slower than full scoring for our
  table. 4c: the target's Q6_K table reads 497 instead of 341 MiB (+0.17 ms per round). 4e
  (description): 2-bit / 1-bit scoring tables cost 0.022 / 0.056 accepted tokens per round, a 1-bit
  table for the whole drafter 0.20. Decision: keep the current Q4_K table and selection; 4d not
  built.
- New batch: 159 long-project sessions generated (157 completed tasks, 688 turns) in 15 minutes.
  The first selection was refused because its worst-case forecast (every thinking-on answer at
  its 2,048-token cap) exceeded the 120 GB ceiling; rerun at 126 GB with unchanged quotas:
  138 sessions (58,634 span positions) and 246 thinking-on requests, worst case 124.1 GB.
- New batch captured (384 records, 112 GB, 70 minutes); final TRAIN mixture `train-final-01`
  (`f756aea5…`, six components, 9,017 sequences) admitted.
- Step 11 passed (after two refusals before any training: real disk shortage, then a declared
  capacity too low in my generator; the unqualified fused corpus, 55.8 GB, was retired through
  `tools.retention` with receipt `retirement/fused-train-receipt.json`): 400 uninterrupted
  updates against 200 + resume to 400 gave equal losses before and after the interruption and a
  byte-identical final export (`73b5b604…`), which served on the screen with exact outputs
  (48.75%); peak 13.5 GB allocated, 14.2 GB reserved; 0.266 s per update on the final mixture.
- Step 12 done (08:32–12:56; 42,000 updates, 15,856 s update clock including checkpoints; recipe
  `5d5c7d0e…`; no failures). Final export `ae6569f6…` screens **57.64%** (exact outputs), against
  the yardstick's 55.2%, the 35-minute calibrated run's 56.56% and Baseline's 60.6%. Checkpoints
  kept every 7,000 updates as full states; the run wrote only the final export, so the five
  earlier states are re-exported through the store's export path (the final state re-exports
  byte-identically to `ae6569f6…`). A build agent's full-size CPU check evicted the run's data
  from the page cache around 11:40 (data wait 0.10 → 0.29 s per update) and was stopped.
- Step 13 running: every checkpoint on the screen, the 120-request panel and the three slices
  (`step13/`); then step 14a (greedy depth calibration of the final drafter; sampled and agentic
  slices at depths 2, 3, 4 for both drafters), then step 14 (fresh three-arm suites at the chosen
  depths), then the owner's additions (retests, build W, round-state, per-slot).

Decided by the owner 2026-11-17 10:55: (1) add the vocabulary shortlist as a native feature
(build W: runtime edit, shortlist stored with the drafter, applicable to Baseline for the fair
comparison); every claim about it must separate what is native to our drafter from what
Baseline would also gain. (2) Measure quantized vocabulary tables (Q3_K, Q2_K) on the real
runtime (build W). (3) A second seed each for fast arithmetic and fused storage on the calibrated
recipe, judged on the pooled two-seed bound (`group3/retest-preflight.md`), after step 14.
(4) Run round-state weighting (build F2: a fast map from grouped forward passes, target under 75
GPU minutes) and per-slot rounding (build P: anchor and mask clip bounds in training, export and
runtime, with a timing prototype first), each as one calibrated arm against `g3v2-calibrated`
with the group 3 thresholds, after the retests.

Decided by the owner 2026-11-16 19:15: `greedy-long` is judged on the supplementary interval
that pools category strata with fewer than two source groups (the maintained rule cannot give it
an interval: one conversation and one structured request among its 24 long requests). Only for
`greedy-long`; the other four scenarios keep the maintained rule, and the maintained
`greedy-long` result ("not demonstrated") is still reported beside it.

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

### Group 0: where the gap is, per scenario

| # | Check | Use |
| --- | --- | --- |
| 0a | Fresh native measurement of candidate and Baseline on every release scenario, with draft depth calibrated for each drafter and scenario | The real gaps to close; replaces the older WSL figures |
| 0b | Candidate and Baseline at real four-slot round states: first wrong slot, first-choice and top-16 recall, round starts against random positions | Where the deficit sits |
| 0c | Training labels by scenario, thinking setting and context length, counted in conversations, labels and sources, beside each scenario's gap | The release agentic tasks start at 11.5–19.9K tokens; our tool sessions are 1.6–6.8K |
| 0d | Candidate and Baseline on prompts from the training set | Seen against unseen prompts |
| 0e | Clipping frequency and spike positions; input ranges per layer split by anchor and slot | Whether rare positions set the clipping bound; whether per-slot rounding is worth a run |
| 0f | Gains set to 1 and midpoints to 0 in the trained candidate; feed-forward inputs unrounded | Dependence only. These do not rule a feature in or out by themselves. The calibration check on an untrained drafter moves to group 3, after calibration is built |
| 0g | How the runtime selects tokens (it takes the top ten and keeps the first) and what the CPU-side 0.47 ms is spent on | Whether a plain top-choice path or less waiting is available, for both drafters |

### Group 1: training speed

| # | Step | Note |
| --- | --- | --- |
| 1a | Profile a representative slice, including long contexts, loading and checkpoints | Expected gain is between 1.2–1.8 and 2–4 times; the profile decides |
| 1b | The largest low-risk change: one multiplication per feed-forward table, fewer synchronising checks, or tensor-core arithmetic | Exact only for the whole-number forward sum |
| 1c | Qualify at identical updates: forward values and every gradient family, then a matched run on the real screen | Repeated later on the selected feature set |
| 1d | Larger updates as a separate recipe change | Changes update count and schedule |

New arithmetic enters the recipe identity and branches explicitly.

### Group 2: training data

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

### Group 3: which features matter

Purpose, set by the owner: filter impactful features from wasteful ones, all from a fresh
start.

1. **One run: blank** (signs only, fixed row scale, fixed clipping bound, no gains, no
   midpoints) against the full recipe. Blank becomes the base only if its loss is shown to
   be inside the budget. The extras cost almost nothing at serving, so they are not given
   up for a result that merely looks level.
2. **If blank is worse:** remove learned row scale, then learned clipping bound, from the
   full recipe; as a pair if neither explains the loss.
3. **Four runs on the base:** base, calibrated start, gradual precision, both. Gradual
   precision is written out first: the forward and backward rule, `S·F(W/S, t)` with
   `F = tanh(t·x) / tanh(t)` ([Tu et al.](https://arxiv.org/abs/2508.06974)), on the
   original weights; as `t` goes to zero `F` is the identity, and the schedule starts from
   a small positive `t`, not zero. Our stored sign values are rescaled and clamped and have lost the
   weight sizes, so the run keeps the source weights for the soft phase. The start check
   is per table under matched inputs: each feed-forward table reproduces the public
   table's output at the soft start. Inputs stay rounded, so whole-drafter equality with
   the public drafter is not expected. A build that fails this check is rejected. The run
   ends with a period at hard signs.
4. **Second seed** for the best alternative and its control.

Add-ons, one at a time, only where group 0 points to them: training weighted toward real
round states, position weights from expected accepted length, a training-only second
plane, per-slot rounding settings, an input-rounding offset, a larger step for the
clipping bound. Each needs its settings, saved-state and export support built before it
can be chosen.

### Group 4: vocabulary scoring

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

### Group 5: the reference

Pin Baseline's identity and a cutoff date. Check DSpark, and the public Qwen3.5-4B
DFlash 2 checkpoint if its origin can be confirmed, each in its intended settings. Later
public releases are separate comparisons.

### Group 6: serving and integration

- Bound the serving cost of any new format with a timing prototype before training it.
- Every chosen piece gets its settings, saved-state, export and runtime support through
  the maintained entry points; nothing unsupported reaches the handover.
- Memory check of the assembled recipe at the longest context.

### Group 7: the assembled rehearsal, light version (owner's choice)

| # | Step | Note |
| --- | --- | --- |
| 7a | Integration pilot, about 10 minutes: train, interrupt, restore, continue, export, load and serve with the chosen vocabulary path and depths; exact outputs checked; memory at the longest context | Catches shape, memory, resume, export and dispatch failures |
| 7b | One long run of the assembled recipe from a fresh start, two to three training hours, on everything we hold including the new batch from 2g, with labels seen and passes over the data recorded beside training hours | Measured against the continued best state (55.2%) as yardstick. Short runs can mislead: the data continuation gained 0.18 points at 30 minutes and 6.1 at three hours |
| 7c | The learning curve from 7b | The run's own schedule is frozen beforehand; the curve recommends length, schedule and stopping rule for the release run, with its uncertainty, and does not prove an optimum |
| 7d | The 7b result on all release scenarios with fresh paired timing, both drafters at their best depth per scenario | The selection, against the definition of "faster" above |

Given up by choosing the light version, and stated in the hand-over: no matched control
at this length, no second seed of the long run, and no separate measurement of what the
new data batch is worth. The long run shows that the assembled recipe works end to end
and where it lands; it does not attribute the gain between features and data.

If the long run clears every scenario, the owner may promote it instead of training again.
A proposed release recipe that differs from the 7b run in any way is reported separately
from 7b's measured margin; its predicted gain is an extrapolation.

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

Local card only (owner, 2026-11-16). Expect about 24–29 GPU hours and 28–34 wall hours,
including likely conditional comparisons, overlapping builds, required gates and closeout;
these are expectations, not budgets. A step that needs more than 1.5 times its figure is
reported, not silently extended.

Measured building blocks behind the figures: a short training run with its checks takes
36 minutes (8,209 updates); a one-hour training run takes 68 minutes; the three-drafter
60-request screen about 2 minutes and the 120-request panel about 10; the thinking-on and
sampled suites for three drafters about 41 minutes and the agentic suite about 31; this
morning's eight reference panels and two round profiles took 38 minutes in all.

| Step | GPU hours | Basis |
| --- | ---: | --- |
| 3. Fresh scenario measurements, diagnostics, depth checks | 2.0 | Three suites once (72 minutes), sampler qualification, depth neighbours on the slices, diagnostics |
| 4. Trainer profile and qualification | 1.4 | A 10-minute profile and two short runs; no speed-up assumed in any figure below |
| 5. Data tools and the new batch | 3.7 | One matched storage pair (1.2 hours); replay, batching and cold-loop checks (0.8); about 1.5 to 2 hours of writing, sized after its pilot |
| 6. Vocabulary scoring timing | 0.6 | Timing prototypes and whole-drafter checks |
| 7. Feature runs | 5.4 expected | Five core short runs, two removal comparisons and two triggered add-ons at 36 minutes each; additional triggered runs are extra |
| 8. Second seeds and scenario checks | 3.0 | Two short runs, the slices and panels, and one triggered full comparison (72 minutes); each further one adds the same |
| 9. Combined-loss pair and requalification | 1.4 | Two short runs and numerical checks |
| 10. Other public drafters | 0.3 | One screen each; unverified ones skipped |
| 11. Integration pilot | 0.35 | Uninterrupted and interrupted runs, export, serving at the longest context |
| 12. The long run | 3.2 | Up to three training hours plus checkpoints and diagnostics |
| 13. Readouts on all scenarios | 1.5 | Five or six readouts at 15 to 18 minutes |
| 14. Final timing | 2.0 | Three suites, greedy and supplementary panels, a reversed-order repeat where needed |
| **Total** | **about 25** | Wall time adds about four hours that cannot overlap the GPU: preparation, gates, integration and closeout |

Builds are done by parallel coding agents and gate the GPU: reports and per-scenario depth
(two to four hours) before step 3's policy figures and step 8; the trainer change and
feature code (two to five hours) before steps 4 and 7; the data tools, the slowest at four
to eight hours, before step 5. Steps 6 to 8 run ahead of step 5 if the data tools are
late. A trainer gain would shorten the training steps but is not counted. Most likely to
stretch the total: more triggered feature runs or full comparisons than assumed, and a
second round of work on the data tools or on writing long-context data.

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

### Scenario measurements

Existing entry points in `benchmarks/` ([benchmarks](../docs/benchmarks.md)); all need
native controls for this runtime before use, and the sampler qualified for it.

| Scenario | Short-run finalists and long-run readouts | Final timing (step 14) |
| --- | --- | --- |
| Greedy, short | 60-request screen | Screen and 120-request panel, reported by input-length band |
| Greedy, long | 120-request panel (18 requests at 2–8K tokens, 6 at 8–16K) | The same; six longest requests cannot resolve a 3% margin, which is stated |
| Thinking-on | 12-request slice | 40-request suite |
| Sampled | 12 requests at two seeds | 40 requests at three seeds |
| Agentic coding | 12 tasks across six projects | 60-task suite, with task rate, completion and exact trajectory |

Steps 8, 13 and 14 also include the maintained prefix18 and legacy live12 cohorts as
supplementary evidence, retaining their existing definitions and inspecting their
individual outcomes.

Depth calibration exists for greedy requests only; sampled and agentic deployment depths
are found by measuring depth 3 and its neighbours directly, on the slices.

For the owner-policy verdict, count exactly five scenarios: `greedy-short` (the ordinary
requests of the 120-request panel with fewer than 2,048 input tokens), `greedy-long` (the
remaining requests of that panel), `thinking-on`, `sampled` and `agentic-coding`. Ordinary
requests include structured and tool requests. Step 2 builds these five aggregates, each
with a paired 95% speed interval against Baseline computed from that scenario's request or
task rows under the maintained source-grouping rules, before step 3 uses them. These five
alone decide the two permitted exceptions. A scenario whose interval is missing or
insufficient cannot pass: the result is "not demonstrated". The 60-request screen and
finer length bands remain supplementary. The existing seven-scenario report, its stricter
verdict and its escalation rules stay unchanged and are reported separately. The hand-over
states one of three outcomes: policy passed, policy failed, or not demonstrated.

### Operating rules

- **Source.** Every run executes a frozen copy of committed source. Another session has
  uncommitted edits in the main checkout (`benchmarks/sandbox.py`, `docs/development.md`);
  they are left alone and are not part of any executed source unless that session commits
  them.
- **GPU and CPU.** One GPU job at a time under the lease in
  [execution](../docs/execution.md). Writing code runs in parallel; large compiles and the
  full gate count as CPU-heavy and wait while a timing run holds the GPU.
- **Preflight.** Every step over about one GPU hour gets a preflight with a pilot of ten
  minutes or less, the result that would change the decision, and kill criteria. Each
  feature comparison's preflight records its numerical adoption threshold and its tie rule
  (a tie keeps the control) before the run. The single long run's one-seed exception is
  recorded there as the owner's choice.
- **Close calls.** A speed decision inside its uncertainty gets fresh matched timing
  repeats, with the arm order reversed, before it is used; intervals over prompts do not
  measure drift between runs. All repeats are pooled, none is dropped, and a decision
  still unresolved after them is "not demonstrated".
- **Two different bars.** A short-run feature is adopted on its own predeclared threshold
  against its matched control. Passing the owner policy against Baseline is decided only
  for the assembled recipe.
- **Guards.** Time figures are not budgets, but every intent still carries finite update,
  wall and attempt limits, set generously, and there is no automatic retry after
  initialisation.
- **Interruptions.** The pause for checkpoint, export, screen and restore is requalified
  for the assembled recipe before the long run. Scenario slices that do not fit the pause
  are run after training from saved exports.
- **Serving changes.** Any change to the runtime or scoring path gets fresh controls for
  both drafters before its numbers are used.
- **Serving support first.** A feature or format the runtime cannot serve gets its timing
  prototype and export path before it is trained, not after.
- **Discarding.** "Discard" in the data loop means rebuilt values whose token streams and
  rebuild route are verified and kept. States leave only through the store's retirement
  route; nothing selected, needed for recovery or raw is removed.
- **Watching.** Long jobs are watched with the job-watch procedure and closed with its
  done-check; no periodic health-check automations.
- **Builds to start first,** because they gate the GPU: per-scenario depth and reports;
  data replay, batched capture and fused storage; the long-project data source; feature
  controls with training logs.

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
| "A representative new batch" | Its size is recorded straight after its small pilot, from the measured writing speed. Quotas fixed before writing: long agentic sessions and thinking-on requests both required, with caps per source and project, chosen by origin and coverage, never by DEV results, and with the confirmation and agentic-project exclusions applied |
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
norm limit of 1.0; blocks `{"4": 1.0}`; grouping `max_windows` 32, which at four slots is
a ceiling of 128 windows per update before the 3 GiB activation budget applies; position
weights `decay`; `frozen_format: baseline-k`; `trained_tables` off. The optimizer
constants and the gradient limit are not intent fields: they are pinned through the
frozen executed source and recorded in the hand-over. Normalising weights still train. This supersedes the older note in
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
7. Witness calibration and gradual precision, including calibration's effect on an
   untrained drafter; run blank, any removals and the four-run square with matched
   controls; then only the add-ons that meet their rule.
8. Second seeds; the panel and the three scenario slices for short finalists. Run the
   Level 2 scenario report for each short finalist; if it returns `level3_required`,
   complete the Level 3 comparison before adopting that finalist, applying the owner's
   stated tolerance policy separately from the unchanged stricter verdict.
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

Finish step 3 (register suite and slice references, group 0 at depth 3), then step 4's trainer
profile on the GPU while the builds finish; merge each build after review and its gate.

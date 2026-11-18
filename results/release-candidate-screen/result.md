# A three-hour calibrated run ties Baseline on ordinary requests and beats it by 10% on agentic coding, but narrowly misses the owner's speed policy

The 2026-11-16 screen (reviewed plan `tasks/feature-validation.md` at main commit `7f621ed9`, groups
0–7, run on branch `t3/candidate-screen`) set out to decide every field
of the release candidate and to rehearse the assembled recipe once. Its main findings:

- **One feature mattered: a calibrated start.** It sets each input clipping bound from the
  public tables' output error on TRAIN windows and fits row scale and midpoint by least squares.
  In 35-minute fresh runs it gained **+7.4 acceptance points** over two data-order seeds, more than
  the previous best drafter reached in four hours. Every other feature tied or lost within noise.
- **One long run** of the assembled recipe trained 3.1 GPU hours on about two-thirds of one pass
  over the new 9,047-record TRAIN set. In fresh paired timing at depth 3 it runs at **1.105× Baseline
  on agentic coding** and **0.978–1.003× on the four request scenarios**, with exact outputs and
  equal task completion. That is four shortfalls against two allowed, all but one inside the 3%
  tolerance: the owner policy **fails narrowly**. The learning curve was still rising until the
  schedule's tail.
- A **native vocabulary shortlist** adds 3.3% to our speed with no acceptance loss (1.022× the
  released Baseline on the 120-request panel), though Baseline would gain 4.4% from the same change.

## Outcome

### Final timing of the long run (plan 7d)

Fresh three-arm runs (target, Baseline, our final export `ae6569f6…`), runtime-v8, Q4KM, exact
verification, depth 3 for both drafters in every scenario (the measured best, below). Greedy,
thinking-on and sampled were repeated with our drafter before Baseline; the repeats reproduced
the ratios (0.991, 0.976, 0.994) and both orders are pooled. Intervals are paired source-group 95%
bootstraps; `greedy-long` uses the merged-strata interval by the owner's decision of 2026-11-16.

| Scenario | Units | Speed vs Baseline (95%) | Acceptance ours / Baseline | Verdict under the owner's rule |
| --- | --- | --- | --- | --- |
| Agentic coding | 60 tasks, 6 projects | **1.105 (1.089, 1.120)** task rate | 80.9% / 63.4%; completion 37/60 both | at least Baseline |
| Greedy, long (≥ 2,048 input tokens) | 24 requests × 2 | 1.003 (0.990, 1.016) | 58.2% / 61.2% | shortfall, tolerated |
| Sampled | 40 requests × 3 seeds × 2 | 0.994 (0.988, 1.000) | 45.7% / 50.0% | shortfall, tolerated |
| Greedy, short | 96 requests × 2 | 0.989 (0.979, 0.999) | 47.5% / 52.3% | shortfall, tolerated |
| Thinking-on | 40 requests × 2 | 0.978 (0.969, 0.986) | 59.9% / 66.6% | below the 0.97 tolerance |

Owner policy (at most two shortfalls, each with a lower bound ≥ 0.97): **failed**, with no
correctness or completion loss. The stricter seven-scenario verdict in `benchmarks/scenarios.py`
returns `not-established` (agentic coding and coding requests pass; chat/writing, long documents,
reasoning, sampled and thinking-on have lower bounds below 1). The maintained grouping rule cannot
give `greedy-long` an interval; under it the owner verdict is `not-demonstrated`.

### Learning curve (plan 7c)

Every 7,000 updates, candidate-only against saved runtime-v8 references, all outputs exact.
Acceptance in %:

| Updates | Screen (60) | Panel (120) | Thinking slice | Sampled slice | Agentic slice |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7,000 | 49.8 | 41.8 | 52.5 | 40.4 | 70.9 |
| 14,000 | 52.2 | 43.6 | 54.1 | 42.9 | 73.7 |
| 21,000 | 54.0 | 45.7 | 56.3 | 44.4 | 76.0 |
| 28,000 | 55.8 | 47.2 | 57.2 | 46.0 | 77.9 |
| 35,000 | 57.3 | 48.4 | 59.1 | 46.9 | 79.3 |
| 42,000 (final) | 57.6 | 48.7 | 59.4 | 47.3 | 79.3 |

Reference points: Baseline 60.6% (screen) and 53.4% (panel); the previous best drafter 55.2% and
46.2%. Gains run at 1.5–2 points per 7,000 updates and flatten only where the cosine step reaches
its tail. Mid-run points were not annealed and understate a run stopped there. Training loss fell
from 1.75 to about 1.4; DEV prefix survival stayed at 56–57% (120 windows, noisy).

### Candidate specification (hand-over)

Status: **measured** (a named comparison decided it), **kept** (deliberately unchanged, value
written out), **extrapolated** (a recommendation beyond what was run), **not demonstrated**.

| Field | Value | Status and evidence |
| --- | --- | --- |
| Start state and transition | Fresh from the public DFlash BF16 drafter `18cefa06…`, initialization `2f852a41…` | Kept (decided before the screen) |
| Weight format and feature set | One-bit feed-forward signs; learned row scale, row midpoint, input gains (shared gate/up), clipping bound | Measured: blank −0.57 (−1.24, +0.09) not admitted; clip step, gradual precision, round-state weighting, per-slot rounding not adopted |
| Initial values | **Calibrated start**: 16 TRAIN sequences × 16 windows, seed 20261116; clip bounds by output-weighted rounding error; trust-limited least-squares row fit; latents row ×3 clamp | Measured: +7.61 (+6.65, +8.60) and +7.21 (+6.10, +8.41) on two seeds; pooled +7.41 (+6.53, +8.30) |
| Precision schedule, hard-sign period | Hard signs throughout | Measured: gradual precision +0.16 (−0.46, +0.83) alone, −0.39 (−1.16, +0.38) with calibration |
| Step sizes, warm-up, decay | Sign 0.004; scale, midpoint, gain, clip, norm 1e-5; warm-up 2% (840 of 42,000); cosine to zero | Kept; clip step 1e-3 measured −0.21 (−1.25, +0.77) |
| Optimizer, gradient clipping | Fused AdamW, betas 0.9/0.999, epsilon 1e-8, no weight decay; one shared norm limit 1.0 | Kept (pinned by the executed source); the limit acted in most updates of the long run (mean factor about 0.56 at 40 minutes), logged per family |
| Update size, block size | Grouping `max_windows` 32, 3 GiB activations, 14 GiB device cap; blocks `{"4": 1.0}` | Kept |
| Position weights | `decay` | Kept; the reach-based add-on was not triggered (deficit shrinks with slot) |
| Which tables train | None (attention, fusion, vocabulary frozen; Baseline's Q4_K/Q6_K tables) | Kept (decided) |
| Data: bank, mix, order, passes | `train-final-01`: 9,047 records, 4.49M supervised positions (6K bank, expansion, 138 long-project sessions, 246 thinking-on requests, 30 pilot records); seed 8101 | Measured for 7b at about 0.67 passes; **recommended about two passes** (extrapolated) |
| Data storage and rebuild route | Eight-tap FP16; token replay at 32 tokens per batch rebuilds captures bit-identically | Measured; fused F16 storage not admitted: −0.20 (−1.11, +0.72), pooled over two seeds −0.36 (−1.07, +0.32) |
| Trainer arithmetic | `reference` | Measured: `fast-exact-tf32ff` 1.16× faster, exact checks pass, but −0.41 (−1.45, +0.58) on the assembled recipe and pooled −0.55 (−1.37, +0.27) over two seeds: not adopted |
| Training length, stopping, checkpoint | 7b: 42,000 updates, final checkpoint by rule | Measured for 7b; **recommended: schedule sized to about two passes (about 126,000 updates, about 9.5 GPU training hours at 0.27 s per update), final checkpoint** (extrapolated from the still-rising curve and the earlier S4 data curve, which gained until about two passes) |
| Export format and runtime kernel | W1Ax metadata 2 / GGUF 3; one-bit feed-forward on the recovered tensor-core kernel (runtime-v8); attention and fusion Q4_K/Q6_K copied from Baseline; vocabulary Q4_K | Measured (exact outputs everywhere); resume, export and serving traced in the step 11 pilot |
| Vocabulary scoring path | Full Q4_K scoring + top-10 selection, **plus the native shortlist** (all TRAIN label tokens + the request's own tokens) on runtime-v9 | Measured: shortlist +3.3% (3.1–3.5%) on the panel, no acceptance change; Baseline would gain +4.4% from the same change; Q2_K/Q3_K tables 2–6% slower; shortlist effect on thinking, sampled and agentic not measured |
| Draft depth per scenario | 3 for greedy, thinking-on, sampled and agentic | Measured (depth calibration; depths 2, 3, 4 on the sampled and agentic slices for both drafters) |
| Reference and release test | Baseline DFlash Q4KM `83586a63…`, cutoff 2026-11-16; Level 3 + owner policy; DSpark and DFlash 2 reported separately | Kept; DSpark 0.50× and DFlash 2 0.45× of Baseline at their authors' settings (descriptive) |
| Predicted margin per scenario | 7b measured: agentic 1.105, greedy-long 1.003, sampled 0.994, greedy-short 0.989, thinking 0.978. With the native shortlist added: greedy panel 1.022 measured; other scenarios about +2–3% extrapolated | Measured for 7b; one run and one seed; the effect of longer training is extrapolated and unquantified |

### Feature screen (group 3)

Fresh 8,209-update runs on the full TRAIN set, seed 8101, screened on the 60-request DEV screen,
paired source-group bootstrap against the matched control. Thresholds fixed in advance: a feature
needs Δ ≥ +1.0 with the interval above zero; a simplification or storage change needs its lower
bound ≥ −1.0.

| Run | Screen | Against | Δ points (95%) | Decision |
| --- | ---: | --- | --- | --- |
| Control (S4 recipe, every value explicit) | 48.95% | — | — | base |
| Blank quantizer | 48.38% | control | −0.57 (−1.24, +0.09) | not admitted |
| Gradual precision | 49.11% | control | +0.16 (−0.46, +0.83) | not adopted |
| **Calibrated start** | **56.56%** | control | **+7.61 (+6.65, +8.60)** | **adopted** |
| Calibrated start, seed 8102 | 56.42% | control 8102 (49.20%) | +7.21 (+6.10, +8.41) | pooled +7.41 (+6.53, +8.30) |
| Calibrated + gradual | 56.17% | calibrated | −0.39 (−1.16, +0.38) | not adopted |
| Calibrated + clip step 1e-3 | 56.35% | calibrated | −0.21 (−1.25, +0.77) | not adopted |
| Calibrated + round-state weighting (×2) | 56.22% | calibrated | −0.34 (−1.22, +0.54) | not adopted |
| Fast arithmetic (uncalibrated) | 48.72% | control | −0.23 (−0.91, +0.42) | passed alone |
| Calibrated + fast arithmetic | 56.15% | calibrated | −0.41 (−1.45, +0.58) | not adopted |
| Calibrated + fast arithmetic, seed 8102 | 55.73% | calibrated 8102 | −0.69 (−1.79, +0.37) | pooled −0.55 (−1.37, +0.27) |
| Calibrated on fused F16 storage | 56.36% | calibrated | −0.20 (−1.11, +0.72) | not admitted |
| Calibrated on fused F16 storage, seed 8102 | 55.89% | calibrated 8102 | −0.53 (−1.57, +0.53) | pooled −0.36 (−1.07, +0.32), not admitted |
| Calibrated + per-slot rounding | 55.62% | calibrated, rescreened on its runtime | −0.94 (−2.14, +0.12) | not adopted |

Per-slot rounding (owner request): a second learned clipping bound for slot 0 of each block, the
anchor token, in training, export and runtime (`slot_rounding_v1`, admission fixed by build P2
after the first parity run crashed). On the rebuilt runtime, target, Baseline and the calibrated
control reproduced runtime-v8 exactly, and an anchor bound equal to the shared bound reproduced
the control; the per-slot path costs 0.002–0.05 ms per round. Calibrated + per-slot rounding:
55.62% vs 56.56% (control rescreened on that runtime), **−0.94 (−2.14, +0.12): not adopted**. The
earlier ceiling check had predicted no gain: removing input rounding changed nothing on DEV.

Fused storage, second seed (owner request): calibrated on fused F16 storage, seed 8102, against
the calibrated seed-8102 control: 55.89% vs 56.42%, −0.53 (−1.57, +0.53). Pooled over both seeds
**−0.36 (−1.07, +0.32): not admitted**, narrowly (the bar is a lower bound of −1.0 or better). Two
seeds point the same small way, so a small cost is possible. The retest needed three attempts:
a disk refusal, a segmentation fault inside PyTorch's CPU library during conversion (not
reproduced on the retry) and a too-coarse disk bound on the training intent (8 GiB per state
declared against 4.75 GiB used; lowered to 6 GiB, which does not affect training).

Calibration's first version chose bounds from two massive-activation channels and collapsed after
one update; that bug was fixed (output-weighted bounds, trust-limited fit) before any decision.
Calibration removes clipping almost entirely: final clipped input shares were 0.0–0.1% in every
projection, against up to 57.5% in the control's last down projection.

### Where the gap was (group 0) and other checks

- At depth 3 before the screen, the previous best drafter ran 0.915 of Baseline on thinking-on,
  0.974 sampled, 0.965 greedy and 1.027 agentic (fresh native measurement, runtime-v8).
- Offline round simulation (within 1 point of native) located the deficit at real round starts:
  4.9 points behind Baseline there, 2.3 over all positions. On 60 TRAIN prompts the previous best
  drafter beat Baseline by 10.7 points (+7.5, +13.8) while trailing on unseen DEV prompts:
  generalisation from data, not capacity, limits it.
- Gains set to 1 and midpoints to 0 barely change the trained drafter; removing input rounding
  (bound kept) changes nothing on DEV, also for the calibrated drafter, which is why per-slot
  rounding has a near-zero ceiling.
- Trainer profile: feed-forward and vocabulary matmuls ran as FP32 SIMT kernels (55% of training
  GPU time).
- Data tools: token replay at 32 tokens per batch rebuilds stored captures bit-identically;
  batched writing at 8–32 slots is 2.2–2.5× faster but only 8 of 32 answers match one-at-a-time
  writing, so new data is written one request at a time.
- Vocabulary scoring (group 4): one-bit and low-rank first passes are slower than full scoring for
  our table; the target's own table is larger to read; the shortlist (above) is the one gain.
- Public comparators (group 5) at their authors' settings, BF16, exact outputs: DSpark depth 8
  0.50× and DFlash 2 depth 15 0.45× of Baseline (descriptive).
- Calibrated 35-minute finalist, Level 2 and Level 3 candidate-only against saved runtime-v8
  references, all outputs exact: 120 requests 47.7% / 53.4% (Baseline), thinking-on suite 57.1% /
  66.6%, sampled suite 44.2% / 50.0%, agentic suite 79.4% / 63.4% with 37/60 completed by both.
  The candidate-only verdict was `not-established` (Level 3 required).
- Step 11 pilot on the release data and settings: interrupted and resumed runs matched an
  uninterrupted one exactly, exports were byte-identical and each served with exact outputs.
- New TRAIN batch (group 2): 138 long-project sessions (initial contexts 11–20K tokens, permissive
  licences, DEV and confirmation projects excluded by family) and 246 thinking-on requests, 384
  records, 112 GB of captures, about 1.5 GPU hours.

## Scope and limitations

- One target (Qwen3.5-4B Q4KM), one RTX 5080, runtime-v8 (v9 for the shortlist), native Linux. NVFP4
  was not measured.
- All decisions use reused DEV cohorts; the protected confirmation sets and heldout24 were not
  touched. A pass would still need fresh confirmation.
- The long run is one seed with no matched control at its length and no separate measurement of
  the new batch's value (the owner's light version). Its gain cannot be split between calibration,
  data and length.
- Paired intervals cover prompt sampling, not drift between sessions; the reversed-order repeat
  showed no order effect at the 0.3% level.
- Short feature runs (35 minutes) can rank features differently from long runs; ties are not
  proof of no effect.
- The shortlist was measured on greedy requests only; its thinking, sampled and agentic effects
  are extrapolated. The recommended length (about two passes) is an extrapolation.
- The screen ran about 46 hours of wall time against 35–41 expected, mainly the 19:15 host freeze
  (about four GPU hours lost) and the owner's additions after the main verdict.

## Decision

Decided by the owner during the screen (2026-11-16/17): `greedy-long` is judged on the merged-strata
interval; the vocabulary shortlist is added as a native feature, with claims separating the
Baseline-applicable share; second seeds for fast arithmetic and fused storage; round-state
weighting and per-slot rounding were run.

Proposed, not decided: the release recipe is the 7b recipe with the native shortlist and a
schedule sized to about two passes of `train-final-01`. The measured 7b margin does not pass the
owner policy; the predicted release margin depends on longer training, which is extrapolated.
Thinking-on is the hardest scenario (−6.7 acceptance points, 0.978 speed).

The final export `ae6569f6…` beats the recorded acceptance leader `5749a05f…` on the 60-request
screen by +2.41 points (+0.98, +3.78) (57.64% vs 55.23%), which meets the
[selection rule](../../docs/domain.md#selection). Recording it as the new provisional acceptance
leader was proposed here; the owner approved it on 2026-11-18. Its screen is the saved reference
`285434a8…` and `runtime-v8-linux-01/specs/q4km-screen-leader-ae6569f6.json` checks new states
against it. The export stays under `arms/s12-long/output/exports/` while it leads.

## Evidence and reproduction

All raw evidence lives under `/home/philip/BitDraft/artifacts/feature-validation-01/` (main
checkout's ignored `artifacts/`): `pins/references.json`; `intents/`, `arms/*/intent.json`, journals
and screens; `group0/`, `group3/` (preflights and every paired comparison), `step3/` (runtime-v8
thinking, sampled and agentic references), `step4/` (trainer profile), `step12/` (long-run preflight
and admission), `step13/` (learning curve), `step14/` (depths, three-arm runs, reversed repeats,
policy and scenario reports), `builds/` (each build's runbook and GPU records), `logs/`, `closeout/`
(compact extractions) and `scripts/` (every queue and analysis script). Executed sources are frozen
in `executed-sources/`; each run's intent names its `code_sha256`. Build briefs are in `briefs/`.
[result.json](result.json) holds the compact measurements.

## Artifact disposition

Retained under `artifacts/` for an identified use:

- **Release inputs:** `datasets/train-final-01/mixture.json` and its components, including
  `datasets/long-sessions-01` (120 GB: 138 long-project sessions and 246 thinking-on requests,
  written once at about 1.5 GPU hours). Needed for the release run; regenerating them would
  repeat the target's generation.
- **Long run:** the final state (42,000 updates, `26022c2a…`, recovery point) and every export
  (final `ae6569f6…` and the 7K–35K curve exports). The 14K and 28K states (`977779c6…`,
  `b14ebe63…`, 10.2 GB) stay as branch points until the owner decides between continuing
  this run and a fresh release run; retire them through the store's route once that is decided.
- **Short runs:** every arm's intent, journal, exports, screens and comparison; the calibrated
  control's recovery state `c3ee0c58…` (the recovery reference of every retirement receipt
  and the base for later short screens on this recipe).
- **Timing and reference evidence:** `step3/` references, `step14/` runs and reports,
  `builds/native-vocabulary/` (runtime-v9 and its parity, shortlist and table runs),
  `runs/vocabulary-scoring/`, `builds/per-slot-rounding/runs/`, group 5 comparator records,
  `executed-sources/`, briefs, scripts and logs.

Retired through the project routes (receipts under `feature-validation-01/retirement/` and
`logs/state-retirement-*.json`):

- Closed short-run training states, kept only as exports, journals and screens (state store).
  Exact resume of those short runs is lost; none was selected.
- The first fused F16 corpus (55.8 GB) and the second-seed fused corpus (55.8 GB); regenerable with
  `w1a1_eagle.numerical.fusing` in about 15 GPU minutes.
- Replay qualification captures (12.8 GB), group 0 round-diagnostic captures (5.7 GB) and the
  six per-slot rounding runtimes (2.8 GB); each is regenerable from the retained source and
  inputs as its plan's `limitations` state.

One deviation: the partial output of the crashed fused conversion (45 GB in
`datasets/train-expansion-01-fused-2`, written by this screen minutes earlier and unreadable) was
removed with a recursive delete instead of the retirement command, to free disk for the retry.
No retained or shared file was inside it.

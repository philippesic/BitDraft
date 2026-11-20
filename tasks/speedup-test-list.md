# Running list of speed and precision tests for the candidate

## Goal

Keep one current list of the tests the owner and the lead agent agreed are worth running to
make low precision pay in the candidate's speed, and to price what it costs in acceptance.
The list is a record of what to test and why, not a schedule.

## Authorized

Recording only. Items 1, 2, 3 and 8 were authorized by the owner on 2026-11-14, and items 9
and 10 on 2026-11-16; all are finished. Item 11 was authorized on 2026-11-20. The other items are listed, not authorized: each starts only on the owner's
explicit go-ahead. The overnight screening sheet under State was authorized on 2026-11-15.
GPU and CPU-heavy work waits until no other thread holds the GPU lease.

## Done when

Every item below has a result entry under `results/` or a recorded decision to drop it.
Then delete this file.

## State

Owner: Philip. Recorded 2026-11-14 from the candidate-overhaul discussion.

Rule for adding an item: say whether it depends on low precision or another concrete
difference of the candidate, what it would do for Baseline, and whether existing drafters
use it. An optimization Baseline could adopt equally does not count toward closing the gap.

Context from the [round profile](../docs/benchmarks.md#round-profile) of S4 at depth 3
(record `artifacts/benchmarks/round-profile-baseline-s4-02/round-profile.json`, runtime-v6,
[result](../results/round-profile-baseline-s4/result.md)): drafter GPU time is 1.47 ms of a 9.4 ms round. One-bit feed-forward is
0.36 ms of that and reading its weights once would take about 0.06 ms, so items 1–3 together
can save at most about 3% of request speed. Vocabulary scoring (0.40 ms) and the attention
tables (0.27 ms) are still 4-bit.

| # | Test | Question | Cost | Retraining | Specific to low precision | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Time the dispatched one-bit kernels against a tensor-core kernel (signs expanded in registers, weights read once) and a shared-sign DP4A kernel at 2, 4, 8 and 16 slots, byte-identical outputs required | Can one-bit feed-forward run faster than 4-bit on this GPU? | GPU minutes | No | Yes | Done 2026-11-14 (branch `t3/benchmark-w1a1-kernels-3`, command `benchmark-kernels`, report `artifacts/benchmarks/imma-recovery/report.md`, all outputs byte-identical). The first variants gave no gain at 4 slots. The November 4 tensor-core kernel was then recovered from the archive and adapted to any slot count: at 4 slots one-bit feed-forward takes 136 µs for six layers against 320 for the dispatched kernel and 292 for Baseline's 4-bit tables (weights in cache); at 8 slots 149 against 607 and 534. In the served drafter (experimental runtime `runtime-v7-kernel-imma`, identical tokens and counts on 54 completions, three DEV requests) drafter time per round fell 6.4%, 8.6% and 15.6% at depths 1, 3 and 7 (1.80 to 1.64 ms at depth 3). Sharing sign bits across slots loses everywhere. Merged. Owner decision 2026-11-14: adopt the recovered kernel; no paired panel on WSL, so the benchmark runtime takes it on the native Linux host, where item 9 re-records every saved control. Result entry pending |
| 2 | Measure S4's share of zero activation codes and the share of codes equal between neighbouring slots, per layer, for gate/up and down inputs | Is skipping zero inputs or reusing work between slots worth a kernel? | One captured round | No | Partly: needs integer codes and exact sums | Measured 2026-11-14 on the continued state, 120 DEV windows, 4 slots (record `artifacts/qualification/s4-code-overlap-01/measurement.json`): skipping zeros would remove about 14% of feed-forward contributions, reuse between slots about 14%, both about 19%; gate/up inputs are 20% zero, down inputs 3%. At most about 0.06 ms per round before any overhead. Dropped by the owner 2026-11-14 as kernel projects. A gate that outputs exact zeros stays an option under model shape. Result entry pending |
| 3 | Run the public RSR-core GPU kernel on S4's gate, up and down tables against the dispatched kernel | Does grouping inputs by column pattern beat DP4A here? | Small | No | Yes | Done 2026-11-14 (branch `t3/benchmark-w1a1-kernels-3`, public kernel pinned at `99bcccde`, MIT licence, all outputs byte-identical): at 4 slots the best setting (blocks of 4 rows) takes about twice as long as the dispatched kernel per table and 3.4 times as long as fused gate/up (626 against 185 µs); its index is 4.2 bits per weight against 1 packed; the public kernel repeats per slot. Merged. Dropped by the owner 2026-11-14 for this GPU; the pinned code stays available for a CPU target. Result entry pending |
| 4 | Weight-precision ladder under S4's recipe: feed-forward at full precision, 4-bit, 2-bit and 1-bit, equal data and time | How many acceptance points does one bit cost? | GPU hours, four short runs | Yes | Yes | Screened 2026-11-16 as 30-minute runs, scored on DEV only, because these formats cannot be served ([result](../results/s4-feature-screen/result.md)). Full precision +2.2 top-1 / +1.7 survival points, 2-bit +1.4 / 0.0, three-valued +1.9 / +4.2. The 4-bit run crashed on the host and was then dropped by the owner as off-goal. Measured 2026-11-20 against a served 4-bit control ([result](../results/one-bit-attribution/result.md)): at equal data and updates one bit costs 7.84 panel points at 21,000 updates (10.84 at 2,000), and a 66-minute 4-bit control matches the leader's speed against Baseline |
| 5 | One-bit or 2-bit vocabulary and attention tables | Can low precision reach the 0.67 ms that feed-forward does not touch? | GPU hours | Yes | Yes | Partly measured 2026-11-17 ([result](../results/release-candidate-screen/result.md)): the final drafter's vocabulary table converted to Q3_K or Q2_K keeps acceptance within 0.6 points on the panel but runs 2.1–5.7% slower, because this runtime's Q2_K/Q3_K kernels are slower than Q4_K (drafting 1.53–1.76 ms per round against 1.24); not adopted. Trained one-bit or 2-bit tables and attention tables are not started |
| 6 | Teacher training: learn from the full-precision drafter's outputs (output matching, layer matching, gradual precision blending) | Does a teacher close the remaining acceptance gap? | GPU hours | Yes | Applies to the candidate only; Baseline is fixed | Rejected by the owner 2026-11-16. In 30-minute runs, matching the BF16-feed-forward teacher's hidden values gave +0.2 acceptance points (−0.4, +1.0) at the normal memory setting on native Linux. On WSL, output-score and hidden matching looked +1.0 to +1.2 against a weakened smaller-memory control ([result](../results/s4-feature-screen/result.md)) |
| 7 | Train the vocabulary table. Order after an independent review: save a pilot set of slot values split by prompt; probe per-token scale, per-token bias, a 2,560 × 2,560 output map and a rank-16 correction on it; convert to 4-bit and screen with the table tied; then frequent-token rows or a full table pulled toward the original; then joint training of table and body, trying both a low-rank correction and an optimizer with per-row and per-column statistics (Adafactor-style), each against a matched body-only control | Does a trainable table add acceptance once deployed tied and in 4-bit? Expected 0–3 points | First go or no-go about 1–2.5 GPU hours; full path 6–12 | Yes | The candidate carries its own table; Baseline borrows the target's and cannot train it | Not started. Owner decision 2026-11-14: run on the planned native Linux install, where CUDA sees 15.47 GiB with the display attached (measured 2026-11-16) |
| 8 | Store the frozen attention and fusion tables as Q4_K instead of Q4_0 | Does the runtime's shared-read path remove the 0.13 ms per round these tables cost over Baseline, and does acceptance hold? | GPU minutes, plus at most one optimizer-hour if acceptance drops | Only if acceptance drops | No: removes a handicap | Done 2026-11-14 on the continued state, no retraining needed (branch `t3/q4k-frozen-tables`, export `5749a05f…`, copying Baseline's own tables): the four table groups fell from 417 to 281 µs per round at depth 3, equal to Baseline's 282; screen acceptance 54.88% unchanged against 55.23% switched, difference +0.35 points (−0.37, +1.08). Merged. Owner decision 2026-11-14: `baseline-k` is the frozen table format of every new candidate recipe ([execution](../docs/execution.md#frozen-table-storage)). The screen also showed request speed 7.5% higher (5.4–10.0%) with the arms run one after the other; the paired check moves to item 9. Result entry pending |
| 9 | Re-measure Baseline and the candidate on the native Linux host: new saved controls for every cohort and precision, and a round profile of both | What are the real speeds and time shares on the new host, and how much CPU-side waiting was WSL? | About an hour of GPU time | No | No: measurement | Done 2026-11-16 ([result](../results/native-linux-references/result.md)): runtime-v8 (runtime-v7 plus the recovered kernel) built natively; target, Baseline and leader `5749a05f…` references on all eight routine panels; round profiles. Rounds are about 11% shorter than on WSL for both drafters, mostly CPU-side waiting. The leader runs at 0.96–1.00 of Baseline's speed with acceptance 4.6–7.9 points lower. Thinking, sampled and agentic references were not re-recorded |
| 10 | Train the attention and fusion tables together with feed-forward, against a matched run with them frozen | Does letting the 4-bit tables adapt to the one-bit feed-forward recover acceptance at no cost in size or speed? | GPU hours, two matched runs | Yes | Partly: the candidate is trained and Baseline is a fixed release, but nothing here needs one-bit arithmetic | Done 2026-11-16 ([result](../results/trained-attention-tables/result.md)): three one-hour matched runs at {32 windows, 1.75 GiB, 15 GiB}, Q4_0 tables. Screen acceptance: frozen 50.14%, small step (1e-5) +1.31 points (+0.55, +2.03), ten times that +0.87 (+0.08, +1.71); DEV within noise. Needs `expandable_segments` on this host. Owner decision pending: a longer confirmation only together with a Q4_K export of trained tables, whose absence costs about as much speed as the gain |
| 11 | Shortlist-only vocabulary table: look up input tokens in the target's Q6_K `token_embd`, converted to Q4_K on demand; score against our own Q4_K table holding only the shortlisted rows, rebuilt from `train-final-02-fused` labels; convert out-of-shortlist request-token rows on demand | Can the file drop from 517 to about 255 MiB (Baseline 364) and save about 260 MiB of VRAM with identical outputs, acceptance and the shortlist's speed? Our Q4_K table is an exact conversion of the target's Q6_K table (verified 2026-11-17), so no row is lost | About half a day: Opus runtime build of 2–3 hours plus 30–45 GPU minutes to check against the full-table shortlist build | No | No: removes a size handicap. Baseline already borrows the target's table and gains nothing | Done 2026-11-20 on the owner's go of that day ([result](../results/shortlist-only-table/result.md)): 284 MB against Baseline's 381 MB (68,728 rows, more labels than the estimate assumed), 340 MiB less GPU memory, identical drafts, owner policy passed with no shortfall at 1.05–1.13× Baseline as released and about 1.03× against Baseline with the same shortlist (1.12× agentic). Owner decision pending: whether this is the release form |

Evidence behind items 1–3: the November 4 binary-method screen and the November 5 execution
campaign ([speed and model quality](../results/execution-speed-and-model-quality/result.md),
[binary methods](../results/binary-method-comparison/result.md)). A tensor-core kernel was
the fastest tested at seven slots; a variant lost at one and three slots; four slots were
never measured.

Parked by the owner on 2026-11-14:

- Pre-summed lookup tables on the GPU. Our tests, the published LUT-GEMM kernel on this
  card and the T-MAC paper agree that they lose on GPUs. They are the first choice on CPUs.
- Mask-and-count (bit-plane popcount) and activations below 4 bits. They pay only with
  2-bit or 1-bit inputs, which is an acceptance question first.

Activation precision, reviewed with the owner on 2026-11-14 and left as it is (4-bit inputs
to gate and up, 8-bit to down). The November 4 campaign's heldout acceptance was 30.3% with
8-bit gate/up inputs, 30.0% with 4-bit, 7.3% with 2-bit and 14.5% with 1-bit
([result](../results/activation-family-comparison/result.md)); fewer bits buy no time while
the kernel computes in 8-bit, and feed-forward is about 0.14 ms of a round with the
recovered kernel. One lead was closed on 2026-11-16: the compiler for this card accepts
a 4-bit integer tensor-core instruction that covers 64 inputs where the 8-bit one covers 32
(`mma.sync.aligned.m16n8k64` with `.s4`), but the machine code for this card's architecture
(sm_120) runs it as two 8-bit instructions, and the one-bit instruction as eight. Both are
native only on RTX 30 and 40 cards, so neither can beat the adopted kernel here.

Generic, helps Baseline equally, not counted toward the gap: runtime CPU overhead (the GPU
idles about 3 ms per round), merging the small attention and handoff kernels, scoring and
picking the token in one kernel, a vocabulary shortlist. Owner decision 2026-11-17: the
shortlist ships natively with the candidate (runtime-v9, +3.3% on the panel); every claim states
that Baseline gains +4.4% from the same change
([result](../results/release-candidate-screen/result.md)).

Owner decision 2026-11-19: the long fused run's final export `ef068eba…` is the provisional
acceptance leader. It is +2.1 to +2.9 points over `ae6569f6…` on the screens and passes the
owner release policy ([result](../results/fused-data-expansion/result.md)). New states are
screened against `specs/q4km-screen-leader-ef068eba.json` and
`screen-v2/q4km-screen2-leader-ef068eba.json`.

Starting point for acceptance work, decided by the owner on 2026-11-14: the final state of
the [learning-curve run](../results/s4-data-curve/result.md) (state `769e184e…`), the best
measured candidate, not the 8,209-update S4 state, with its frozen tables in Baseline's
format. Its export in that format is `5749a05f…`
(`artifacts/benchmarks/frozen-table-switch-01/exports/`); the first training run branches
from the state with `baseline-k`, which records the change of tables. The saved references
of the [scenario benchmarks](../results/scenario-benchmark-levels/result.md) were recorded
on the earlier export `405e326b…`, without this change or the recovered kernel: agentic
coding ties Baseline, thinking-on requests run at 0.88 of Baseline's speed and sampled
requests at 0.95. Item 9 re-recorded the four routine cohorts for the leader on native
Linux; the thinking, sampled and agentic references remain WSL records.

The host moved from WSL2 to native Linux on 2026-11-15; item 9 recorded its controls and
round profile.

Possible later arm, not a listed test: a language model choosing training step sizes. The
paper the owner mentioned has not been identified; its decisions would have to be journaled
for exact replay.

Quantizer features, screened 2026-11-16 ([result](../results/s4-feature-screen/result.md)) as
30-minute DEV-only runs:
- a second one-bit plane: +3.3 DEV survival points;
- three-valued weights: +4.2;
- one scale per 128 inputs: +0.8, no gain.

The owner dropped 4-bit down tables in layers 2 and 5. Three-valued weights and the second plane
are the only candidates for a longer look, and each needs a serving kernel before acceptance can be
measured. The restart test answered the open learning-curve question: the flattening came from the
data, not from the step size reaching zero.

Lessons from published one-bit methods, noted by the owner on 2026-11-16 for later, not
listed tests (BiLLM, PB-LLM, ARB-LLM, OneBit, BitNet b1.58, a4.8 and v2, from memory, not
re-read). Our recipe already matches their row and column scaling (row scale, midpoint and
input gains), BitNet a4.8's 4-bit/8-bit input split and the straight-through sign gradient.
It differs in two values instead of three, one-bit feed-forward only, thousands of prompts
instead of billions of tokens, and no input rotation. Candidate tests:
1. An online Hadamard rotation before 4-bit input rounding (BitNet v2). It targets the
   largest measured loss at initialization, input rounding; serving needs a small kernel.
2. Higher precision for only the most important weights (BiLLM, PB-LLM) instead of a full
   second plane, aiming at most of the plane's +3.3 survival points at about 1.1 bits.
3. Three values instead of two, which the literature and our +4.2 survival agree on.
4. Much more training data, consistent with the data-limited learning curve.

Earlier, untrained evidence is from conversions of the older EAGLE drafter in September
([result](../results/scale-and-head-recovery/result.md)).

## Next

Owner go of 2026-11-20 (confirmed 09:40 PST) covers the lead's four model steps: the 4-bit
control (done), item 11 (done), a second tool-call convention for agentic coding (next) and
one confirmation run. The confirmation sets are spent only once the owner names
the drafter.

The owner decides what follows from the [one-bit attribution](../results/one-bit-attribution/result.md):
whether the release stays one-bit, and whether a 4-bit control of the leader's length or a 4-bit
drafter with a Q4_K vocabulary table is run. Neither is authorized.

The owner decides on item 10 (a longer confirmation with a Q4_K export of trained tables, or
keep frozen tables) and whether three-valued weights or a second one-bit plane get a serving
kernel and a longer run. Write result entries for items 1, 2, 3 and 8. Items 5 and 7 can run
on the native Linux host; item 7's vocabulary probes depend on the item 10 decision. Level 2
and 3 decisions on this host first need thinking, sampled and agentic references.

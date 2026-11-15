# Running list of speed and precision tests for the candidate

## Goal

Keep one current list of the tests the owner and the lead agent agreed are worth running to
make low precision pay in the candidate's speed, and to price what it costs in acceptance.
The list is a record of what to test and why, not a schedule.

## Authorized

Recording only. Items 1, 2, 3 and 8 were authorized by the owner on 2026-11-14 and are
finished. The other items are listed, not authorized: each starts only on the owner's
explicit go-ahead. GPU and CPU-heavy work waits until no other thread holds the GPU lease.

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
| 4 | Weight-precision ladder under S4's recipe: feed-forward at full precision, 4-bit, 2-bit and 1-bit, equal data and time | How many acceptance points does one bit cost? | GPU hours, four short runs | Yes | Yes | Not started |
| 5 | One-bit or 2-bit vocabulary and attention tables | Can low precision reach the 0.67 ms that feed-forward does not touch? | GPU hours | Yes | Yes | Not started; depends on items 1 and 4. Owner intends to try the vocabulary table below 4 bits after item 7 |
| 6 | Teacher training: learn from the full-precision drafter's outputs (output matching, layer matching, gradual precision blending) | Does a teacher close the remaining acceptance gap? | GPU hours | Yes | Applies to the candidate only; Baseline is fixed | Not started; design not yet discussed |
| 7 | Train the vocabulary table. Order after an independent review: save a pilot set of slot values split by prompt; probe per-token scale, per-token bias, a 2,560 × 2,560 output map and a rank-16 correction on it; convert to 4-bit and screen with the table tied; then frequent-token rows or a full table pulled toward the original; then joint training of table and body, trying both a low-rank correction and an optimizer with per-row and per-column statistics (Adafactor-style), each against a matched body-only control | Does a trainable table add acceptance once deployed tied and in 4-bit? Expected 0–3 points | First go or no-go about 1–2.5 GPU hours; full path 6–12 | Yes | The candidate carries its own table; Baseline borrows the target's and cannot train it | Not started. Owner decision 2026-11-14: run on the planned native Linux install, where about 15.5 GiB should be usable against about 14 GiB under WSL |
| 8 | Store the frozen attention and fusion tables as Q4_K instead of Q4_0 | Does the runtime's shared-read path remove the 0.13 ms per round these tables cost over Baseline, and does acceptance hold? | GPU minutes, plus at most one optimizer-hour if acceptance drops | Only if acceptance drops | No: removes a handicap | Done 2026-11-14 on the continued state, no retraining needed (branch `t3/q4k-frozen-tables`, export `5749a05f…`, copying Baseline's own tables): the four table groups fell from 417 to 281 µs per round at depth 3, equal to Baseline's 282; screen acceptance 54.88% unchanged against 55.23% switched, difference +0.35 points (−0.37, +1.08). Merged. Owner decision 2026-11-14: `baseline-k` is the frozen table format of every new candidate recipe ([execution](../docs/execution.md#frozen-table-storage)). The screen also showed request speed 7.5% higher (5.4–10.0%) with the arms run one after the other; the paired check moves to item 9. Result entry pending |
| 9 | Re-measure Baseline and the candidate on the native Linux host: new saved controls for every cohort and precision, and a round profile of both | What are the real speeds and time shares on the new host, and how much CPU-side waiting was WSL? | About an hour of GPU time | No | No: measurement | Decided by the owner 2026-11-14; starts after the move. No timing from the WSL host is to be used for decisions after it |
| 10 | Train the attention and fusion tables together with feed-forward, against a matched run with them frozen | Does letting the 4-bit tables adapt to the one-bit feed-forward recover acceptance at no cost in size or speed? | GPU hours, two matched runs | Yes | Partly: the candidate is trained and Baseline is a fixed release, but nothing here needs one-bit arithmetic | Owner decisions 2026-11-14: runs before item 7, because the vocabulary probes depend on the body; three fresh one-hour runs on the same data and order (frozen control, unfrozen at a small step size, unfrozen at about ten times that), decided on screen acceptance, with the trained tables exported as Q4_0; waits for the native Linux host. The training setting is being built now, code and CPU tests only (branch pending). By parameter count the tables are 150M weights, 0.56 GiB per copy: about 1.7 GiB more with the current optimizer, 0.56 GiB with one that keeps only row and column statistics. Trained tables cannot be copied from Baseline, so export needs its own Q4_K conversion, which the pinned GGUF package lacks. Together with item 7's full-table arm it exceeds the card unless one of them uses the lighter method |

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
recovered kernel. One untested lead, not a listed test: the compiler for this card accepts
a 4-bit integer tensor-core instruction that covers 64 inputs where the 8-bit one covers 32
(`mma.sync.aligned.m16n8k64` with `.s4`, checked by assembling a probe; speed unmeasured).
Gate and up inputs already fit it without retraining. It matters most if the vocabulary
table becomes one-bit.

Generic, helps Baseline equally, not counted toward the gap: runtime CPU overhead (the GPU
idles about 3 ms per round), merging the small attention and handoff kernels, scoring and
picking the token in one kernel, a vocabulary shortlist.

Starting point for acceptance work, decided by the owner on 2026-11-14: the final state of
the [learning-curve run](../results/s4-data-curve/result.md) (state `769e184e…`), the best
measured candidate, not the 8,209-update S4 state, with its frozen tables in Baseline's
format. Its export in that format is `5749a05f…`
(`artifacts/benchmarks/frozen-table-switch-01/exports/`); the first training run branches
from the state with `baseline-k`, which records the change of tables. The saved references
of the [scenario benchmarks](../results/scenario-benchmark-levels/result.md) were recorded
on the earlier export `405e326b…`, without this change or the recovered kernel: agentic
coding ties Baseline, thinking-on requests run at 0.88 of Baseline's speed and sampled
requests at 0.95. Item 9 re-records them for the new configuration.

Planned by the owner on 2026-11-14: move the host from WSL2 to a native Linux install.
A host or driver change requires new saved controls ([domain](../docs/domain.md)), the
memory cap in [execution](../docs/execution.md) needs re-measuring, and a round profile on
the new host will show how much of the CPU-side waiting was WSL.

Possible later arm, not a listed test: a language model choosing training step sizes. The
paper the owner mentioned has not been identified; its decisions would have to be journaled
for exact replay.

Not yet listed, owner decision pending: quantizer features (finer scales, a second one-bit
layer, three-valued weights, selective higher precision). Earlier evidence is from untrained
conversions of the older EAGLE drafter in September
([result](../results/scale-and-head-recovery/result.md)): one fitted scale per 128 weights
raised accepted drafts per round from 0.12 to 0.43 against 1.04 for the 4-bit control, and
keeping one table at 8 bits helped most for attention (0.61) and least for the down table
(0.47). Neither has been tested under training with the current recipe. Item 10 covers training
the attention and fusion tables.

Open question, not yet a listed test: whether the
[learning curve](../results/s4-data-curve/result.md) flattened because of data or because
the step size reached zero.

## Next

Write result entries for items 1, 2, 3 and 8. The owner decides which of items 4–7 to
start, or item 10, and whether to list the quantizer-feature tests; item 9 waits for
the native Linux host.

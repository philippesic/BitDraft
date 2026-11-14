# Running list of speed and precision tests for the candidate

## Goal

Keep one current list of the tests the owner and the lead agent agreed are worth running to
make low precision pay in the candidate's speed, and to price what it costs in acceptance.
The list is a record of what to test and why, not a schedule.

## Authorized

Recording only. Item 1 was requested by the owner on 2026-11-14 and is assigned. Items 2, 3 and 8
were authorized the same day. The other items are listed, not authorized: each starts only
on the owner's explicit go-ahead. GPU and
CPU-heavy work waits until no other thread holds the GPU lease.

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
no result entry yet): drafter GPU time is 1.47 ms of a 9.4 ms round. One-bit feed-forward is
0.36 ms of that and reading its weights once would take about 0.06 ms, so items 1–3 together
can save at most about 3% of request speed. Vocabulary scoring (0.40 ms) and the attention
tables (0.27 ms) are still 4-bit.

| # | Test | Question | Cost | Retraining | Specific to low precision | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Time the dispatched one-bit kernels against a tensor-core kernel (signs expanded in registers, weights read once) and a shared-sign DP4A kernel at 2, 4, 8 and 16 slots, byte-identical outputs required | Can one-bit feed-forward run faster than 4-bit on this GPU? | GPU minutes | No | Yes | Assigned 2026-11-14 (kernel benchmark, branch `t3/benchmark-w1a1-kernels-3`), waiting for the GPU |
| 2 | Measure S4's share of zero activation codes and the share of codes equal between neighbouring slots, per layer, for gate/up and down inputs | Is skipping zero inputs or reusing work between slots worth a kernel? | One captured round | No | Partly: needs integer codes and exact sums | Authorized by the owner 2026-11-14; assigned |
| 3 | Run the public RSR-core GPU kernel on S4's gate, up and down tables against the dispatched kernel | Does grouping inputs by column pattern beat DP4A here? | Small | No | Yes | Authorized by the owner 2026-11-14; queued behind item 1 in the same benchmark |
| 4 | Weight-precision ladder under S4's recipe: feed-forward at full precision, 4-bit, 2-bit and 1-bit, equal data and time | How many acceptance points does one bit cost? | GPU hours, four short runs | Yes | Yes | Not started |
| 5 | One-bit or 2-bit vocabulary and attention tables | Can low precision reach the 0.67 ms that feed-forward does not touch? | GPU hours | Yes | Yes | Not started; depends on items 1 and 4. Owner intends to try the vocabulary table below 4 bits after item 7 |
| 6 | Teacher training: learn from the full-precision drafter's outputs (output matching, layer matching, gradual precision blending) | Does a teacher close the remaining acceptance gap? | GPU hours | Yes | Applies to the candidate only; Baseline is fixed | Not started; design not yet discussed |
| 7 | Train the vocabulary table. Order after an independent review: save a pilot set of slot values split by prompt; probe per-token scale, per-token bias, a 2,560 × 2,560 output map and a rank-16 correction on it; convert to 4-bit and screen with the table tied; then frequent-token rows or a full table pulled toward the original; then joint training of table and body, trying both a low-rank correction and an optimizer with per-row and per-column statistics (Adafactor-style), each against a matched body-only control | Does a trainable table add acceptance once deployed tied and in 4-bit? Expected 0–3 points | First go or no-go about 1–2.5 GPU hours; full path 6–12 | Yes | The candidate carries its own table; Baseline borrows the target's and cannot train it | Not started. Owner decision 2026-11-14: run on the planned native Linux install, where about 15.5 GiB should be usable against about 14 GiB under WSL |
| 8 | Store the frozen attention and fusion tables as Q4_K instead of Q4_0 | Does the runtime's shared-read path remove the 0.13 ms per round these tables cost over Baseline, and does acceptance hold? | GPU minutes, plus at most one optimizer-hour if acceptance drops | Only if acceptance drops | No: removes a handicap | Authorized by the owner 2026-11-14; assigned |
| 9 | Re-measure Baseline and the candidate on the native Linux host: new saved controls for every cohort and precision, and a round profile of both | What are the real speeds and time shares on the new host, and how much CPU-side waiting was WSL? | About an hour of GPU time | No | No: measurement | Decided by the owner 2026-11-14; starts after the move. No timing from the WSL host is to be used for decisions after it |

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

Generic, helps Baseline equally, not counted toward the gap: runtime CPU overhead (the GPU
idles about 3 ms per round), merging the small attention and handoff kernels, scoring and
picking the token in one kernel, a vocabulary shortlist.

Starting point for acceptance work, decided by the owner on 2026-11-14: the final state of
the [learning-curve run](../results/s4-data-curve/result.md) (state `769e184e…`, export
`405e326b…`), the best measured candidate, not the 8,209-update S4 state.

Planned by the owner on 2026-11-14: move the host from WSL2 to a native Linux install.
A host or driver change requires new saved controls ([domain](../docs/domain.md)), the
memory cap in [execution](../docs/execution.md) needs re-measuring, and a round profile on
the new host will show how much of the CPU-side waiting was WSL.

Possible later arm, not a listed test: a language model choosing training step sizes. The
paper the owner mentioned has not been identified; its decisions would have to be journaled
for exact replay.

Open question, not yet a listed test: whether the
[learning curve](../results/s4-data-curve/result.md) flattened because of data or because
the step size reached zero.

## Next

Wait for item 1's benchmark report. The owner decides which of items 2–6 to start.

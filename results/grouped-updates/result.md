# Grouped updates raise HE4 update throughput about twentyfold within the memory limit

HE4 training spent almost every update on fixed work over the 425M sign latents and the
2.5 GB vocabulary, whatever the update carried. One update now trains on a group of
windows from one sequence. On real weights the update itself processes about 2,100
labels/s at typical context instead of about 90, while GPU memory stays below the point
where WSL spills. Loading each group's sequence from disk now limits end-to-end training
to about 970 labels/s.

## Outcome

RTX 5080, continuing the 220,095-update HE4broad state on the frozen 6K TRAIN bank.
Each row is the median of 11 synchronized updates after one warm-up. Context is the group's
largest anchor; labels are full 15-label windows. "Previous" is the code before this change
(one window, unfused AdamW, previous sign function, finite scan every update).

| Context tokens | Previous ms / labels/s | One window now | Rule-sized group | ms | Labels/s | Peak allocated / reserved GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 586 | 160 / 94 | 102 / 146 | 32 | 200 | 2,395 | 12.05 / 12.22 |
| 1,086 | 163 / 92 | 113 / 132 | 32 | 224 | 2,139 | 12.29 / 12.62 |
| 2,283 | 181 / 83 | 124 / 121 | 28 | 255 | 1,647 | 12.53 / 12.80 |
| 4,088 | 205 / 73 | 152 / 99 | 20 | 264 | 1,135 | 12.46 / 13.28 |
| 8,319 | 271 / 55 | 228 / 66 | 10 | 306 | 490 | 12.38 / 12.90 |
| 13,456 | 354 / 42 | 306 / 49 | 4 | 338 | 178 | 12.12 / 12.91 |

The previous path itself peaked at 13.46 GiB allocated / 13.94 GiB reserved at 13,456 tokens.
In the frozen bank half of all windows have context below about 1,100 tokens and 90% below
2,400, so most updates use the full 32 windows.

A fit over 24 measured group sizes (1–48 windows, 600–13,470 tokens) predicts peak allocation
within about 0.1 GiB: 9.7 GiB fixed, 117 KB per context token, and per slot row 4.2 MB plus
0.9 KB per context token. Reserved memory exceeded allocation by up to 0.8 GiB at long
contexts. The rule uses coefficients slightly above the fit and a 3 GiB activation budget.

Grouped logits agree with the single-window path at the model's rounding sensitivity.
Across 24 windows on four sequences (481–13,470 tokens) the largest logit difference was
0.32 and the mean 0.006–0.028, with up to 2 of 16 slot argmaxes changed. Running the
unchanged single-window path on CPU instead of GPU gave the same size of difference
(maximum 0.17–0.21, up to 1 of 16 argmaxes changed), while repeating it on GPU was
bit-identical. The 4-bit activation quantizers turn last-bit differences into changed codes;
on the reduced CPU model grouped and single-window logits agree to 5e-7.

The public `run` entry then continued the state for 300 updates on the same corpus with
default grouping (seed 8101, plain shuffled groups). Loss stayed finite (4.51 at update
100, 5.78 at 200) with gradient norms near the 1.0 clip (1.34, 1.28). The journal's 300
exposure events list 8,575 windows and 126,417 labels, equal to the worker's label
accounting; groups averaged 28.6 windows. Update time averaged 248 ms (1,701 labels/s).
Loading each group's sequence from disk added 187 ms per update outside the update
clock, so end-to-end throughput was about 970 labels/s.

The first attempt of that run, without a device memory cap, allocated at most 12.56 GiB
but reserved 16.8 GiB: varying group shapes fragmented PyTorch's cache beyond physical
memory, and updates averaged 295 ms. The isolated measurements above emptied the cache
between configurations and could not show this. The worker now caps CUDA memory at
`device_bytes` (14 GiB by default); the capped rerun reserved exactly 14.0 GiB, allocated
12.56 GiB and reproduced the same losses.

## Scope and limitations

The disk load of each group's sequence is now the largest remaining cost; prefetching the
next group during the current update is the obvious next step and is not implemented.
This qualifies mechanics, speed and memory on one GPU, one state and one corpus. It does
not measure learning: step sizes were set for one window per update and need retuning,
which is being decided separately. The memory fit is specific to this geometry, toolkit
and WSL host; other devices need their own measurements. Per-update time includes the
host-to-GPU feature copy, which grows with context. Grouping weights every update equally,
so a window in a small group (a short or long-context sequence) carries more weight than
one in a full group.

## Decision

Adopted: grouped updates are the training path, with 32 windows, a 3 GiB activation
budget and a 14 GiB device cap as defaults. The 220,095-update state remains loadable for continue and branch;
exact resume of its recipe is no longer possible, as the owner accepted.

## Evidence and reproduction

Executed source: this branch's grouped-update commits; the previous-path baseline ran a
source-only checkout of `83145a67`. Inputs: source `dflash-4b-bf16.gguf` (`18cefa06…`),
vocabulary `he4-q4-head-basis.gguf` (`a68f55d9…`), state `e1d96b58…` (220,095 updates),
corpus `train-bank-v8-fp16-01/corpus.json` (`3fbc0135…`). [Compact measurements](result.json)
record the tables above. The run's journals and reports are kept beside the scripts
(`run/output`, `run/output-uncapped`, and `run/output-zero-updates`, whose wall budget was
below the parent's recorded 49,625 seconds so it published without updating). Reproduce with `scripts/qualify.py MODE OUTPUT` (`equivalence`,
`grid`, `rule`, or `old` from the previous checkout) and `scripts/noise.py OUTPUT`, run in
the numerical interpreter under the project GPU lease.

## Artifact disposition

Retained in `artifacts/qualification/grouped-updates/` of the producing worktree: raw
measurement JSON, the qualification scripts and the 300-update run's intent, journal and
report for each attempt. Each attempt's final state and GGUF export were deleted after
verification: they were qualification outputs with an unretuned step size and no further
use, so exact resume from update 220,395 is not available. The 220,095-update
parent state, models and corpus were read only.

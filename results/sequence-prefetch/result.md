# Prefetching the next sequence lets HE4 training run within 9% of update speed

Grouped updates made each optimizer step fast enough that loading the group's sequence
from disk took almost as long as the step itself: end-to-end training ran at about half
of update throughput. Loading the next group's sequence in the background, and making
each load cheaper, removes most of that wait without changing what is trained. On real
weights, 300 updates ran at 1,685 labels/s end to end instead of 810. Losses, parameters,
Adam moments and RNG state after those updates are bit-identical to the previous code.

## Outcome

RTX 5080, continuing the 220,095-update HE4broad state on the frozen 6K TRAIN bank for
300 updates (seed 8101, default grouping, 126,417 labels in 8,575 windows) through the
public `run` entry. Page cache was evicted for every file of the 302 sequences involved
before each arm, so every first load was cold. End-to-end throughput counts labels over
data plus successful-update seconds; the training-loop wall clock (first `training-ready`
to `training-boundary`) agrees within 0.5%.

| Arm | Data wait / update | Update / update | Update-only labels/s | End-to-end labels/s | End-to-end / update-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before (previous loader, no prefetch) | 240 ms | 280 ms | 1,507 | 810 | 54% |
| Prefetch, one reader thread | 29 ms | 253 ms | 1,667 | 1,495 | 90% |
| Prefetch, four reader threads (adopted) | 21 ms | 229 ms | 1,842 | 1,685 | 91.5% |

All three arms logged the same losses (4.5100 at 100 updates, 5.7768 at 200) and ended
with the same final loss (5.2535) and gradient norm. The final states of both prefetch arms
match the before arm exactly in all 116 parameter tensors, all 312 Adam tensors, optimizer
groups and CPU/CUDA RNG.

Update time itself varied between arms (280, 253 and 229 ms), more than the halved
host-to-GPU copy explains (about 2 ms at typical context). The host was short of memory,
with the worker partly swapped out in every arm, so compare each arm's ratio rather than
update times across arms. The previous measurement of this run averaged 248 ms per update
and 187 ms of loading.

### Where the load time went

Profiled offline on 24 distinct sequences (mean 1,456 tokens, 57 MB of FP16 features each):

| Stage, cold | Previous loader | New loader |
| --- | ---: | ---: |
| Metadata and small files | 6 ms | 4 ms |
| Feature files: read | (in hash and conversion) | 26 ms |
| Feature files: SHA-256 | 33 ms, reading from disk | 22 ms, over the bytes in memory |
| FP16 to FP32 | 40 ms | none on the host |
| Tap concatenation / interleave | 24 ms | 8 ms (FP16) |
| Finiteness scan | 61 ms (FP32) | 7–16 ms (FP16 bit patterns) |
| Total, cold / warm mean | 165 / 140 ms | 83 / 60 ms |

Disk was not the bottleneck: reading a sequence's eight files cold took about 20–26 ms.
The previous loader converted to FP32 on the CPU, concatenated, then scanned the FP32
array, and hashed each file in a separate read. Four reader threads cut the read-and-hash
part from 50 ms to 14 ms for a typical sequence and from 578 ms to 113 ms for the longest
(13,471 tokens); eight threads gave 11 ms and 89 ms.

Of the remaining 21 ms of data time per update, about 4.5 ms is the two fsynced journal
appends (attempt and exposure) that run inside the data clock. The rest is waiting for
background loads that did not finish within the previous update, mostly for long
sequences. An offline simulation of 3,000 updates predicted a smaller wait (about 5%
with one reader); loads run slower alongside live training than in isolation.

Keeping features in FP16 also halves the host-to-GPU copy inside the update clock: about
2 ms instead of 4 ms at 1,120 tokens and 28 ms instead of 54 ms at 13,456. Pinned host memory
would save about another 1 ms but costs 57 ms to allocate per sequence, so it is not used.

### Memory

Peak GPU allocation was unchanged (12.54–12.56 GiB). Host memory was measured by polling
the worker: peak resident set 11.4 GB before, 12.7 GB with four readers. In every arm part
of the worker was swapped out under host-wide pressure (peak worker swap 3.0, 0.7 and 1.4 GB),
so the resident peaks are not clean footprints. The design bound is the current sequence,
one upcoming sequence and that load's read buffer, all in storage precision: at most about
1.7 GB for the longest sequence, typically about 170 MB. The previous loader's own
transient was larger, about 2.5 GB for the longest sequence. Available host memory stayed
above 6.1 GB in every arm.

### Sequences per update a later mixed-sequence change could load

This is an estimate, not a measurement. With four reader threads a typical cold sequence
costs about 35–45 ms of background work offline (metadata 4, read and hash 14, scan 7–16,
interleave 8), and loads run slower beside training. Against a 230–250 ms update, one
background loader therefore sustains about 4–5 distinct typical sequences per update
before loading again limits training; plan around 4. Loading two sequences concurrently
could roughly double that. Beyond about 10, disk read bandwidth (about 2.5 GB/s observed,
57 MB per typical sequence) becomes the limit, and host memory then holds every distinct
sequence of an update plus the next update's. Long sequences cost about ten times a typical one.

### On-disk layout

A different layout would save little relative to prefetching. Storing taps interleaved
would remove the 8 ms interleave, and one file per sequence would cut seven file opens.
A faster hash than SHA-256 would change the authentication receipts. None of these is
needed to keep loading off the critical path; no proposal is made.

## Scope and limitations

One GPU, one state, one corpus, 300 updates per arm, one run per arm. The data clock
includes journal writes, so it overstates loading by about 4.5 ms per update. Host memory
pressure from other work varied between arms. The per-update wait distribution was not
recorded, so the split between long-sequence waits and other overhead is inferred. This
qualifies mechanics and throughput; it says nothing about learning. The step-size
caveats of [grouped updates](../grouped-updates/result.md) still apply.

## Decision

Adopted: training prefetches the next group's sequence on one background thread with
four reader threads per load, authenticates feature bytes from memory, keeps features in
FP16 on the host and converts them to FP32 on the device. Exactness holds by construction
and was checked on CPU fixtures and on real weights.

## Evidence and reproduction

Executed source: the before arm ran an extraction of `b5332c68` (source identity
`9df04f5d…`); the single-reader arm ran `38240c3d` (`87edc1d7…`); the adopted arm ran
`1c884ab8` (`a7818247…`). Each run froze its source under `sources/`. Inputs: source
`dflash-4b-bf16.gguf` (`18cefa06…`), vocabulary `he4-q4-head-basis.gguf` (`a68f55d9…`),
parent state `e1d96b58…`, corpus `train-bank-v8-fp16-01/corpus.json` (`3fbc0135…`).
[Compact measurements](result.json) record every number above, extracted from the journals.

CPU coverage is in `tests/numerical/test_prefetch.py`:
- losses and final state identical with and without prefetch
- exact resume, branch and continue across a pending prefetch
- an injected update failure through the worker entry gives the same journal and state as
  without prefetch, with no entry for the prefetched group
- background-load release
- authentication refusals and the finiteness bit scan

`tests/numerical/test_model.py` also checks that no capture is alive when evaluation starts.

The scripts are in `artifacts/qualification/prefetch/scripts/` of the producing worktree:
- `profile.py`, `profile_new.py` and `parallel_taps.py` for the CPU profiles
- `transfer.py` for the copy timings (needs the GPU lease)
- `journal_cost.py`
- `simulate.py`
- `evict.py` with `run_arms.sh` and `monitor.py` for the runs
- `compare_states.py`

The runs used the public `run` entry with the numerical interpreter.

## Artifact disposition

Retained in `artifacts/qualification/prefetch/` of the producing worktree, about 6 MB:
- profile and timing JSON
- the scripts
- each arm's intent, report, memory record, journal and frozen source
- the refused first attempt, which lost the GPU lease race before allocation

Each arm's final 5.1 GB state and 0.5 GB GGUF export were retired through
`Store.retire_closed` after the bit-for-bit comparison, with receipts naming the
untouched parent as recovery. They were qualification outputs with an unretuned step
size, so exact resume from update 220,395 is not available. Parent state, models and
corpus were read only.

# The leader with only its shortlisted vocabulary rows is 26% smaller than Baseline and no slower

The one-bit leader `ef068eba…` was larger than Baseline as shipped: 542 MB against 381 MB,
because it carries its own 358 MB vocabulary table while Baseline borrows the target's.
Test-list item 11 asked whether the file can carry only the rows it scores, with nothing
else changed.

It can:
- the file is 284.0 MB (270.9 MiB), 25.5% smaller than Baseline's 381.5 MB, and the server
  uses 340 MiB less GPU memory than with the full table;
- every draft is identical to the full-table file's, on all 180 screen and panel requests;
- it passes the owner release policy against Baseline as released with no shortfall, at
  1.05–1.13 times Baseline's speed;
- against Baseline given the same shortlist it is 1.03 times as fast on ordinary requests
  and about 1.12 on agentic coding.

## Outcome

### File and memory

| File | Bytes | GPU memory after start |
| --- | ---: | ---: |
| Baseline | 381,456,064 | 6,037 MiB |
| Leader, full table | 542,373,856 | 6,221 MiB |
| Leader, full table with embedded shortlist | 542,649,216 | 6,317 MiB |
| Leader, shortlist rows only | 284,037,024 | 5,977 MiB |

The small file holds the body and 68,728 Q4_K rows, one per distinct TRAIN label of
`train-final-02-fused`. The list's estimate was 255 MiB from the 62,844 labels of the
earlier corpus. Rows outside the shortlist are made on demand from the target's Q6_K table:
all 248,320 rows, converted one at a time, equal the export's Q4_K rows byte for byte, so no
row is lost.

### Identical drafts

Runtime-v10 is runtime-v9 plus the edit that serves the small file. Per request, on the
60-request screen and the 120-request panel (greedy, depth 3):

- target, Baseline, Baseline with the shortlist, the leader and the leader with the embedded
  shortlist give the same tokens, drafts proposed and drafts accepted on both runtimes;
- the small file on runtime-v10 gives the same three as the leader with the embedded
  shortlist on runtime-v9.

### Speed on the panel

| Arm | Acceptance | Tokens/s | Drafting ms per round |
| --- | ---: | ---: | ---: |
| Baseline | 53.35% | 276.6 | 1.581 |
| Baseline with the shortlist | 53.33% | 288.9 | 1.187 |
| Leader, full table | 50.81% | 279.8 | 1.248 |
| Leader, full table with shortlist | 50.80% | 288.3 | 0.973 |
| Leader, shortlist rows only | 50.80% | 297.5 | 0.921 |

Paired ratios (source-group bootstrap, 95%): small file against the full-table shortlist
file 1.032 (1.030, 1.034), and 1.022 (1.018, 1.026) on the screen. Drafting time explains
only 0.05 ms of the roughly 0.27 ms per round this implies; the rest is not identified.

### Five release scenarios

Fresh four-arm timing on runtime-v10 at depth 3: target, Baseline as released, Baseline with
the same shortlist as a sidecar, and the small file. One arm order.

| Scenario | Small file / Baseline (95% lower) | Baseline with shortlist / Baseline | Small file / Baseline with shortlist (95%) | Acceptance: small file / Baseline |
| --- | --- | --- | --- | --- |
| Agentic coding (60 tasks) | 1.133 (1.116) | 1.016 | 1.115, ratio of the two | 81.7% / 63.4% |
| Greedy, long | 1.047 (merged strata) | 1.025 | 1.031 (1.022, 1.041), all greedy | 60.0% / 61.1% |
| Greedy, short | 1.083 (1.071) | 1.048 | as above | 49.6% / 52.3% |
| Sampled | 1.062 (1.054) | 1.035 | 1.026 (1.018, 1.034) | 47.7% / 50.0% |
| Thinking-on | 1.079 (1.069) | 1.050 | 1.027 (1.018, 1.037) | 63.0% / 66.6% |

The owner policy passes with merged strata and no shortfall; the leader's full-table file
passed on 2026-11-19 with two tolerated shortfalls. Every output was exact and agentic
completion was 37 of 60 for every arm. Acceptance equals the full-table leader's in every
scenario. No request ran out of request rows.

Two effects make up the gain over Baseline as released, and only one is ours:
- the shortlist itself, which Baseline gains too (1.016–1.050);
- the leader's cheaper round and, on agentic coding, its higher acceptance.

## Scope and limitations

- **One arm order and one run.** The owner skipped the reversed pass on 2026-11-20. On
  2026-11-19 and 2026-11-20 single-order and pooled figures agreed within 0.2%.
- **The like-for-like column** uses an unstratified source-group bootstrap, not the policy's
  stratified one, and the agentic figure is a ratio of two ratios without an interval.
- **Reused DEV data** and the tool-call convention of the training sessions, as for the
  leader. The protected confirmation sets and heldout24 were not touched.
- **The 3% over the full-table shortlist file is unexplained.** It repeats on the screen, the
  panel and the scenarios, but a round profile was not taken.
- **Shortlist coverage.** Tokens outside the TRAIN labels are scored only if they occur in
  the request; DEV shows no acceptance loss, other workloads are unmeasured.
- **Setup:** Qwen3.5-4B Q4KM, one RTX 5080, batch 1, native Linux, runtime-v10, depth 3.
  NVFP4 was not run. The small file needs runtime-v10 and the target's table at start-up.

## Decision

- **Retain** the shortlist-only output of `vocabulary-variants` and the runtime edit
  `vocabulary_shortlist_only_v1` ([benchmarks](../../docs/benchmarks.md#shortlist-only-vocabulary-table)).
- **No promotion.** `ef068eba…` stays the provisional acceptance leader; the small file is a
  serving form of it. Saved references stay on runtime-v8.
- **Proposed, not decided:** make the small file and runtime-v10 the release form of the
  leader; and a round profile to explain the 3%.
- **Unchanged:** the open question from the [one-bit attribution](../one-bit-attribution/result.md).
  A 4-bit drafter of Baseline's layout is 381 MB and gains the shortlist too.

## Evidence and reproduction

All paths are under `artifacts/shortlist-only-table-01/`. Compact measurements are in
[result.json](result.json), written by `scripts/assemble.py`.

**Build:** commits `dae6b42d` and `dc296a41` (merged as `dd7e45cc`), frozen source
`29115117…` in `builds/executed-sources/`. Runtime-v10 `builds/runtime-v10/` (`result.json`
`b7e096f2…`), quantizer `builds/quantizer/`, files and the all-rows proof in
`builds/variants/` (`drafter-shortlist-only.gguf` `156b4f72…`, `drafter-shortlist.gguf`
`5820b573…`, sidecar `7577676d…`, `row-independence.json`).

**Qualification:** `runs/v9-screen`, `v10-screen`, `v9-panel`, `v10-panel` from
`builds/run.sh`. The build's `summary.py parity` reports a panel failure because it compares
the first block by file name and the two runs split the panel into five and six blocks;
`scripts/parity_by_request.py` matches on the request and is the recorded check
(`logs/parity-panel-by-request.txt`).

**Timing:** `release-final/` (specifications from `scripts/make_chain.py`, runs, policy
reports for the small file and for Baseline with the shortlist). Sampled runs use a sampler
check bound to runtime-v10, `release-final/verifier-v10/`. Pilot slices in `release-pilot/`,
preflight in `logs/chain-preflight.md`.

**Commands:** `builds/run.sh`, `scripts/chain.sh`, `scripts/release.sh`.

## Artifact disposition

**Retained** (1.7 GB in all), owner Philip: the small file, the embedded-shortlist file and
the sidecar; runtime-v10 and the quantizer; both frozen sources (`b8963b34…` is an unused
earlier snapshot); every run, policy report and log. They are kept while the release form
is undecided, then the runs follow the retirement command.

**Dependencies:** the leader export `ef068eba…`, the target, the Baseline file,
`train-final-02-fused` labels, runtime-v9. None was changed.

**Retired:** nothing.

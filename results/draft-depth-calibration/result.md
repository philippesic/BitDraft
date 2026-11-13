# A three-minute calibration picks each drafter's measured best draft depth

BitDraft is meant to become a toolkit for converting models into low-bit drafters.
A trained drafter therefore cannot ship with a fixed draft depth, because the best depth
depends on the target, the GPU and the runtime. `python -m benchmarks calibrate-depth`
measures one deployment in 2–4 minutes of GPU time and recommends a serving depth
([how it works](../../docs/benchmarks.md#depth-calibration)). Calibration is separate
from training and from the shared benchmark depth.

On the Q4KM target with exact verification, it recommended the depth that measured fastest
on the 120-request panel for Baseline (3), the 220,095 HE4 state (2, with 3 inside 2%)
and arm D (3, with 2 inside 2%). The measured depth-3 to depth-2 speed ratios were 1.082,
0.992 and 1.013. The predicted ratios were 1.067, 0.997 and 1.016. Each calibration used 24
of the panel's 120 requests. Against the 60-request screen's six-depth sweep (depths 2–15),
the predicted decode-speed ratios fell within 1–3% of measured at every depth for Baseline
and HE4.

## Outcome

### How it predicts

One survival run drafts at depth 15 on a spread DEV subset. The server's per-position
counters give the share of rounds whose first *i* drafts were all accepted. Short cost
sessions at depths 0–15 give decode milliseconds per round. The model divides each deep
round's accepted run into rounds of the shallower depth to estimate tokens per round, then
divides by the round cost. The simpler approach, summing the depth-15 shares up to the
depth, undercounts accepted drafts per round at shallow depths by up to 14% on the
screen's measured sweep. This is because deep rounds start disproportionately at hard
tokens. The splitting model leaves tokens per round 3–6% low for Baseline and 1–3% low for
HE4. Because that error is similar across shallow depths, its ratios between depths stay
within 2%.

### Against the 120-request panel (Q4KM, exact)

Measured values are run means from [depth fairness](../draft-depth-fairness/result.md)
on the full panel. Predictions come from 24-request calibrations, with outputs capped at
512 tokens.

| Drafter | Calibration minutes | Recommended (within 2%) | Measured best | Complete d3/d2, measured / predicted | Decode d3/d2, measured / predicted |
| --- | ---: | --- | ---: | --- | --- |
| Baseline (repeat) | 2.7 | 3 (3) | 3 | 1.082 / 1.067 | 1.094 / 1.096 |
| Baseline (first run) | 2.8 | 3 (3) | 3 | 1.082 / 1.058 | 1.094 / 1.085 |
| HE4 220,095 | 3.5 | 2 (2, 3) | 2 | 0.992 / 0.997 | 0.987 / 0.996 |
| Arm D | 3.2 | 3 (2, 3) | 3 | 1.013 / 1.016 | 0.998 / 1.022 |

The panel measured only depths 2 and 3 in exact mode. The calibration also covers depths
0–15, and depths 4 and above were ranked below 3 for every drafter.

The panel's cohort and output caps differ from the calibration's, so absolute speeds are
compared on the same 24 requests of the panel runs. Decode tok/s, measured / predicted, at
depths 2 and 3:

- Baseline repeat: 251.0 / 246.8 and 274.8 / 270.5
- HE4: 207.6 / 205.7 and 203.5 / 205.0
- Arm D: 214.8 / 220.0 and 212.6 / 224.8

Predicted tokens per round were within about 2% at both depths for all three drafters.
Recomputed with the panel's full-length outputs, complete-request tok/s were within 3.5%,
and target-only within about 1%. The calibration's own complete-request prediction refers to its
512-token-capped requests, where prompt time weighs more. Its gain over target-only is
therefore lower than for full-length answers. For example, HE4 shows 0.96 on capped
requests, against 0.99 predicted and 1.02 measured with full-length outputs.

### Against the screen sweep (60 requests, depths 2–15)

These calibrations used 24 of the screen's requests at 128-token caps (1.8 and 2.2 minutes).
Decode tok/s relative to depth 2, measured / predicted:

| Depth | Baseline | HE4 220,095 |
| ---: | --- | --- |
| 3 | 1.100 / 1.111 | 1.012 / 1.028 |
| 4 | 0.926 / 0.913 | 0.823 / 0.810 |
| 6 | 0.897 / 0.886 | 0.765 / 0.764 |
| 8 | 0.806 / 0.784 | 0.660 / 0.641 |
| 15 | 0.621 / 0.614 | 0.486 / 0.472 |

Both drafters measured and predicted fastest at depth 3 on the decode clock. On the
complete-request clock, the single screen runs put depth 2 ahead by 0.1% (Baseline) and
0.8% (HE4). Those runs' prompt time varied by up to 2.4 s (14%) between runs, about 5% of
request time. The complete clock cannot resolve these depths. The calibration recommends
3 for both, with HE4's depth 2 inside 2%.

### Cost curve and what drives the recommendation

Round cost from the repeat Baseline calibration, in milliseconds per round
(drafter forward / remainder). HE4 and arm D have the same steps with a drafter about
0.3 ms cheaper.

| Depth | 0 | 1 | 2 | 3 | 4 | 6 | 7 | 8 | 11 | 12 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms per round | 5.36 | 8.44 | 8.88 | 9.38 | 12.21 | 13.80 | 13.72 | 16.42 | 18.07 | 20.75 | 22.51 |

The remainder rises in steps after depths 3, 7 and 11: verification of 4, 8 and 12 tokens.
This matches the four-column grouping of exact mode found in the
[consistency study](../cuda-decode-consistency-cost/result.md). The drafter forward grows
from 1.4–1.9 ms at depth 1 to 5.1–5.3 ms at depth 15, so it is not negligible. Within a
cost step, deeper is free. Across a step, depth 4 must add more than 30% to tokens per round
to pay. Survival on the panel subset was 0.71, 0.44, 0.26, 0.18 for Baseline's first four
positions, 0.48, 0.23, 0.13, 0.09 for HE4 and 0.55, 0.29, 0.17, 0.11 for arm D. None comes
close to paying for depth 4.

Per-category best depths differ: Baseline code favours depth 7, HE4 code and documents
favour 3. They rest on 15–1,000 drafted rounds per category, too few for decisions. The
record reports them as indications.

## Scope and limitations

- One RTX 5080, runtime v5, Qwen3.5-4B Q4KM target, F16 KV, context 32,768, greedy,
  concurrency one, exact mode. NVFP4, mode off, other GPUs and other runtimes were not
  calibrated. The command supports mode off (`exact: false`), but it is unvalidated.
- **Repeatability.** The first Baseline calibration ran while another agent's process kept
  a CPU core busy. Its round costs were 2–8% higher at every depth, including target-only,
  and its prompt overhead was 12% higher. Its recommendation was unchanged and its ratios
  between depths moved by about 1%. The survival shares were identical in both runs, because exact greedy
  output is deterministic. Absolute speeds should be read with about ±5% between sessions.
  A calibration does not detect contention. The target-only round cost (depth 0) is the
  quickest sign of a disturbed run.
- **Residual acceptance bias.** Deep rounds miss the benefit that a shallow round gets
  from restarting closer to a hard token. On the screen calibration, predicted tokens per
  round at depths 2–8 ran 4–6% low for Baseline and 2–3% low for HE4. On the panel subset
  they were within about 2%. The short cost sessions could correct this from paired same-text rounds,
  but with 4–8 requests that correction added ±3–4% noise to depth ratios. It is not
  applied.
- **Round cost depends on context.** The cost sessions use 4 requests chosen across input
  lengths, not the token-weighted mix of the workload. Absolute decode predictions differed
  from same-subset measurements by −2% to +6% (arm D at depth 3 is the largest). The panel
  itself recorded arm D's exact-mode rounds as 4–5% dearer than the 220,095 state's, an
  unexplained gap. The calibration found the two equal within 1%.
- **Prompt overhead** is taken from the depth-15 survival run. Per-request overhead in the
  cost sessions rises about 4% from depth 2 to depth 15, which makes shallow-depth
  complete predictions about 1% low. It does not change the ranking, because the overhead
  is nearly the same at every depth.
- **Near ties stay near ties.** HE4 and arm D have depths 2 and 3 within 2%. The panel's
  measured differences (0.9% and 1.3%) were inside its run-to-run spread. The record
  lists depths within 2% of the best for this reason.
- Calibration does not check exactness. That remains the benchmark's role.
  Drafts truncated at the end of capped cost requests could make deep round costs slightly
  optimistic. The survival run's depth-15 round cost (512-token caps, 24 requests) agreed
  with the cost session's (256-token caps, 4 requests) within 2% in every calibration.
- The validation reuses DEV cohorts: 24 requests drawn from the very panels used as ground
  truth. Measured values were not rerun here. The panel's references are single or double
  runs with their own 0.4–3.5% spread.

## Decision

- Use `calibrate-depth` to choose a drafter's serving depth for a new target, runtime or
  GPU, and quote the record's `near_best` list alongside the recommendation.
- On today's runtime it confirms depth 3 as the fair shared benchmark depth proposed by
  [depth fairness](../draft-depth-fairness/result.md). It recommends 3 for Baseline and arm
  D and lists 3 within 2% of the best for HE4. That default and the saved references are
  unchanged here; the decision waits for the runtime fix.
- After any runtime change to verification cost, rerun the calibration before re-deciding
  the shared depth. A clean target-only cost at depth 0 shows that the session was
  undisturbed.

## Evidence and reproduction

[result.json](result.json) bundles all six calibration records: identities, settings,
cost curves, survival shares, predictions and per-category results, each with the SHA-256
of its `calibration.json`. It also holds the comparisons with the panel and screen
measurements and the same-subset recomputations.
[executed-analysis.txt](executed-analysis.txt) contains the three comparison scripts
(`compare.py`, `subset.py`, `bundle.py`), run with the small check environment from the
worktree root. [spec-example.json](spec-example.json) is the HE4 panel specification.

```sh
uv run --locked --no-sync --offline python -m benchmarks calibrate-depth SPEC [OUTPUT]
```

Inputs, read-only from the main checkout through temporary links in the executing
worktree's `artifacts/`:

| Input | Path | SHA-256 |
| --- | --- | --- |
| Panel specification | `qualification/lean-checks/q4km/requests-establish.json` | `f61916f6…` |
| Screen specification | `qualification/lean-checks/q4km/screen-establish.json` | `f355e334…` |
| Target | `models/qwen35-4b-q4km.gguf` | `3874209…` |
| Baseline | `models/dflash-4b-q4km.gguf` | `83586a63…` |
| HE4 220,095 | `qualification/original-zero-export-02/exports/99d2be79….bin` | `99d2be79…` |
| Arm D | `training/recipe-test-arm-d-01/exports/6a07cd5f….bin` | `6a07cd5f…` |
| Runtime | `native/runtime-v5` server | `c83a7965…` |

All six runs executed the frozen source `53d6e89a…` on 2026-11-13 under the project lease.
They waited for other agents' leases without interrupting them. GPU use was 16.5 minutes:
six calibrations (2.7, 2.8, 3.5, 3.2, 1.8 and 2.2 minutes) plus two smoke runs. The first
smoke run exposed an off-by-one in counter pairing, which was fixed before any recorded run.

## Artifact disposition

- **Raw outputs, disposable after review.** The six calibration directories (8.7 MB: per-session
  server logs, per-request rows, owner and library receipts, `calibration.json`) and the
  frozen executed source are in the executing worktree's ignored
  `artifacts/benchmarks/depth-calibration/`. result.json holds every `calibration.json`
  in full with its hash, and the comparisons need nothing else. Server logs and
  per-request rows are lost when the worktree is removed. The two smoke runs' outputs
  were deleted after the fix.
- **Read-only inputs.** No models, saved references, runtimes or benchmark defaults were
  created, changed or deleted. The temporary links into the main checkout are removed
  with the worktree.

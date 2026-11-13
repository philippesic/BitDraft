# The final HE4broad state beats target-only on requests but not Baseline or live tasks

The current HE4 checkpoint is the final 220,095-update state of the November 10 run on
15,647 teacher captures. It is the starting point for production continuation. On the
broad Q4KM DEV request panel it is about 7% faster than the target alone, while Baseline
is about 27% faster. On the live tool tasks no drafter, including Baseline, beat
target-only. This supersedes the forty-sequence November 8 run (F004) as HE4's standing.

## Outcome

Every arm used Qwen3.5-4B Q4_K_M on the RTX 5080 with greedy decoding, F16 KV and draft
depth 2, so each verification round proposes at most two tokens. Every output matched
target-only tokens and stopping. A round's emitted tokens are its accepted drafts plus one
target token; target-only emits one token per decoding pass.

| 600 DEV prompts × 2 repeats | Target-only | Baseline | HE4 220,095 |
| --- | ---: | ---: | ---: |
| Acceptance | — | 65.77% | 41.85% |
| Accepted drafts per round (at most 2) | — | 1.31 | 0.84 |
| Tokens emitted per round | 1 | 2.31 | 1.84 |
| First draft accepted | — | 76.9% of rounds | 54.2% of rounds |
| Both drafts accepted | — | 54.4% of rounds | 29.4% of rounds |
| Request tokens/s | 168.3 | 213.5 | 180.9 |
| Speed relative to target-only | 1.00 | 1.27 | 1.07 |

HE4/Baseline speed was 0.847 (paired 95% interval 0.842–0.852) and acceptance was 23.9
points lower (23.5–24.4). HE4's gain over target-only is concentrated in code (236.9 vs
176.8 tok/s) and reasoning (209.1 vs 185.5). It was roughly level in conversation and
structured output, and slower in documents (120.7 vs 126.0) and writing (177.3 vs 181.5).
At 8,192–16,384 input tokens target-only was fastest: 94.8, Baseline 90.2, HE4 81.1.
Drafters also spend more time processing the prompt. The native server spent 0.18 ms
per prompt token with either drafter, against 0.11 ms without one.

| Tool-prefix harness, 180 requests | Target-only | Baseline | HE4 220,095 |
| --- | ---: | ---: | ---: |
| Acceptance | — | 82.36% | 62.76% |
| Accepted drafts per round | — | 1.65 | 1.25 |
| Request tokens/s | 152.5 | 186.3 | 171.7 |

| Live tool tasks, 12 tasks × 3 repeats | Target-only | Baseline | HE4 220,095 |
| --- | ---: | ---: | ---: |
| Completed | 24 / 36 | 24 / 36 | 24 / 36 |
| Seconds per assigned task | 7.40 | 7.49 | 7.99 |
| Client model seconds | 257.1 | 259.3 | 278.1 |
| Native prompt processing seconds | 39.7 | 82.2 | 82.0 |
| Native token prediction seconds | 196.9 | 128.7 | 149.5 |

Each live arm made 363 model calls and evaluated 233,682 new prompt tokens to generate
33,402 tokens. Both drafters cut prediction time, but each doubled prompt processing.
Baseline therefore only broke even, and HE4 was about 8% slower than target-only. In the
12-task lean panel, with a three-update descendant, Baseline was 0.6% faster than
target-only and HE4 9% slower. Baseline's sign flips between the two panels, so its
difference from target-only is smaller than the observed variation.

The run's own three-prompt training panel (five timing repetitions) showed the same
ordering. The final state reached 46.94% acceptance and 199.5 tok/s, against 228.4 for
Baseline and 174.4 for target-only. The seven-update-hour state was the best observed
at 48.85% and 222.1 tok/s, against 248.7 and 184.4. The 120-request lean panel for a
three-update descendant agrees ([checks](../reusable-acceptance-checks/result.md)):
40.8% and 174.5 tok/s on Q4KM; 40.4% and 181.1 on NVFP4.

A 2026-11-12 rerun of this state itself on that 120-request Q4KM panel, candidate-only
against the saved controls, gave 40.85% acceptance (20,384 of 49,899), 0.82 accepted drafts
per round and 175.3 tok/s: 1.06× the saved target-only record (164.6) and 0.86× Baseline
(204.9, 63.87%). Its tokens and counters equal the saved record of the same weights
exactly; speed differs by 0.3%. The server spent 34.1 s processing prompts and 212.2 s
predicting tokens within 259.5 s of request time.

| 120 requests, HE4 rerun | Acceptance | Tok/s | Target-only | Baseline | Prompt s | Prediction s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Code, 30 | 75.8% | 224.7 | 162.9 | 241.7 | 4.1 | 23.4 |
| Conversation, 20 | 35.9% | 178.7 | 172.6 | 215.0 | 4.2 | 48.2 |
| Documents, 25 | 45.7% | 105.3 | 117.5 | 121.3 | 21.3 | 27.9 |
| Reasoning, 20 | 37.3% | 203.8 | 184.9 | 244.8 | 1.1 | 71.9 |
| Structured, 15 | 59.1% | 135.6 | 122.2 | 129.5 | 2.7 | 4.8 |
| Writing, 10 | 27.2% | 180.9 | 180.7 | 222.7 | 0.6 | 35.9 |
| Input < 512, 60 | 40.9% | 206.0 | 180.1 | 243.8 | 4.0 | 132.5 |
| 512–2,047, 36 | 37.7% | 172.0 | 165.4 | 200.7 | 6.8 | 51.3 |
| 2,048–8,191, 18 | 49.2% | 118.1 | 125.4 | 132.5 | 13.5 | 20.4 |
| 8,192–16,384, 6 | 40.8% | 69.4 | 87.1 | 80.9 | 9.9 | 8.0 |

Target-only and Baseline columns are their saved tokens/s, so these speed comparisons
are descriptive rather than paired timing.

## Scope and limitations

These are reused DEV prompts, already consumed for selection; they are not unseen
confirmation or measured traffic. There is one state, one target precision for the
full panels, one GPU and one greedy pass per repeat. Request speed intervals pair
requests within one execution. The live panel has eight source groups, too few for an
interval, and its 12 tasks come from one project. Prompt and prediction seconds are
native server partitions, distinct from the whole-request or whole-task clocks.
Per-position rates come from cumulative server statistics, including two warmups per
server. The NVFP4 full panels were stopped before completion and are not used.

The training panels used three prompts and cannot rank states. Whether the seven-hour
state would also lead on the broad panel is unmeasured.

## Decision

Retained as HE4's current standing. Baseline remains the incumbent, with no promotion.
Proposed, not decided: drafter evaluations should report prompt processing separately,
because a drafter that wins decoding can still lose tool workflows where new context
dominates. Request scores cannot override that live result.

## Evidence and reproduction

[Compact measurements](result.json) record every arm, the paired statistics, category
and input-length strata, live timing partitions and the identity of each source. Sources:

- Full Q4KM DEV panels: `/home/philip/BitDraft/artifacts/benchmarks/full-qualification-01/`
  (`q4km-requests`, `q4km-harness`, `q4km-live`), specification
  `artifacts/benchmarks/full-qualification-spec.json`. The operator stopped the suite
  during NVFP4; its `operator-stop.json` is retained.
- The 120-request rerun used spec `lean-checks/q4km-requests-cached-spec.json` with its
  candidate replaced by the `original-zero-export-02` export; source `a7747d24…`. Its output
  is `artifacts/benchmarks/he4-220095-q4km-requests-01/` in the worktree that ran it;
  `lean_requests_rerun` in [result.json](result.json) holds its compact values.
- The HE4 arm is `original-zero-export-02`, a zero-update re-export of the original final
  state. `artifacts/qualification/original-export-canonical-equivalence.json` confirms
  its tensors match the original final export.
- The training-run panels are in the run's closeout summary and report,
  `artifacts/datasets/retained-he4broad-original/closeout-reports/final-summary.json`
  and `final-report.md`, also preserved in
  `artifacts/archive/legacy-closeout-20261112/compact-support.tar.gz`.

Rounds and per-round values are derived as emitted − accepted − requests. On these panels
this agreed with the native round counters within 0.1%. Per-position acceptance comes
from the last `#acc rate/pos` statistics line of each server log. Timing partitions sum
`prompt_ms` and `predicted_ms` from the stored responses and transcripts. No GPU work was
rerun for this result; it extracts and recomputes the stored records.

## Artifact disposition

Retain `full-qualification-01`. Saved reference registrations were built from its Q4KM
panels and authenticate those raw records. The original 220,095 state, its re-export and
the compact archive also stay. The earlier ad hoc breakdowns in `artifacts/qualification/`
(`benchmark-runtime-breakdown-q4km.json`, `live-q4km-*-breakdown.json`) match these
values and are superseded by `result.json`. Nothing was deleted.
The 5.8 MB rerun output was not copied into the main checkout. Its compact values are
in `result.json`, and the retained export and saved controls can reproduce it.

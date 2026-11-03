# Ordered generalization progress — November 3, 2026

This supersedes the older report's current-state paragraphs, preserving them as
history. Phase 1 remains incomplete. There is now a repeatable development-set
throughput gain, but no untouched generalization or independent quality result.
The quick-screen predictor, backlog screening, combinations and final four-target
DFlash confirmation have not started. The user requested a full update and a wait;
no new experiment was launched during that check-in. GPU compute is idle and the
owned runtime tmux socket has no server.

## Historical controls

Two historical Qwen3 F16 arms each completed 1,920 updates and 600 native measured
requests. Legacy and occupancy-only treatments show no reliable improvement over
their unchanged parent and remain about 8% below their historical Q4 comparator.
These are separate from modern Qwen3.5 results. The first interrupted historical
training lost its final state; verified periodic checkpointing was added before
rerunning. Paid successful checkpoints survive. Two full-state retention arms
remain unrun because of storage capacity.

## Modern data and training

Native serving and conditional capture differ: a 55-token prompt is processed as
51+4 by serving; the last four feature rows differ by 8.58% relative RMS. Actual
serving-state collection covers 24 fit and 12 development groups, with 356/193
actual emitted draft successes. Source-indexed data now supplies 2,138 windows
(1,414 fit, 724 development) without copying full prefixes. All native success
masks and unsupported labels remain in evaluation. Development has historical
F16 TRAIN exposure and is not pristine final holdout.

A compact scale-only trainer updates 76,880 parameters in nine packed projections;
signs and norms remain frozen. Native exports, optimizer/RNG/cursor restoration
and periodic verified checkpoints work. The broader zero-update numeric gate
passes: 64/2,646 native-reachable decisions differ (2.42%), with prefix mean error
0.00889; every native protection remains in evaluation. Training protects the
intersection reachable by the unchanged GPU actor, with margins from that actor.

Matched CE and retention-weight-0.1 arms reached 480 and then 1,920 updates from
the same initializer and sample order. A retention-weight-10 arm reached 1,920
from the same initializer. Development model-side accepted-prefix totals rise
from 190 to 398/402/405 respectively. Previously successful native/actor prefix
tokens lost are 40/37/35. Preservation therefore remains unresolved. These are
fixed-state model readouts, not native throughput or untouched confirmation.
All final checkpoints and intermediate continuation states remain available.

## Measured native development result

At 480 updates, five arms × 12 development prompts × five repetitions produced
300 measured full requests, plus 50 excluded warmups. All token IDs, text,
stopping and repeated outputs match. Root independently audited raw responses,
25 server releases and 40 loaded-image pins.

| Arm | Full-request tokens/s |
| --- | ---: |
| Frozen DFlash Q4_K_M | 194.895 |
| Target-only | 152.903 |
| W1A8 initializer | 109.652 |
| W1A8 CE, 480 updates | 114.020 |
| W1A8 retention, 480 updates | 114.313 |

CE improves 3.98% over initialization; paired whole-prompt 95% interval is
1.56–6.35%. Retention improves 4.25%, interval 1.61–6.75%. Both remain only about
59% of DFlash throughput. Native accepted drafts increase from 193 to 228 per
repetition. This 80-token-limit panel has a different denominator from the older
235.9 tokens/s DFlash benchmark; the older rate is not the matched comparator here.

## Isolated runtime findings

A cross-microbatch input alias overwrites an earlier EAGLE bridge feature before
batch materialization. An isolated ownership fix compiles and passes three reused
full-response checks; the intended feature source is corrected. It is not adopted
into the training lineage or combined benchmark.

The EAGLE graph also computes its large output head when no logits are requested.
An isolated zero-output head skip compiles and preserves three full responses and
acceptance counts. On one complete trace, noncandidate native state arrays are
byte-exact and all 111 selected candidate sets/logits agree after alignment by ID.
Candidate-buffer order alone is unordered. Repeated timing has not been run.

An eight-arm 1,920-update/head-skip comparison is prepared, not launched. It
separates training and runtime effects with matched unchanged controls; the bridge
fix is separate. CE at 1,920 is not rejected and has not had native timing.

## Confirmation tooling, operations and next gate

A fresh-input generator/judge adapter is implemented with eight passing synthetic
tests, per worker report. It requires model/runtime/source selection to be frozen
before generating 24 performance and 24 independent quality cases. No real fresh
cases have been generated or evaluated; root integration/review remains.
The backlog inventory is 98 segment IDs in 34 families, including 18 controls and
80 treatments/continuations, not 98 independent mechanisms. No validated screening
labels have been assigned.

Storage failures were handled without deleting unique paid state. Exact duplicate
caches reclaimed 982,757,376 WSL bytes; a later verified owned-cache relocation
reclaimed 684,920,832 WSL bytes while consuming 707,977,216 C: bytes. Current free
space is approximately 2.15 GiB WSL and 2.37 GiB C:. Full-state capacity remains
unresolved; compact training fits. No push occurred. Current trainer/source work
is preserved in the isolated actual-training worktree; newer serving/fresh-input
changes there are not integrated yet. The canceled generalization-audit worker
has no recovered final report; its unavailable thread was not counted as completed.

Next experiment after the user's wait: native repeated timing for the longer
checkpoints and isolated head skip, followed by frozen selection and untouched
token/quality/five-repeat confirmation. Phase 2 requires prospective short-to-long
predictive validation, beyond the current two checkpoints. Phase 3 stays gated.

## Evidence

All paths below are relative to the repository and kept outside Git:

- `runs/w1a8-modern-actual-training-data-20261103-01/`: data report, source audit,
  bridge alias and serving-cost source audits.
- `runs/w1a8-modern-actual-training-20261103-01/`: reference02, ce01,
  retention01, ce1920-01, retention1920-01, retention10-1920-01;
  `native-analysis480.json`, `actor-analysis480.json`, `requests480-02/result.json`,
  and prepared `models1920-headskip.json`.
- `runs/w1a8-modern-{bridge-fix,headskip}-20261103-01/`: separate builds,
  source patches, smoke responses and aligned parity receipts.
- `experiments/w1a8-generalization-20261102/report.md`: historical controls,
  failed attempt, recovery and storage evidence.

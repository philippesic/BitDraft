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

## Longer checkpoint and head-skip comparison completed

The eight-arm panel finishes exit0 in410.45s whole-job wall, with480 measured
and80excluded warmup requests across40startups. Root audits every raw measured
response, all40server releases and41loaded executable/library images. Tokens,
text, stopping and repeats agree exactly. Source/input/model pins and raw timings
are preserved in actual-training `requests1920-headskip01/` and root independent
`native-analysis1920-headskip.json` (20,000 paired whole-prompt resamples).

| Arm | Full-request tokens/s | Accepted/proposed per repetition |
| --- | ---: | ---: |
| Frozen DFlash Q4_K_M |203.060|542/756|
| Target-only |158.838|—|
| W1A8 initializer, original |112.484|193/1439|
| W1A8 initializer, head skip |112.967|193/1439|
| Retention0.1 at1920, original |127.337|309/1211|
| Retention0.1 at1920, head skip |127.821|309/1211|
| Retention10 at1920, head skip |127.948|309/1210|
| Ordinary EAGLE Q4_0, head skip |164.452|508/819|

Longer training improves original-runtime W1A8 throughput13.20%, paired95%
interval8.44–18.98%, over matched initialization. It remains62.71% of DFlash
and80.17% of target-only. Head-skip effects are inconclusive on these short
prompts: initializer+0.43%[-0.22,+1.14%], trained+0.38%[-0.56,+1.24%]. Do not
adopt it as a measured speed win. The stronger retention model's native speed and
accepted count are indistinguishable here. No untouched confirmation is claimed.
These controls were rerun in the same panel; older rates remain separate.

All owned server groups are absent and compute is released. New panel artifacts
consume12,083,659logical bytes. Current free bytes are2,274,824,192WSL and
2,542,604,288C: (before subsequent small source/checkpoint writes).

Next preservation hypothesis:59%of original two-window batches contain no
protected window; protected slots cover351/2828fit-depth slots. A separate
trainer will add deterministic group-balanced protected FIT replay every update,
with an explicitly normalized margin loss. Matched zero/positive penalty arms and
head-only optimizer scope will separate protection and representation effects.
Original trainer/source/checkpoint contracts remain immutable. Development labels
remain readout-only; all native protection failures remain visible.

Fresh generator review fixes the manifest receipt identity, supports prior prompt
wrappers and NULL-safe SQL with discriminating fixtures, and binds selected model
and complete runtime source contract. Nine synthetic CPU tests and Ruff pass.
No real cases are generated. The fresh serving adapter is being implemented
separately, sharing frozen runtime validation and native lifecycle methods.
A source refactor revalidates all560previous request/gold bodies without launching
GPU (`hook-refactor-input-check.json`). Source is committed in the retained
actual-training worktree; integration and further trainer/adapter work remain.

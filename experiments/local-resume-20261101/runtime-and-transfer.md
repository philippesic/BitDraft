# Local RTX 5080 runtime and training-transfer investigation

The isolated SM120 candidate preserves the conservative runtime's measured
scores while recovering substantial serving throughput in development. Final
confirmation is pending on newly generated inputs. The historical F16 training
track remains separate; no new training or old campaign restart occurred.

## Modern quantized-target runtime

The runtime baseline is the recovered October 31 opt-in
`GGML_CUDA_DECODE_CONSISTENT=32`, not the unmodified default. Four exact local
Qwen3.5 target files are retained: 4B/9B Q4_K_M and 4B/9B NVFP4, all with F16
K/V. Source is modern llama.cpp `792fcf4e`, distinct from the parent's `5eb221ec` pin.
No parent gitlink changed and nothing was pushed.

The [candidate patch](patches/qwen35-sm120-consistent-groups.patch) retains the
conservative fusion disable, stable one-partition F16 attention and scalar
floating-projection paths. It restores two-query-column vector attention, then
shares quantized weight reads across groups of up to four query columns, using
SM120's same four-warp reduction and Q8_1 activation staging. The group change
is restricted to contiguous Q4_K/Q5_K/Q6_K/NVFP4 projections with K >= 2560 on
SM120. It retains the scalar fallback elsewhere. It does not claim native FP4
activation computation, SM75, GB10, MoE or other-backend validation.

The native experiment's private candidate commit is recorded in
`runs/upstream-consistency-20261101/quant4group-source-receipt.json`. Immutable
CUDA-only overlays hold every other runtime library fixed. The source and
library receipts, exact commands and process maps are in that run directory.

Fresh native checks:

- Conservative reference and two-column-attention ablation each passed 4,272
  score/state comparisons across all four targets, with zero measured drift.
- The grouped-column candidate passed 5,496 comparisons, also zero drift and
  no greedy flips, including widths 2/3/4/5/7/8/16/17/32, partial rejection and
  continued decoding. The native test's score gate is 1e-5; observed delta was 0.
- The actually executed unmodified negative control failed: 53 greedy flips in
  687 comparisons. No preparation check is relabeled as CUDA execution.
- All 240 measured development requests compare exactly across their matched
  target-only/speculative arms. These are five reused development prompts,
  two alternating repetitions; they are not final confirmation.

Development full-request rates, with MTP maximum 3 in each row:

| Target file | Conservative target-only tok/s | Conservative MTP tok/s | Candidate MTP tok/s |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B-Q4_K_M.gguf | 176.73 | 167.08 | 234.71 |
| qwen35-4b-mtp-nvfp4.gguf | 186.51 | 188.13 | 248.72 |
| Qwen3.5-9B-Q4_K_M.gguf | 120.95 | 126.10 | 193.02 |
| qwen35-9b-mtp-nvfp4.gguf | 127.31 | 130.91 | 197.54 |

The separate 9B DFlash maximum 16 screen also matches all tokens, but its low
acceptance wastes verification work. Grouping raises its rate substantially;
MTP is faster on this development panel, so MTP3 is frozen for confirmation.
Raw metrics: `development-stock-01/` and `development-9b-mtp-01/` under the run
root. Counts are returned output token IDs divided by full HTTP request wall;
loading, excluded warmups and metric requests are separate.

### Correctness boundary and independent quality

A new 80-request probe compares the original and conservative **target-only**
runtimes. Both reproduce themselves, but 15 of 20 distinct prompt/model pairs
change output; their second repetitions reproduce the same differences. The
old patched-versus-patched speculation tests therefore do not prove transparent
equivalence to default unmodified decoding. The final claim is explicitly
against the fixed conservative arithmetic and target-only reference.

The independent development judge exercises JSON arithmetic/logic/schema,
Python edge cases and an in-memory SQLite query. It accepts all eight correct
controls and actually rejects all eight known-wrong controls. A v1 restriction
incorrectly rejected ordinary lambda keys; v1 and its scores are preserved,
and v2 fixes that restriction before final freeze. Rejudging all retained
outputs yields unchanged scores for original/conservative/candidate: 4/8 for
4B Q4_K_M, 3/8 for 4B NVFP4, 4/8 for 9B Q4_K_M, 5/8 for 9B NVFP4. Baseline arithmetic
and format failures are real observations, not hidden as runtime success.
The manual thinking-off prompt text matches the GGUF's canonical template in
an independent CPU Jinja rendering; a server tokenization cross-check remains
an optional diagnostic. These small tasks are not general model-quality proof.

The candidate/settings/evaluator freeze and generated confirmation manifest are
`final-freeze.json` and `confirmation-inputs/receipt.json`. New 24 performance
prompts and 24 quality tasks were generated after freeze; all 24 correct and 24
known-wrong quality controls were executed. Final requests use five matched
repetitions and compare all tokens/text/stopping behavior and repetition
identity; native metrics must agree with response draft counters. Source or
selection changes require a separately declared confirmation. The frozen run
started locally at 09:54 UTC: independent quality first, then 1,920 full requests.
Its process identity and stop procedure are recorded in
`runs/local-confirmation-20261101-01/identity.json`; final results are pending.

## Historical F16-target training transfer

This uses the unchanged historical Qwen3 F16 target/KV and DSpark D5/body7/full
Markov runtime 0.25.1, not the modern Qwen3.5 runtime 0.28.0. The comparator here
is the frozen same-family Q4_K_M drafter; it is not Q4_0 EAGLE. No W1A1 win over
the primary Q4_0 EAGLE baseline is established.

A fresh diagnostic reproduces all 72 request token sequences and counters.
Synchronized CPU wall spans show fresh_best's draft work is cheaper per round
than Q4, but it needs 765 verification rounds versus 670. Additional target
verification accounts for 1.173 of the 1.282-second diagnostic round-time gap.
These are diagnostic CPU wall spans with explicit synchronization, not CUDA
kernel percentages or a replacement for clean full-request timing.

On 12 preselected real TRAIN-history states, native replay reproduces the
parent's actual accepted-prefix count 12/12. Fresh improves fixed-state prefix5
from 31 to 39, while Q4 scores 44. This bounded replay does not establish fidelity
for every history or explain the serving loss by itself.

A stronger actual-frontier join validates all 72 measured requests, six excluded
warmups, 2,198 complete rounds and 16 no-proposal emissions. On 549 common,
nonterminal positions with full five-token proposals, fresh accepts 1,420 versus
parent 1,445. It gains 67 tokens on 367 parent-unsaturated positions but loses 92
on 182 formerly full-five positions; 32 previously successful positions are
harmed. First-token survival improves slightly, then depths 3–5 lose. Fixed
equal-window scoring also underweights prose's share of actual serving rounds.
Common output prefixes do not prove identical hidden states.

The deterministic selection retains all 32 damaged frontiers plus 32 unchanged
and 32 improved controls. An identical-feature/history counterfactual capture
and replay has completed; CPU analysis is in progress to distinguish a model/data/objective tradeoff
from state or numerical differences. Capture startup failures and added capture
constraints are preserved; no failed capture is treated as valid data. The capture reproduces all 48 requests, 1,444 ordinary round decisions and
1,433 complete forced-round records from the original diagnostic before any
features are used. Both models then replay the exact same captured features,
chunk sequence and seed for all 96 selected states.

The most justified direction is preserving successful full-prefix behavior
while improving difficult states, with serving-occupancy and continuation-depth
weighting. Blind continuation and isolated head/kernel optimization do not
address the measured penalty. Selection causality is not yet proven.

## Costs and current limits

New training cost is 0 updates / 0 trainer-seconds. The earlier paid checkpoints,
failed receipts and allocations are preserved. Do not add historical fresh and
aggregate trainer clocks without their original accounting join.

The first full CUDA candidate compilation measured 585.486 seconds; configure
reported 10.9 seconds. The second incremental build passed but lacked a complete
wall timer. Source preparation, hashing, CPU checks, export and compile are
separate from native/process diagnostic wall, HTTP request evaluation wall,
server startup and excluded warmups. Individual receipts retain measured costs;
a complete total preparation ledger is not claimed. Native process walls
include model loads and capture, not only GPU compute. Final totals are pending.

Reports and reproducible source audits: [upstream investigation](../upstream-cuda-consistency-2026-11-01/report.md),
[modern evidence](modern-evidence.md), [training-transfer audit](transfer-diagnosis.md).
Large models, raw arrays and logs remain outside Git under
`runs/upstream-consistency-20261101/`; current ownership/stop procedures are in
[the active checkpoint](../../docs/goals/nine-model-qat-overnight.md).

# Local RTX 5080 runtime and training-transfer investigation

The isolated SM120 candidate delivers a validated 32.0–51.0% full-request
throughput improvement over the conservative MTP runtime across all four pinned
Qwen3.5 targets. All 1,920 final requests preserve matched target-only output;
independent quality outputs and scores also match. This is a bounded opt-in
runtime result, not a W1A1/Q4_0 EAGLE win or equivalence to default arithmetic.

The separate historical F16 investigation explains the serving penalty: gains on
difficult states coexist with damage to previously successful continuations,
causing extra verifier work. No new training or old campaign restart occurred.

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

The upstream leads were checked against actual source and local configurations.
Apoze's quantized-KV flash-attention fix does not replace the local **F16-KV**
workaround. Its earlier F16 matrix-dispatch correction is useful reference
material for the older parent pin, not a full fix for the modern mixed-weight
models. Danbev's GB10 warp edit targets SM121; it does not act on RTX 5080 SM120.
The generic SM120 reduction already uses four warps for one through four MMVQ
columns. That source fact motivates grouping up to four columns while retaining
the tested reduction topology. Exact upstream sources, ports, objections and
CPU checks are in the [upstream audit](../upstream-cuda-consistency-2026-11-01/report.md).
Reference ports and logging changes were not enabled in the final runtime.

Fresh native checks compare serial and batched decoding **within each runtime**;
they do not establish full-logit equality between candidate and conservative:

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

### Final held-out confirmation

Source, settings and evaluators were frozen before generating 24 new request
prompts and 24 quality cases. The performance panel uses five alternating-order
repetitions, fresh server per cell, two excluded warmups, concurrency one,
context 4096, batch/ubatch 512, F16 KV, flash attention, temperature zero, seed
1234, output limit 256 and disabled prompt caching. Each cell has 120 measured
requests. Rates divide returned output-token count by full HTTP request wall,
including prompt processing and decoding; model loading and warmups are separate.

| Target | Conservative target-only tok/s | Conservative MTP3 tok/s | Candidate target-only tok/s | Candidate MTP3 tok/s | Gain over conservative MTP3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B Q4_K_M | 175.32 | 173.18 | 175.62 | 244.05 | +40.9% |
| 4B NVFP4 | 183.38 | 188.68 | 183.67 | 248.99 | +32.0% |
| 9B Q4_K_M | 117.57 | 120.58 | 118.43 | 182.11 | +51.0% |
| 9B NVFP4 | 127.25 | 127.31 | 127.66 | 191.70 | +50.6% |

| Target | Candidate MTP3 repetition range tok/s | Gain 95% paired prompt-cluster interval |
| --- | ---: | ---: |
| 4B Q4_K_M | 242.96–245.81 | +39.2–42.4% |
| 4B NVFP4 | 247.45–249.90 | +30.6–33.2% |
| 9B Q4_K_M | 181.36–182.60 | +49.1–52.8% |
| 9B NVFP4 | 191.06–192.08 | +48.8–52.2% |

Candidate MTP3 also beats its own target-only reference by 39.0%, 35.6%, 53.8%
and 50.2%, respectively. Every matched repetition improves and all 24 prompt
clusters improve when averaged across repetitions. The 20,000-resample paired
bootstrap keeps all repetitions of each prompt together; intervals describe
this panel, not production capacity or variation across independent sessions.
The new inputs are parameterized tasks, not proof of training-corpus novelty.

All 1,920 responses match same-repetition conservative target-only token IDs,
text and stop state, and reproduce across all five repetitions. There are zero
counter mismatches or truncated responses. CUDA process maps identify the intended
immutable library overlays. Accepted/proposed/round counts are identical across
the two speculative runtimes: this is cheaper computation at unchanged acceptance.
The speedup combines restored two-column attention and grouped quantized
projections; their individual contributions and hardware memory traffic were not
measured. Fusion remains globally disabled and attention still uses one partition.

The final 288 independent quality requests produce identical candidate and
conservative outputs and scores on all 96 paired cases: 13/24 for 4B Q4_K_M,
9/24 for 4B NVFP4, 11/24 for 9B Q4_K_M, 14/24 for 9B NVFP4 (47/96 total).
Original scores 46/96 and differs on nine outputs. Baseline failures remain
substantial. The judge independently checks JSON values/schema, Python edge
cases and SQL answers, with 24 correct and 24 known-wrong executed controls;
its restricted Python runner and eight parameterized task templates limit any
broad model-quality inference. It is not a general answer-quality certification.

[Recomputable summary](confirmation-summary.json) and
[analysis script](analyze_confirmation.py) retain cell rates, repeated rates,
checks, confidence intervals and immutable evidence hashes. Reproduce with:

```sh
.venv/bin/python -B experiments/local-resume-20261101/analyze_confirmation.py --verify
```

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
`final-freeze.json` and `confirmation-inputs/receipt.json`. The 24 performance
prompts and 24 quality tasks were generated after freeze; all 24 correct and 24
known-wrong quality controls were executed. Final requests completed five matched
repetitions with all tokens/text/stopping behavior, repetition identity and
metric/response draft counters checked. Source or
selection changes require a separately declared confirmation. The frozen run
completed locally at 10:53 UTC with supervisor exit zero. Its authenticated
process groups are absent and the RTX 5080 has no listed CUDA compute applications.
`runs/local-confirmation-20261101-01/{state,identity}.json` retains the receipt.

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
and replay tests a model/data/objective tradeoff against state or numerical
differences. Capture startup failures and added capture
constraints are preserved; no failed capture is treated as valid data. The capture reproduces all 48 requests, 1,444 ordinary round decisions and
1,433 complete forced-round records from the original diagnostic before any
features are used. Both models then replay the exact same captured features,
chunk sequence and seed for all 96 selected states.

The identical-input replay retains damage on 28 of 32 selected damaged states
(three ties, one reversal). Parent accepted-prefix fidelity is 94/96 and fresh
fidelity is 90/96 against their actual serving histories. Of the 30 damaged
states with faithful parent replay, 27 retain the loss and three tie. Across the
32 damaged states the actual parent/fresh totals are 160/68; identical-input
replay yields 154/71. All 32 improved controls retain an improvement. This supports
a learned tradeoff under fixed native inputs, while the residual replay mismatch
limits any claim that all state/numerical explanations have been excluded.

The [counterfactual summary](frontier-counterfactual-summary.json) records the
reproducible checks, source hashes and limitations.

The most justified next experiment crosses serving-history data with the
existing versus acceptance-weighted objective, at matched exposure and initialization.
Include ordinary successful full-five states alongside difficult states, split
original groups before capture, weight by observed serving occupancy and deeper
prefix survival, and retain a penalty for damaging successful continuations.
Select on development full-request benefit and confirm on untouched groups.
This discriminates data coverage from objective mismatch; the current evidence
does not identify either one uniquely.

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
server startup and excluded warmups. Final performance used 2,967.32 seconds of
measured HTTP requests, 107.59 seconds of excluded loading and 244.30 seconds of
excluded warmups; its phase wall was 3,356.27 seconds. Final quality used 126.53
HTTP seconds (162.43 seconds of phase wall including loading). Complete final
confirmation wall was 3,530.64 seconds, including integrity preparation and
harness overhead. These are nested clocks, not additive GPU-hour measurements.

The [cost ledger](cost-summary.json) separately records development evaluation,
native gates, historical captures/replays, CPU compilation and failed startup
preparation. A complete preparation total is not claimed because some manual
work and the incremental compile were not fully timed. Native process walls
include loads/capture. Prior closure records 8,516.06 aggregate trainer seconds
and separately 3,146.14 fresh trainer seconds; overlap is unresolved, so they
are not added or presented as new training.

Reports and reproducible source audits: [upstream investigation](../upstream-cuda-consistency-2026-11-01/report.md),
[modern evidence](modern-evidence.md), [training-transfer audit](transfer-diagnosis.md).
Large models, raw arrays and logs remain outside Git under
`runs/upstream-consistency-20261101/`; current ownership/stop procedures are in
[the active checkpoint](../../docs/goals/nine-model-qat-overnight.md).

The two clean integrated evidence-worker worktrees and branches were removed
only after checking patch equivalence, ignored files and empty submodules. The
source-audit worktree is retained because it contains pre-existing staged
submodule state. About 12.1 GB of source/cache/evidence remains, including useful
actual-history captures and replay arrays; no paid checkpoint was deleted.
WSL free space increased from 28,869,906,432 to 28,901,277,696 bytes during the
small checkout cleanup; no Windows volume reclamation is claimed. Bounded
retention reasons and exact paths are in the ignored `retained-inventory.json`.
All experiment supervisors/groups are terminal; Windows graphics memory remains
allocated, with no project CUDA compute application listed. No push occurred.

The requested runtime result and historical bottleneck diagnosis are complete
within these boundaries. Further generalization to longer contexts, concurrency,
other KV types, stochastic decoding or a new training recipe needs its own
comparison and fresh confirmation; none is implied by this result.

# W1A8 standing and selected DFlash baseline

DFlash Q4_K_M is now the primary baseline for the modern Qwen3.5 correctness-mode
track. **None of our W1A8 models has been measured against that baseline on those
four targets.** Our substantial historical W1A8 EAGLE and DSpark work used
Qwen3-4B with an F16 target. It has produced real training/native gains, but no
validated full-request win over its matched Q4 control. Historical results do
not establish the relative standing of an unbuilt modern W1A8 candidate.

This is the November 2 user-selected comparison policy. It supersedes the older
primary-Q4_0 instruction for the modern track. It does not change prior frozen
experiments, target precision, teacher ancestry or old denominators.

## Selected modern baseline

| Target | DFlash draft format | Frozen depth | Full HTTP request tok/s | W1A8 on same target/runtime |
| --- | --- | ---: | ---: | --- |
| Qwen3.5 4B Q4_K_M | Q4_K_M | 2 | 235.9 | Not measured |
| Qwen3.5 4B NVFP4 | Q4_K_M | 3 | 258.3 | Not measured |
| Qwen3.5 9B Q4_K_M | Q4_K_M | 3 | 200.8 | Not measured |
| Qwen3.5 9B NVFP4 | Q4_K_M | 3 | 209.2 | Not measured |

[Baseline pins/settings](baseline.json) select the existing held-out rows from
the [58-row matrix](../qwen35-correctness-matrix-20261101/matrix.csv), unchanged.
The same family/format uses the previously selected depth for each target.
Q8_0's small nominal lead on 4B Q4_K_M was unresolved against Q4_K_M and MTP;
the user selected Q4_K_M everywhere for simplicity, not a universal winner claim.

The platform is RTX 5080 with the retained quant4group CUDA overlay and opt-in
consistency mode32, F16 target/draft KV, context4096, batch/ubatch512, concurrency1,
greedy generation, no prompt cache and two excluded warmups per cell. Rates pool
returned token IDs over full HTTP request wall across five repetitions. They
are capped-output workload rates, not concurrent capacity or unrestricted-answer
latency. The independent quality limitation and exact output gates remain as
recorded in the [matrix report](../local-resume-20261101/runtime-and-transfer.md#qwen-correctness-mode-baseline-matrix--november-1-follow-up).

Target-only remains mandatory for correctness. Q4_0 EAGLE remains useful as an
architecture diagnostic. Historical Qwen3 F16/SM75 studies keep their original
primary control. Future W1A8 success requires a new matched comparison to the
selected DFlash baseline, with unchanged controls and fresh final confirmation;
these recorded rates are an anchor, not a substitute for rerunning controls.

## What the measured W1A8 models achieved

W1A8 means one-bit weights and eight-bit activations in the declared selected
projections. It does not mean the entire draft, attention or vocabulary head is
one-bit. It is distinct from W1A1, W1A16 and W8A8, and from ordinary GGUF Q4
weight storage. Both hardware and tensor coverage differ across lineages.

| Historical experiment | W1A8 request tok/s | Its paired Q4 request tok/s | W1A8 / paired Q4 |
| --- | ---: | ---: | ---: |
| Untrained EAGLE all-nine, RTX 2080 Ti, fixed D5 | 41.41 | 79.74, Q4_0 EAGLE | 51.9% |
| EAGLE fixed-A8 two-hour reference, RTX 5080 | 95.69 | 135.79, Q4_0 EAGLE | 70.5% |
| EAGLE learned-A8 + affine midpoints + low-inertia recipe, RTX 5080 | 69.62 | 134.73, Q4_0 EAGLE | 51.7% |
| EAGLE calibrated 14.43h snapshot, RTX 5080 | 92.85 | 143.87, Q4_0 EAGLE | 64.5% |
| DSpark parent3504, latest historical panel, RTX 5080 | 204.06 | 222.45, same-family Q4_K_M | 91.7% |
| DSpark earlier EAL checkpoint, same panel | 206.53 | 222.45, same-family Q4_K_M | 92.8% |
| DSpark fresh selected checkpoint, same panel | 202.65 | 222.45, same-family Q4_K_M | 91.1% |
| DSpark fresh early checkpoint, same panel | 194.19 | 222.45, same-family Q4_K_M | 87.3% |

All rows use historical Qwen3-4B F16 targets. **Do not divide these rates by the
modern DFlash rates.** Different target architecture, precision, runtime, input
lengths, draft settings and output caps make that ratio uninterpretable.

The EAGLE results come from the [untrained precision grid](../w1ax-policy-grid-results.md),
[two-hour comparison](../a8-qat-recovery/comparison.md) and
[14.43h snapshot](../nine-model-qat-overnight/current-eagle-a8-acceptance-20261006.md).
The two-hour arms each have five repetitions over 24 development prompts and
match their Q4 and target-only outputs on that panel. The later 14.43h comparison
matches Q4 but fails strict target-only parity on one prompt. That failure must
remain scoped to its experiment. Its training improved accepted drafts/round
4.32-fold versus initialization, while remaining far below Q4. The separate
1,000-step pilot reached only 0.127 accepted drafts/round against Q4's 1.306;
it has no full-request timing or matched zero-update training-effect control.

The untrained SM75 grid tested 12 depth/confidence policies. EAGLE W1A8 stayed
below Q4 across them. Independently selected development policies yielded
53.31 versus 92.06 request tok/s (57.9%); that is a policy-selection observation,
not a held-out optimum. Keep this hardware result separate from RTX 5080.

The DSpark rows are recomputed from the executed local
[closure](../../runs/5080-broader-training-20261031/SESSION-CLOSED.json): 120
requests, 14,290 output tokens per arm, five repetitions over 24 reused
prompts, historical D5/body7/full-Markov runtime and F16 target/KV. The closure
explicitly records no unseen release and throughput goal unmet. EAL's small
aggregate advantage over parent includes an unusually fast repetition and is
not a robust training-win claim. Earlier panels reported parent ratios around
94.5% and 90.6% of their own Q4 controls; 92.8% is the latest panel's fastest
observed aggregate, not a global best-ever bound.

## Other consequential work and what it establishes

| Work | Actual result | What remains missing |
| --- | --- | --- |
| EAGLE fusion calibration, discrete-sign fitting, full-body training infrastructure | CPU exports, arithmetic/gradient/resume checks and real native-trained checkpoints exist. Long calibrated EAGLE training raised acceptance over initialization. | Fusion reconstruction scores alone do not establish a serving gain; trained EAGLE still loses to its matched Q4 controls. |
| DSpark calibrated and scale-only 8h/16h lineages | Actual W1A8 exports and GPU/native evaluation exist. The local precision inventory records 16 binary projections; 8h/16h native prefix totals improve over initialization on audited panels. | No throughput success follows from those prefix gains or checkpoint labels. Training-exposure-unseen is not an untouched release set. |
| Later DSpark sign/scale/norm and acceptance-oriented training | Parent3504 and EAL2160 are actual 13-projection W1A8 exports, followed by additional resumed/fresh candidates and real serving measurements. | Native score gains do not reliably improve request speed. Fresh selected checkpoint remains slower than parent and Q4 on the latest panel. |
| First-error loss, lower sign LR, frozen-sign controls | Full-rate fits TRAIN better but damages some good prefixes; development prefix totals 115/180 versus parent119. Lower LR/frozen each reach122 but fail the prospectively fixed multi-group/domain gate. | No successful serving promotion; parent-margin retention is a proposed follow-up, not an executed improvement from that report. |
| Data-aware signs, CE/norm/KD/input-gain/rotation/base-restart and broader-answer probes | Preserved attempts include local fit gains without native gains; the sign-search summary reports more native-screen rounds, 193→199. | No justified full-request promotion or evidence that a capacity ceiling has been proved. Some older reports' linked external paths are not present in this checkout; scope is summary-level for those attempts. |
| Native packed W1A8 execution and runtime optimizations | Real integer execution, activation packing, shared packing/head pruning, algorithm/shape selection and actual GPU serving exist. This is more than simulated low-bit arithmetic. | Faster draft operations alone have not closed the acceptance/verification deficit. New runtime settings still require fresh correctness admission. |
| Cheaper dynamic vocabulary head | Implemented CPU prototype and cost/recall analysis; parked before native-engine integration. Much measured head work includes Markov computation the proposal retains. | No new native throughput result. Unchanged-trajectory estimates remain below Q4, and selected-row numerical equivalence is unproved. |
| DFlash W1A8 | Actual calibrated CPU exports and synthetic native graph checks, plus later preparation, exist. | No production-trained W1A8 GPU serving endpoint found. Released Q4/BF16 DFlash measurements are not W1A8 results. |
| Modern Qwen3.5 W1A8, either size/target format | None in the audited matrix. Modern ordinary EAGLE exports were repaired for partial RoPE and layer-tap indexing before comparison. | Target-matched W1A8 initialization/training/export/admission and serving measurements. Historical Qwen3 drafts cannot simply be relabeled Qwen3.5 drafts. |

Sources: [transfer audit](../local-resume-20261101/transfer-diagnosis.md),
[actual-history diagnosis](../local-resume-20261101/runtime-and-transfer.md#historical-f16-target-training-transfer),
[precision inventory](../../runs/training-transfer-diagnosis-20261027/precision-inventory.json),
[first-error/freeze review](../../runs/w1a8-algorithms-20261020/acceptance-loop2-frozen-result-astra11.md),
[dynamic-head report](../../runs/w1a8-campaign-20261023/dynamic_head01/report.md),
[fusion reports](../fusion-binary-real-a8-2026-10-03/README.md), and
[CPU block exports](../nine-model-qat-preparation/calibrated-native-cpu-artifacts.md).
The initial October 4 CPU-only status was superseded by actual later DSpark
training and serving. It remains appropriate for the earlier historical screen,
not a description of current DSpark coverage.

## Why training has not turned into a serving win

The strongest evidence points to lost continuation quality and extra target
verification. In the latest matched panel, fresh W1A8 needs 3,825 rounds versus
Q4's 3,350, a 14.2% increase. Synchronized diagnostic spans show the draft itself
can be cheaper per round; extra verifier work accounts for most of the measured
round-time gap. This is diagnostic wall attribution, not GPU kernel percentages.

The actual-history audit finds gains on difficult positions offset by losses on
previously successful full prefixes. On 549 comparable frontiers, fresh gains
67 accepted tokens but loses 92. An identical-input replay retains losses on
28 of 32 damaged states, supporting a learned model/data/objective tradeoff.
Residual replay mismatches remain, so state/numerical causes are not universally
excluded. Captured argmax-label checks pass their narrow audit; there is no
basis to call all historical teacher labels corrupted. Dense head costs also
remain, but head optimization alone does not explain away the observed losses.

The evidence-backed next direction is to preserve successful continuations while
improving difficult states, with actual serving-history coverage and acceptance/
depth-aware objectives. For the selected modern deployment goal, first establish
a target-matched W1A8 candidate; then test it against the pinned DFlash control.
Neither a new training run nor a port is started by this inventory request.

## Costs, verification and limits

This inventory performed zero training, zero GPU evaluation and zero new model
requests. It reads existing reports, compact receipts, pinned model precision
metadata and executed closure totals. [Evidence JSON](evidence.json) retains
historical ratios, scopes and source hashes. Two read-only workers independently
reviewed model coverage and historical measurements; root corrected stale
DSpark readiness claims and checked exact closure arithmetic.

Historical two-hour EAGLE budgets are 7,200 accounted seconds per arm; the later
calibrated snapshot reports 51,950.92 trainer seconds. DSpark 8h/16h checkpoint
labels, older preserved campaigns and later continuation/fresh clocks must not
be added into an invented total. The latest closure reports 8,516.06 training
seconds and separately 3,146.14 fresh seconds with unresolved inclusion. Capture,
export, compilation, native evaluation, search and preparation are separate.
A complete all-lineage training/preparation total is not established here.

The final modern matrix remains the only held-out four-target comparison in
this inventory. The historical panels are development evidence with their own
correctness limitations; independent broad answer quality is not established
for all W1A8 lineages. No positive W1A8 result, model migration or completed
nine-model campaign is inferred from these records.

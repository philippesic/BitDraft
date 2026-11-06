# Current project status

Current: the remaining-component campaign is active. Fifteen physical4096-update
windows are complete: H0/H4/H3/E4/tied HE4 and all six draft-cache pairs, plus row-scale/midpoint/gain/clip windows and exact
private-Q6 controls reusing H0. Norm storage is running, then all-scalar. All four
native recovery readouts are retained. Combinations, formal repeated comparisons,
final heldout and final charts remain unfinished; no throughput promotion.


## Active: remaining component precision and recovery

User now authorizes executing the remaining head/embedding/drafter-KV/scalar
plan with matched recovery, combinations and charts. See
[COMPONENT_CAMPAIGN.md](COMPONENT_CAMPAIGN.md) for scope, resource observations
and operational ownership. Head Q6_K/private Q6_K/private Q4_K is first;
qualified initial primary arms get cumulative0/256/1024/4096update readouts.
One delegated worker will verify and own the GPU lane. Earlier planning-only
statements are superseded; preserve the frozen optimized G4/D8 and references.

## Execution optimization complete; integrated locally

Next planned precision work: [REMAINING_PRECISION.md](REMAINING_PRECISION.md).
Start with shared Q6_K/private exact Q6_K/private Q4_K head isolation, then
private/tied embeddings, K/V cache ladders and scalar storage. Give damaged
variants matched recovery opportunities, including reserved slow starters and
explicit training eligibility. The original planning-only restriction is superseded by the active authorization above.

The 38-item execution inventory is covered and the retained runtime is integrated.
[Report](../experiments/execution-campaign-20261105/report.md) ·
[Item ledger](../experiments/execution-campaign-20261105/ledger.md) ·
[Campaign evidence](EXECUTION_CAMPAIGN.md).

Fixed learned-small2048 W1 GPU forward falls 1.428→1.183ms (17.2% reduction;
paired95% 15.5–18.8%). Q4 is1.178ms; W1/Q4 ratio1.004 [0.989,1.022], an unresolved
near-tie rather than a W1 latency win. Full draft host cost falls2.441→2.174ms;
Q4 is2.161ms. Actual request throughput rises2.51% with original/optimized W1
proposals, counts, required logits and cache state preserved. Q4 remains the
full-request throughput incumbent. No training or precision change.

Parent independently verified34publication hashes, recomputed rates/counts from
700responses/600measured rows, checked180warm GPU graph regions and30cache/logit
comparisons, checked proposal/fallback audits, and reproduced optimized source
hashes from the frozen-base patch. GPU idle/released; no pushes. The broader
research goal remains open. Original precision-chart artifacts stay immutable.

## Completed precision comparison

Next execution investigation is agreed in
[EXECUTION_OPTIMIZATION.md](EXECUTION_OPTIMIZATION.md): trace/replay qualification,
copy removal and known-bound specialization, three-position DP4A reuse versus
signed INT8 MMA, then one profile-selected fusion or unused-anchor-head test.
Astra xhigh and root agree; this planning update launches no GPU work. Q4 also
uses DP4A in the frozen dispatch, so instruction family alone is not the cause.

The modern Qwen3.5-4B Q4_K_M / RTX5080 campaign and one bounded clipping control
are complete: 258 measured panel/arm records, 67 formal confirmed rows, and
10,580 formal target-identical responses. Reports and artifacts:
[main report](../experiments/precision-campaign-20261104/report.md),
[matched clipping control](../experiments/precision-campaign-20261104/clip-control-report.md),
[execution checkpoint](PRECISION_CAMPAIGN.md). Combined charts are under
`runs/precision-campaign-20261104-01/charts-clip-control`; the original 240-point
`charts-initial` package retains every original hash.

DFlash Q4_K_M remains the throughput incumbent. Initial heldout: Q4 242.0 request
tok/s versus compact learned G4/D8 163.1, with 363.78 versus 174.94 MiB drafter
bytes. The later matched 1,024-update control on a separate fresh cohort gives
fixed clipping 167.51 versus learned 162.21 tok/s (paired ratio 1.0327,
95% 1.0180–1.0484); Q4 still reaches 245.61. DEV is tied within uncertainty.
No W1 deployment promotion, no inference-only or cross-cohort attribution.

Original vanilla 14,231 updates / 1,800.021702 seconds and all fixed references
remain unchanged. All paid recovery states and native exports are retained;
owned jobs/process groups are released and the GPU is idle. Storage/process
proof: `runs/precision-campaign-20261104-01/final-storage-process-inventory.json`.
Parent reviewed and integrated the campaign locally, verified both published
artifact packages, and independently passed 16 function fixtures plus 20 unittest
cases. Inline chart preview executes without console errors. No pushes; no new
training launched during review. The broader nine-model goal remains
active; no 9B/NVFP4 qualification or nine-model restart was performed. Historical
preparation-only/no-sweep limits below were superseded for this campaign by the
latest explicit authorization.

## Stage protocol and historical authorization

Agreed precision plan: [PRECISION_SEARCH.md](PRECISION_SEARCH.md). After the
foundation, explore gate/up A8/A4/A2/A1 families, down through A1, and bounded
component reductions. Retain a tradeoff frontier separately from the deployment
incumbent. Root accepted all Astra xhigh review revisions; no measured winner,
new training allocation or GPU ownership change follows from this plan.

User correction: the common foundation includes signs, row scales, midpoints,
input-channel gains and existing norm gains BEFORE comparing precision setups.
Then add progressively experimental features. See the agreed
[exact stage/feature scope](CANDIDATE_STAGES.md). The existing delegated vanilla
run is a minimal control, not the expanded foundation. Preserve its run/budget;
this scope revision launches no new sweep and chooses no precision winner.

Root has launched the1800-second initial run after actual sign/scale updates,
exact resume and native zero/updated correctness qualification. Ownership and
stop instructions: runs/dflash-modern-train-20261104-14/ownership.json.
Use the baseline registry for actual checkpoints; launch is not completion.

November4 user decision: use a frozen zero-update W1A8 reference, a frozen short
checkpoint, matched no-feature continuations and a separately promoted incumbent.
[Candidate protocol](CANDIDATE_BENCHMARKING.md) is the entry point;
[baseline registry](../experiments/dflash-w1a8-baseline-20261104/manifest.json)
will contain the actual benchmark/export/checkpoint pins. Preparation and initial
training are authorized for one delegated worker. Coordinator proposes30minutes
of measured update time, checkpoints at0/5/15/30minutes; consult the registry
and run evidence for produced checkpoints.
Initial lane: modern4B Q4_K_M target on5080. Keep DFlash Q4_K_M external baseline
frozen. No feature sweep or long-QAT allocation. This continues the existing goal.

The dated records below preserve historical evidence; earlier statements that
architecture is undecided do not override this current DFlash/W1A8 preparation.

## Historical design leads

Near-term design lead: revisit W1A1 parallel proposals, retries and target-verified
trees. User recalls old untrained EAGLE W1A1 at roughly half W1A8 acceptance and
draft latency; exact historical comparison remains to be reverified. Recorded in
`docs/DECISIONS.md`; no architecture selection; the bounded execution screen below is now complete.

## Binary execution comparison — completed November 4

Nine native paths plus published FP16 LUT-GEMM screened on RTX5080. Historical
Qwen3-4B F16 + retained DSpark, two repetitions × three reused DEV prompts; no
training. A8 Tensor Core default 231.70 decode tok/s; DP4A 230.93, bit-serial 223.96,
global LUT4 223.44, exact shared LUT 227.68. Binary pipeline 132/217/708/731/430µs
per round respectively. A1 override on the same A8-trained weights has 28.6%
acceptance versus 52.4%; its Tensor adapter 156.63 tok/s and 173µs/round beats tested
A1 XOR/BMMA/shared-LUT binary pipelines, but this is not trained A1 quality.
SM120/SM120a binary-MMA PTX lowers to INT8 IMMA in the tested code, not native BMMA.
All 90 final native responses match target-only tokens; 54 kernel checks pass.
No modern DFlash baseline evaluation or architecture selection. Detailed charts:
`experiments/binary-methods-20261104/README.md`. Raw/frozen evidence and cleanup:
`runs/binary-methods-20261104-01/`; 48 owned server identities checked absent, GPU
empty. Retained ~238MB of evidence/runtime; 191,508,480 more WSL free bytes observed
after removing preliminary build/trace scratch. No Windows-space claim. Next:
discuss architecture/activation precision with Tensor Core expansion as default
and W1A1 XOR/popcount retained as a shape-specific option.

## Quick binary pipeline profile — completed November 4

RTX5080, historical DSpark parent3504/Qwen3-4B F16. Three-repetition isolated
actual-FFN-shape stage timings plus one warmed native DEV request per arithmetic
path. Native imma_mix uses integer Tensor Cores; existing planes7 bit-serial dots
cost6.53× more, with identical128output tokens and33round/164proposal/94accept
counters. Retained binary pipeline sums4.345ms versus23.113ms; only0.70% of all
kernel-duration sums in the retained historical request. Not draft-only share,
removable wall time, modern-target result or throughput claim. Activation prep
can dominate single-position standalone projections.20kernel correctness cases
pass. Nsight Compute counters denied; occupancy/bandwidth saturation unmeasured.
No BMMA timing, training or architecture selection. All jobs finished, GPU empty.
Report: experiments/binary-pipeline-profile-20261104/README.md.
Raw: runs/binary-pipeline-profile-20261104-01. Next: discuss binary execution
mechanism using retained integer Tensor Core execution as engineering comparator.

## Original DSpark W1A8 versus DFlash — completed November4

The user-requested retained DSpark W1A8 parent3504 comparison is complete.
Original Qwen3-4B F16 target, same runtime, D5, six consumed DEV prompts×five
plain/traced repetitions. W1A8 207.42tok/s versus DFlash Q4_K_M178.49 (+16.21%,
paired interval+9.87–+24.20%); same-family DSpark Q4_K_M223.45 (W1A8−7.17%).
State update0.358/0.346/0.354ms per round respectively; differences unresolved.
All320responses exact, trace counters unchanged. No EAGLE, Qwen3.5 substitution,
training or policy search. 338.05s supervised wall; all43identities gone,
GPU empty. This small reused-prompt result does not establish a modern baseline
win or optimal DFlash setting. Report: experiments/dspark-w1a8-dflash-20261104/README.md.
Raw run: runs/dspark-w1a8-dflash-20261104-01. Candidate remains DSpark parent3504;
no automatic training continuation or architecture selection.


## DSpark versus DFlash Q4 catch-up diagnostic — completed November4

Matched modern Qwen3.5 4B Q4_K_M, both draftsQ4_K_M/depth2, same runtime,
five alternating plain/traced repetitions. DSpark state update0.268ms/round
versus DFlash0.355ms:24.58% cheaper, paired95%interval22.36–26.90% cheaper.
No DSpark catch-up penalty appeared. DSpark emits fewer tokens/round2.064 vs
2.466 and remains slower end-to-end. All420responses exact; instrumentation
preserves draft/round counts. DSpark tracing adds1.73%request wall, so component
spans remain diagnostics. No training.257.28ssupervised job wall; all processes
stopped, GPU empty. Historical DSpark W1A8 and modern EAGLE W1A8 are distinct.
Report: experiments/dspark-dflash-catchup-20261104/README.md. Raw run:
runs/dspark-dflash-catchup-20261104-01. Pinned source worktree retained.
Next: joint architecture/precision discussion; no ongoing GPU jobs.


## Completed: preliminary binary candidate screen — November 4, 2026

The queued message authorized light architecture/precision exploration. The
zero-update screen is complete: [report](../experiments/binary-screen-20261104/README.md).
No final design, permanent rejection, fresh generalization or serving win claimed.

Actual RTX5080 DP4A/A1/A4/A8 standalone kernels pass200 final code/integer/scale
cases across current and released DFlash projection shapes. Packing-inclusive
warm-device timing favors bit-serial A8 on the head but not every small shape;
A4/A1 offer larger potential reductions. Current deployed W1A8 is DP4A, whereas
these new prototypes exercise real POPC instructions. They are not native ports.

Nine unadapted activation policies use12already-TRAIN groups/24teacher-forced
positions. A8 matches12target choices; all-A4/all-A1 match8/4; MLP-only A4/A1
match10/9. The baseline favors A8; no equal-QAT ranking follows. Preserving fusion
precision did not rescue full-model A4. Group128/two-plane local reconstruction
and storage/capacity tradeoffs are recorded; no runtime or whole-model promotion.

Parallel binary DFlash remains a structural option requiring8tap captures,
new objective/cache/export contracts and actual resource/quality tests. The
ordinary released DFlash scores must not be inherited by a binary conversion.

Training updates0;126.66s supervised GPU-job wall including startup and data/model
loading. Independent CPU audit reproduces all token/CE summaries and policy masks;
A8 repeats exactly and parameters stay unchanged. All18owned identities released,
no GPU compute app remains. Raw logits/activations, kernels and audits retained;
duplicate A8 artifacts hardlinked and secondary CUDA cache consolidated. Source
worktree `/home/philip/.codex/worktrees/binary-screen-20261104` is pinned and retained.
Raw evidence: `runs/binary-screen-20261104-01/`. No push. Next: joint design choice.

## Completed: data breadth and decoding costs — November 4, 2026

The authorized follow-up is complete, directly in this thread. No architecture,
precision or long-QAT recipe was selected. [Report](../experiments/data-cost-transfer-20261104/README.md)
and [design handoff](../experiments/data-cost-transfer-20261104/design-use.md).

192 additional TRAIN groups (64/domain),11,391 windows and exact native embeddings
are admitted with bounded metadata/array streaming. Whole prompts are short;
features remain tied to the current target/taps/history contract. Apply the new
[TRAIN exposure index](../experiments/data-cost-transfer-20261104/train-exposure.json)
before future held-out selection. Untouched final inputs remain unopened.

Two fixed-candidate arms completed1,920 extra updates each. Old-reference and
broader continuous/split-resume states/exports match exactly. Broader data has no
established speed gain: consumed natural24 −0.32%,95% interval[-1.61%,+0.80%];
all three short-output DEV checkpoints also unresolved. No pruning rule follows.

Five paired native profile repetitions preserve outputs and draft/round counters.
Current narrow draft generation is1.534ms/round versus DFlash1.766, but catch-up
reverses the advantage: complete draft work2.340 versus2.120ms/round,10.36% more
(95% interval7.86–12.61%). Current needs2,785 rounds versus1,900 for4,685 tokens.
Bounded Nsight traces identify the head at31.7% of draft GPU kernel duration but
2.51% of all kernel duration; these sums are not removable request time.

All2,498 qualification/serving/diagnostic responses match target tokens/text/stop;
five CPU tests pass.192 TRAIN captures are counted separately.3,888 physical
updates used639.33s update wall within7,334.20s supervised job wall, including
failed attempts and preparation/evaluation. Not pure GPU time or additive clocks.
451 recorded owned process identities are gone; no GPU compute process remains.
Paid bank/checkpoints and useful traces retained (~5.30GB WSL,0.73GB C:). Pinned
source worktree `/home/philip/.codex/worktrees/data-cost-transfer-20261104` retained.
Raw completion/cost/release/inventory: `runs/data-cost-transfer-20261104-01/`.
No push. Next: joint architecture/precision design, then selected-design adapter
qualification and bounded native comparisons before any major QAT allocation.

## Active goal: contender workflow qualified — November 4, 2026

The user-authorized infrastructure implementation and extensive tests are complete
in the current thread. [Qualification report](../experiments/contender-infra-20261104/README.md)
and [workflow guide](../scripts/contender_pipeline/README.md) are the entry points.

119CPU tests pass. Real GPU stop/resume and optimized data/audit paths produce
exact parameters/Adam/RNG/exposure and byte-identical exports. All476native
responses match target tokens/text/stopping, including360measured requests.
The final copied deployment bundle also runs natively. All34native groups and
supervised jobs are released; no GPU compute process remains.

Common plans support staged execution, cumulative budgets and explicit recovery;
comparisons reject incomplete/mismatched evidence; array/metadata streaming and
protected confirmation/release checks are tested. The snapshot audit rebuilt the
same manifest/check totals in39.97s versus103.77s in one CPU comparison.

No architecture/precision/feature recipe or major QAT allocation was selected.
Unknown model/export/kernel adapters, larger selected TRAIN recipe/horizon and
broader deployment scope remain design-specific qualification. Final evaluation
inputs remain untouched; no general early predictor or permanent pruning rule.

Raw state: `runs/contender-infra-20261104-01/`.144physical reference updates;
105.05s update wall inside1113.09s supervised command wall, with separate
CPU preparation/audit work. Checkpoints, shards and evidence retained. No push.
Next step is the joint architecture/precision discussion and bounded candidate
plans using this workflow.

## Prior checkpoint: design discussion ready — November 4, 2026

The authorized backlog/design-readiness work is complete directly in this thread.
No architecture/precision or new feature recipe was selected, no major long-QAT
allocation started, and no push occurred. No owned job or GPU compute app remains.

[Start with the readiness brief](../experiments/design-readiness-20261104/readiness.md).
All98 canonical segments/34families and32feature IDs reconcile to original
records;51 retained source receipts rehash. Detailed evidence/dependencies/costs
and scoped missing raw assets are linked there. The current working reference
trains76,880 row scales in9tensors, with fixed signs. Separate selected-O sign
qualification exists; all-sign/grouped/new-architecture work remains explicit.

Four compatible objective arms and three extensions, two8-arm×5repeat native
panels, exact48-update equivalence, exact480-update split-resume, native recapture/
materialization/export/admission all pass. All1,246 panel/qualification responses
match target tokens/text/stopping; no new feature-vs-control throughput interval
excludes no change. No universal early predictor or permanent rejection inferred.
Paid replay5760 remains the recorded candidate; priorfresh24 transfer is unchanged.

Current paid candidate uses2785 counted speculative rounds vsDFlash1900 for
identical4685tokens on consumedDEV. Narrow drafting spans are cheaper but omit
catch-up; aggregate generation cost/round gives no corresponding advantage.
Current head/teacher probability coverage and all25 opaque partition indices
are audited; no new TRAIN/evaluation prompts selected or sealed payload opened.

Bounded follow-ups finish: gradient observation leaves exact control state;
synthetic2.8GB full-shaped CPUstate saves/restores in11.12/9.75s (not GPU/QAT
qualification); process reuse preserves state/export but gives no material
startup gain. Profiling identifies one-time data authentication as the main
small-trial preparation cost. Current1GiB cache/audit/depth limits and larger
training-data requirements are explicit rather than claimed already scalable.

Direct execution uses7,824 physical updates,6,720 unique treatment-trajectory
updates;17.57min measured update wall is contained in82.15min nonoverlapping
supervised wall, not additive/pureGPU accounting. All91 native groups released.
Paid checkpoints,~394MB unique deployed/checkpoint payload and~68MB raw evidence
remain. Five duplicate exports are losslessly hardlinked, preserving every path/
byte; synthetic scratch/task-local bytecode removed. No VHDX compaction.
[Machine readiness/costs](../experiments/design-readiness-20261104/readiness.json)
and raw `runs/design-readiness-20261104-01/cleanup-final.json` retain details.
Next is the joint architecture/precision/trainable-scope/head/data design choice;
known selected-design qualification/capture/sharding work is in the brief.

## Prior checkpoint: ordered W1A8 generalization work — November 3–4, 2026

The two user-authorized bounded diagnostics are complete and raw-audited. On
12 reused DEV prompts with five repetitions, old-only CE5760 reaches130.512
request tokens/s and expanded CE5760 reaches132.290: expanded+1.36%, paired
prompt/repetition95% interval−0.49–+3.26%. No serving advantage from the expanded
recipe is established at equal5760 updates. At3840, old129.461 versus expanded
130.216 gives+0.58% (−0.84–+2.06%). Both rankings are unresolved; this two-recipe
retrospective check is inconclusive for early native-serving prediction, with no
validated general pruning rule. These are consumed DEV data, separate from the
fresh24 transfer below. Equal updates are not equal measured GPU seconds.

The old2400→5760 continuation reproduces the original3840 scales/Adam/cursor/
RNG/exposure exactly. All490 panel responses and14 new-endpoint qualification
responses match target tokens/text/stopping. All36 native groups and owned
supervisors are released. The work finished experiments/audit in17.6minutes,
inside the30-minute total cap; no extra training arms, new held-out campaign,
source edits or push. Recommend a bounded broader training allocation with
predeclared native checkpoint checks, supported by priorfresh24 transfer; this
small rank check cannot justify pruning longer runs.
[Compact result and full raw evidence](../experiments/w1a8-modern-native-20261103/duration-predictor.json).

User-directed storage cleanup is complete. Available space increased by171.48GB
in WSL and40.33GB on C:, leaving approximately173.66GB/42.64GB free respectively.
7,450 obsolete payloads and1,615 aliases removed;440 retained files verified and
current initializer/candidate/checkpoint hashes unchanged. Active modern assets,
selected historical endpoints, source and compact evidence remain. Current/legacy
dependencies discovered during review were excluded. Historical raw replay data
removed by this authorized purge requires regeneration. No VHDX compaction or
experiment restart during cleanup; the new bounded diagnostic authorization is above.
[Cleanup record](../experiments/storage-purge-20261104.json).

Phase1 now has confirmed throughput transfer to24 unused corpus prompts, beyond
its earlier synthetic-family confirmation. Candidate/runtime froze before input
selection. On Qwen3.5 4B Q4_K_M/RTX5080 correctness mode, selected replay5760
reaches165.886 tokens/s versus initializer134.060: +23.74% (paired stratified
prompt/repetition95% interval+20.85–26.89%). It exceeds the earlier conservative
candidate140.041 by18.46% (15.77–21.36%). Gains appear in each domain and all five
repetitions. This is unused against recorded project exposure, not proven
pretraining novelty, semantic independence, or transfer across all four targets.

The matched unchanged5760control reaches165.340; replay's extra0.33% has an
interval spanning−0.63–1.27%, so no incremental replay speed win is established.
Replay was selected before inputs for similar DEV speed with better preservation
on a fixed native diagnostic (9/113 losses versus17/113). Frozen DFlashQ4_K_M
reaches252.685 on the same fresh panel; selected W1A8 is65.65% of DFlash and also
below target-only176.550. Historical Qwen3 F16 stays separate.

The complete panel has840 performance requests,168 quality requests,84 warmups,
42 startups and936 speculative responses exactly matching target tokens/text/
stopping. Root independently audits all raw responses, model/runtime/image pins,
source snapshots, native context admission and released groups. All48 tasks fit
4096context with256output reserve; no replacement/truncation. Independent source
review passes684 checks and reconstructs all24 selections against512historical
TRAIN/current60/consumed48/priorDEV/pilot/matrix and25 opaque partition indexes.
No dedicated DEV/SEALED prompt shards opened. The new48 cases are consumed and
must be excluded from future untouched confirmations.

All seven arms score12/24 on the frozen synthetic quality slice (code5/8,
reasoning0/8, prose7/8); outputs are identical across arms. This shows no added
quality loss, while exposing real target-model weakness on these reasoning
cases. Independent judge/oracle review passes168rejudgments,24positive/50negative
controls and8independently recomputed reasoning oracles; failures are wrong
answers, not formatting or budget exhaustion. No broad prose-quality or
absolute quality-floor claim. Prior narrow synthetic confirmation remains+5.30%
with exact outputs; its earlier cohort is separately consumed.

Full native preservation is complete and root-audited on all883fixed histories/
956prior successes: unchanged5760 loses114 (oldFIT2/DEV41/addedFIT71); replay
loses72 (2/42/28). Replay recovers45 positions and introduces3 losses, so its
benefit is concentrated in addedFIT. All40native processes and owned groups are
released;27loaded images rehashed. This does not establish zero-loss preservation.

No GPU experiment is currently running. The original old-data continuation
stopped before updates at its storage guard; after the authorized cleanup, the
bounded retry above completed with exact3840 reproduction and native serving.
The original failed preparation and all paid modern checkpoints remain preserved.

The user explicitly confirms that perfect draft preservation and DFlash
competitiveness are not gates. No zero-loss or absolute quality-floor exit
requirement is inferred. This small native predictor check is complete, while
general predictor validation, backlog/combinations and final four-target
confirmation remain pending. Paid continuation checkpoints and horizon19200
remain preserved. Historical raw data removed by the authorized purge requires
regeneration; useful final/best endpoints and compact evidence remain. No push.

[Fresh natural confirmation](../experiments/w1a8-modern-native-20261103/natural-confirmation.json),
[fresh natural raw audit](../runs/w1a8-modern-protected-20261103-01/natural-replay-confirmation01/root-analysis.json),
[progress](../experiments/w1a8-modern-native-20261103/progress.md), and
[active goal](goals/nine-model-qat-overnight.md) retain evidence and costs.

## Selected baseline and W1A8 standing — November 2, 2026

DFlash Q4_K_M is now the user-selected primary baseline for all four modern
Qwen3.5 correctness-mode targets. Frozen depths/rates: 4B Q4_K_M d2 235.9;
4B NVFP4 d3 258.3; 9B Q4_K_M d3 200.8; 9B NVFP4 d3 209.2 tok/s. Target-only
remains the correctness reference; Q4_0 EAGLE is a secondary diagnostic for this
track. This supersedes the primary-comparator language in the dated matrix
checkpoint below, without changing its measurements or historical controls.

There is no modern-target W1A8 checkpoint/measurement in this matrix. Historical
EAGLE W1A8 reached 51.7–70.5% of paired Q4_0 EAGLE request throughput in the
reported trained comparisons. Actual trained/native-served historical DSpark
W1A8 exists; the latest panel reaches 87.3–92.8% of same-family Q4_K_M, with
no untouched release or proven win. Earlier panel ratios differ. No production
trained DFlash W1A8 serving endpoint was found. Historical targets are Qwen3 F16,
not the four modern Qwen3.5 targets; do not compare their raw tok/s directly.

[Baseline pins](../experiments/w1a8-standing-20261102/baseline.json) and
[full W1A8 inventory](../experiments/w1a8-standing-20261102/report.md) retain
model coverage, measured results and limitations. This update used CPU evidence
inspection only: no new training, GPU evaluation or runtime change.

## Completed: Qwen3.5 correctness-mode baseline matrix — November 1, 2026

The requested 4B/9B × Q4_K_M/NVFP4 remeasurement is complete on the local RTX
5080, using the existing immutable quant4group correctness runtime (mode32).
All 58 selected configurations completed five held-out repetitions: 290 cells,
8,352 measured HTTP requests. Root audited all raw responses, all 48 frozen
source/model pins (including full hashes of 25 GGUFs), and every cell's CUDA
library map. All 7,776 speculative requests exactly match their target-only
control's tokens, text and stopping state; repeated outputs also match.

Q4_0 EAGLE remains the primary research comparator. Its held-out rates are
187.3/180.8/173.1/184.6 tok/s for 4B Q4_K_M, 4B NVFP4, 9B Q4_K_M and 9B NVFP4.
The fastest observed rates are 237.8/258.3/200.8/209.2 tok/s, from DFlash Q8_0
for the first target and DFlash Q4_K_M for the other three. On 4B Q4_K_M,
DFlash Q8_0, DFlash Q4_K_M (235.9) and embedded MTP (235.0) are statistically
unresolved on this panel. A deployment win must also beat these alternatives
and target-only; this is not a W1A8 result. See the
[full 58-row matrix](../experiments/qwen35-correctness-matrix-20261101/matrix.csv)
and [report](../experiments/local-resume-20261101/runtime-and-transfer.md#qwen-correctness-mode-baseline-matrix--november-1-follow-up).

Independent quality scores are 12/24, 9/24, 12/24 and 13/24, identical across
speculative arms and controls. An independent oracle audit confirms genuine
answer errors. A post-confirmation default-runtime replay scores 11/24, 9/24,
12/24 and 13/24, with no case-score regressions from correctness mode on this
panel. Exact parity does not establish strong answer quality. Original and
correctness-mode outputs differ, so this is not stock-output equivalence.

Modern EAGLE was measured using separately corrected exports: partial RoPE,
Q/K permutation restricted to the rotated slice, and native input taps
[2,16,29]. Source/real-weight CPU checks pass; complete native-versus-training
forward-graph fidelity remains unproven. Original exports and trained checkpoints
are preserved. Historical Qwen3 F16 training is unchanged.

Training cost is zero. Nonoverlapping supervised experiment intervals total
6.38 hours, including 4.59 hours for held-out confirmation. Preparation and CPU
audits are separate; the failed preparation's aggregate wall was not recorded.
The [cost ledger](../experiments/qwen35-correctness-matrix-20261101/cost-ledger.json)
records components and overlap. All four supervisors exited zero; root verified
same-boot process-group absence and no CUDA compute applications. Release and
bounded cleanup evidence: `runs/qwen35-correctness-matrix-20261101-01/release-and-storage.json`.
No further job is running, no new training or migration was selected, and no
push occurred. The broader nine-model history remains open in its existing goal.

## Follow-up: upstream foundation qualified — November 1, 2026

The user requests a clean solution without a custom numerical workaround.
The old native F16 alternative FAILED its fresh test (two new prompts differ in
all five repetitions), despite passing the small development screen. No training
or baseline migration was started; evidence is preserved in
`experiments/clean-baseline-20261101/f16-rejection.json`.

Unmodified upstream vLLM0.32.0, retained Qwen3-4B BF16 and ordinary AngelSlim
EAGLE3 passed a 24-prompt development gate with positive draft acceptance and
exact token/text/stop equality under its supported batch-invariant mode.
Separate environment `runs/clean-engine-qualification-20261101/venv`; original
training environment untouched. No engine source patch or custom binary port.
Two failed preparation attempts (client API return type, long IPC path) are
preserved; fixed using normal API and documented RPC path setting.

Final confirmation completed all 20 cells, 480 performance and 96 quality
requests: five alternating repetitions with source/settings frozen before fresh
inputs. Invariant EAGLE3 equals target-only token/text/stop on 120/120 paired
performance requests and 24/24 quality cases; every arm repeats identically.
Both invariant arms score 14/24 on the frozen independent judge, with the same
ten failures. All 24 correct and 24 wrong controls behave correctly. EAGLE
acceptance is real: 7,965/22,155 draft tokens. Default mode fails parity on 14/24
prompts in every repetition.

Invariant EAGLE averages 19.951s versus 32.092s target-only per 24-request panel:
1.609× observed full LLM.generate speed, 37.8% lower time. Mode cost versus
default is 4.01% target-only and 11.78% EAGLE; cross-mode outputs differ.
Repetition CVs are 0.25–0.38%. Inference JIT warnings are not fully phase-localized:
these are observed full API wall times, not pure-GPU or fully warmed steady-state
costs, HTTP capacity, concurrency, a binary result or a Q4_0 win.

Recommend this pinned ordinary upstream foundation if dense Qwen3-4B BF16 and
an engine change are acceptable. No migration, new teacher, training or binary
port is adopted. Same-engine primary Q4_0 and custom binary execution remain
required before a project-level win. The user owns that major research choice;
see the pending decision in DECISIONS and
[full evidence](../experiments/clean-baseline-20261101/report.md).

Final phase wall 1117.33s: 503.74 performance, 53.24 quality, 368.02 initialization,
44.71 excluded warmups; remaining time is integrity/subprocess/harness work.
Development gate 177.30s; aggregate preparation clock unavailable; training zero.
Supervisor `runs/local-vllm-final-20261101-01/state.json` finished exit 0 at 20:37UTC.
Root rechecked same boot, absent process groups and no GPU compute apps. Removed
21 owned RPC scratch directories; reusable environment/evidence retained. Storage
and release receipt: `runs/clean-engine-qualification-20261101/release-and-storage.json`.

## Local RTX 5080 investigation completed — November 1, 2026

The requested runtime and training-transfer investigation reached its bounded
end goal. The isolated CUDA candidate improves Qwen3.5 MTP full-request
throughput by 32.0–51.0% over the conservative workaround across 4B/9B Q4_K_M
and NVFP4. Five alternating repetitions of 24 fresh prompts per cell produced
1,920 exact token/text/stop matches, with unchanged acceptance counters. All
candidate quality outputs match the conservative control (47/96 passing cases);
absolute quality remains modest. This is opt-in runtime evidence, not a
W1A1/Q4_0 EAGLE win or output equivalence to the unmodified default backend.

The upstream quantized-KV fix does not replace our F16-KV path, and the GB10
warp override does not act on RTX 5080. The validated candidate instead combines
restored two-column vector attention with grouped quantized projections while
retaining the conservative floating paths, fusion disable and one-partition
attention. Changes remain isolated and reviewable; parent gitlink unchanged.

Historical F16-target work stays separate. Improved difficult-state scores hide
losses on previously successful continuations and extra verifier rounds. A
96-state identical-input native replay retains most of those losses, supporting
a model tradeoff rather than history differences alone. Replay fidelity is
bounded. The justified next training experiment balances actual serving-history
coverage, successful-prefix retention and acceptance/depth weighting at matched
exposure, with original-group splits and fresh full-request confirmation.
No new training, restart or shortened historical allocation occurred.

[Full findings and patch](../experiments/local-resume-20261101/runtime-and-transfer.md),
[recomputable confirmation](../experiments/local-resume-20261101/confirmation-summary.json),
[cost ledger](../experiments/local-resume-20261101/cost-summary.json), and the
[latest goal checkpoint](goals/nine-model-qat-overnight.md#final-confirmation-and-release--november-1-2026)
retain details. The broader nine-model history is preserved; this checkpoint
does not declare every historical campaign complete.

Final confirmation ended at 10:53 UTC with exit zero. Root rechecked the
recorded boot/birth identities and absent process groups; all owned supervisors
are terminal and no CUDA compute application is listed. The RTX 5080 is idle
apart from Windows graphics allocations. No worker owns a GPU job. Current
release and retention evidence are ignored files under
`runs/upstream-consistency-20261101/{release-proof,retained-inventory}.json`.
About 12.1 GB of useful captures, arrays, source and build cache is retained;
paid checkpoints and pre-existing dirty/shared files are protected. No push.

This project runs directly in WSL on the RTX 5080 machine from
`/home/philip/binary-eagle-decoding`. Use the local shell and GPU; no SSH route,
host registry, machine link or device checkout is required. Follow
[local execution](AGENT_OPERATIONS.md#local-gpu-execution). The earlier access
blocker was incorrect; declining the unnecessary link card did not pause work.
[Archived status](STATUS_HISTORY.md) preserves earlier evidence as history,
not current jobs or launch instructions. Q4_0 EAGLE remains the primary EAGLE
comparison baseline. New runtime changes require revalidation.

## DFlash baseline implementation inspection — November4

Source inspection completed; requested training remains unstarted. Modern source
has6layers/8taps/18FFN and16slots with sliding-window attention; historical
training/export/native guards cannot admit it. See
[implementation checkpoint](../experiments/dflash-w1a8-baseline-20261104/README.md). No owned job is live;
GPU released. Full1800second update allocation remains unspent.

## Packed zero-update serialization produced

Implementation now includes `scripts/dflash_baseline/export_zero.py`. Actual
462MiB artifact and audit: `/home/philip/binary-eagle-decoding/runs/dflash-modern-20261104-02/{zero.gguf,zero-audit.json}`.
All18 FFN matrices use nonnegative-is-positive packed signs and F32 row mean
absolute scales; original BF16 attention/fusion and F32 norms remain byte-identical.
Packed signs/scales pass byte-hash roundtrip. Shared head/embedding remain external.
Native admission is **not qualified**; no throughput or correctness claim.
Author graph downloaded from https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py
into the run source directory; this moving-source download is not yet a qualified
training-source pin. No training/capture/evaluation performed;1800seconds unspent.
No GPU process launched. Training/resume, modern capture, native adapters and
matched evaluation still require implementation. This is partial work, not completion.

## Modern native tensor admission checkpoint

Separate ffn18/A8 loader guards and shared-head admission implemented as a
reproducible runtime patch under experiments/dflash-w1a8-baseline-20261104/runtime/.
CPU library builds and frozen zero.gguf loads successfully. Native CUDA execution
and correctness are not qualified. No captures or training updates performed;
all1800 update seconds remain. No GPU jobs launched. Next: qualify shared-target
CUDA decoder and semantic rejection cases, then implement captures/QAT/resume.

November5 execution campaign: implementation active in the isolated
`t3/dflash-execution-optimization-campaign` worktree. Exact compact learned-small
2048 export remains primary; original W1/Q4 trace completed, execution candidates
under qualification. See `docs/EXECUTION_CAMPAIGN.md` and canonical
`runs/execution-campaign-20261105-01/live-ownership.json` for current owner/jobs.
This does not reopen training or complete the broader goal.

November5 execution optimization campaign completed. See
[final report](../experiments/execution-campaign-20261105/report.md) and
[38-item ledger](../experiments/execution-campaign-20261105/ledger.md).
Frozen learned-small2048 W1 GPU forward improves17.2% [15.5–18.8%]; request
throughput improves2.51% [1.37–3.50%]. W1/Q4 draft-latency difference is unresolved;
Q4 remains request-throughput incumbent244.39vs176.21tok/s. Exact native
required-output/cache/proposal/counter gates pass. No training/precision/target
change. Runtime, standalone reconstruction patch, five PNG/SVG pairs and
interactive HTML are published; all owned GPU groups released, safe cache
cleanup measured separately for WSL/Windows. Parent owns review/integration;
no push. The existing nine-model research goal remains open.

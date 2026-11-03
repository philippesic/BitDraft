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


## Complete check-in after longer training and preservation tests — November 3

This checkpoint supersedes earlier live-job/next-action claims above. User asks
for a wait and the complete account since authorizing the ordered continuation;
no new experiment is launched during this check-in. GPU compute query is empty.

The eight-arm quiet-logging comparison completes 480 measured requests plus 80
warmups in 449.10 seconds whole-job wall. All exact token/text/stop/repeat checks
pass and all 40 server groups are released. Balanced protected replay reaches
125.452 tokens/s against unchanged quiet initializer 110.587: +13.44%, paired
whole-prompt 95% interval +10.01–17.80%. DFlash is 196.359, so W1A8 reaches only
63.89% of it. Balanced versus ordinary CE is +0.90%, interval spanning no effect;
its additional preservation benefit is clearer than any extra serving speed.
Head-only training is approximately unchanged from initialization. Trace versus
quiet logging and the separately tested zero-output head skip show no reliable
speed advantage. The bridge-ownership fix remains isolated and unadopted.
Evidence: protected run `requests-balanced-quiet01/`,
`native-analysis-balanced-quiet.json`, `quiet-protocol-release.json`.

Two matched continuations reach 3,840 updates: the prior balanced replay control
and a separately source-pinned migration protecting all 356 native FIT successes,
including five excluded from the old actor/native intersection. Model-side
results improve but old development preservation does not: both lose 42 native
successes in the actor readout. No development labels enter the optimizer.
A native replay helper reconstructs actual frozen-history inputs using original
packed libraries, native target embeddings and F16 KV. On 96 diagnostic windows,
the unchanged parent preserves all 53 successes while the full candidate loses
11; this confirms model damage beyond a Torch-only numeric artifact.
Evidence: `stage2-plan.json`, `stage2-cpu-admission.json`,
`actual-native-{parent,candidate}01/`, matched continuation outputs on C:.

Controlled export choices restore the original output head, then reduce the
trained body-scale change to one half and one quarter. On the 53-success panel,
losses fall from 11 to 4 to 1 to 0. The quarter candidate is then tested across
ALL 549 original native successes, without an outcome-selected subset: 548 are
preserved, including all 356 FIT and 192/193 development successes. One reasoning
proposal regresses; this is not an observed final-answer error because the target
verifier corrects drafts. Selection uses reused development and cannot establish
untouched generalization. Evidence: `body-quarter-plan.json`,
`body-quarter-export.json`, `quarter-all-native01/result.json`.

The conservative quarter candidate completes all five repetitions of target-only,
DFlash, initializer and candidate: 240 measured primary requests. Rates are
154.748, 196.722, 111.039 and 116.606 tokens/s respectively. Candidate/init is
+5.01%, paired whole-prompt 95% interval +3.43–7.22%; candidate/DFlash is 59.27%.
All primary responses match tokens/text/stopping and repeat exactly. Candidate
accepts 251 drafts per repetition versus initializer 193. This remains a reused
12-prompt development panel. The wrapper stops on its storage floor after an
additional 48/60 Q4 diagnostic requests; that diagnostic is incomplete. Root
preserves the original failure, recovers the complete primary comparisons, audits
all 288 measured raw responses, rehashes 40 images and verifies all 25 server
releases. No inference failure is disguised as a completed five-arm panel.
Evidence: `quarter-requests01/{failure,recovered-primary}.json`,
`native-analysis-quarter.json`.

The quarter model and full runtime are frozen before fresh-input generation in
`fresh-quarter-confirmation01/{candidate-selection,models}.json`. No real fresh
cases have been generated. Generator and runner tests cover selection binding,
prior-input exclusions, independently judged JSON/Python/SQL answers, oracle
positive/negative controls, native context limits, exact HTTP accounting and
interruptible owned-server cleanup. These are implementation checks, not a
validated short-to-long screen or actual fresh answer-quality evidence.

A separate source audit identifies a next objective hypothesis: second-step CE
still uses the OLD parent's reachability mask, withholding labels even when the
current actor correctly reaches that step. At the 3,840-update FIT snapshot, a
detached union of old reachability and the current actor's correct prefix would
add 573 supported second-step labels, including 270 currently wrong predictions.
This was audited only; no implementation or fresh-result-driven retuning occurred.
A future isolated comparison must account for changed effective loss mass and
keep development labels out of gradients.

Operations: the restarted historical control ultimately publishes verified full
checkpoints, and occupancy-only history sampling fails to show a reliable serving
win. Two full-state retention arms remain unrun for storage, not scientifically
rejected. Modern compact training changes 76,880 scale parameters across nine
projections; signs/norms remain frozen. Paid optimizer/RNG/cursor checkpoints and
successful continuations are preserved. Source remains in the pinned actual-
training worktree; newer native helpers are not yet integrated. No push.

Storage recovery consolidates byte-verified duplicate cache/raw payloads while
preserving paths. Exact duplicate CE1920 GGUF aliases the original. Three older
materializations (CE480, retention480 and head-only1920) are replaced by compact
native-scale archives with full-file reconstruction/hash proof; checkpoints and
restore instructions remain. These are recoverable archives, not currently
materialized GGUF files. No unique paid state is deleted. Current free-space
snapshots are volatile and recorded separately; storage remains a practical risk.
The canceled background audit has no recovered final report and is not counted
as completed. Later focused source/native audits supply the findings above.

All references in this checkpoint without an absolute path are relative to
`runs/w1a8-modern-protected-20261103-01/`. No Phase 1 completion, predictive-screen
validation, backlog classification, survivor combination or four-target W1A8 win
is claimed. Historical F16 remains separate from modern Qwen3.5 Q4_K_M/NVFP4.


## Frozen fresh confirmation launched after check-in — November 3

The preceding turn was a status-only check-in, without new experimental evidence.
The explicit goal continuation now resumes real work. Root revalidates idle RTX
5080, pause=false, terminal prior jobs and available storage, then completes the
post-selection exclusion inventory from 60 explicit already-used source files:
512 unique TRAIN prompts, 12 development, five pilot and 147 old-matrix prompts.
Raw extraction/deduplication counts and pins are retained; indices cannot by
themselves prove global exclusion completeness. No unopened sealed data is read.

The generator admits 24 new performance and 24 quality instances, zero exact or
normalized overlaps. All independent positive/semantic-negative judge controls
pass. Model/runtime selection remains frozen. Independent bounded read-only
source audit finds no run blocker, but confirms narrow Python task variation and
the need for 12-family-cluster uncertainty. No broad task/pretraining novelty
claim is justified. The exact frozen five-arm file is supplied, avoiding the
adapter's permissive subset admission.

Root launches `local-w1a8-modern-fresh-quarter-20261103-01` through the existing
180s-grace supervisor/local tmux socket. Live boot/birth/cmdline identities and
stop procedure are in that run's identity.json. Sole GPU owner=root; no tests,
builds, compression or unrelated large hashes during timing. Output:
protected run `fresh-quarter-confirmation01/serving01/`. Native tokenization
already verifies all 48 tasks plus output reserve fit context4096. Expected600
performance,120quality,60warmup requests across30startups; no complete outcome yet.

Before launch, one unused retention10-1920 GGUF is copied to C:, whole-file hash
verified, then the original WSL path atomically replaced with a symlink. This
recovers40,804,352WSL bytes without changing payload or paid checkpoint. Receipt:
`retention10-export-relocation.json`. Timing protocol/source/candidate is unchanged.


## First untouched-instance transfer confirmed — November 3, 16:22 UTC

Fresh supervisor exits0; all30server groups reaped/absent and compute released.
Root independently audits all780raw requests:600performance,120quality,60warmups,
all expected five arms, tokens/text/stop/repeat/cache/count accounting and40loaded
images. All576speculative measured responses equal target. No zero-output case.
Candidate140.317tokens/s versus init133.261: +5.295%, paired12-family-cluster95%
interval+4.547–6.019%; gains code5.70%,reasoning5.07%,prose5.07%. Matched DFlash
262.857,target176.780,Q4_0EAGLE235.493. Candidate/DFlash53.38%, candidate/target79.37%.

Independent answer scores13/24 for EVERY arm, same case outcomes:code5/8,
reasoning1/8,prose7/8. Root reruns all120actual judgements and24positive/50negative
controls. Exact agreement preserves target quality; absolute quality is weak.
This is first positive post-selection-instance transfer, limited to these12related
synthetic families (Python public algorithms change little), not broad novelty.
The candidate still loses1/549old native draft tokens. Phase1 remains active.

Runner wall907.79274s; supervisor interval16:06:55.942491–16:22:04.180032UTC.
No training cost in this confirmation. Detailed pins/results:
`fresh-quarter-confirmation01/{root-analysis,consumed}.json`; compact tracked
`fresh-confirmation.json`. Frozen input files remain unchanged; consumed receipt
requires excluding all48cases from later untouched tests. No fresh labels enter
subsequent training. The paired interval is across12families, not24independent
prompts. Five repetitions alone do not prove broad generalization.

New isolated reachability trainer/tests pass28related trainer tests;31actual-data/
fresh/confirmation tests also pass. pytest is unavailable; standard unittest runs
all59successfully. Original native C++/builder source matches compiled snapshot.
Source committedb361cdd5 and integrated; no old helper/source contracts changed.
Worktree remains pinned for actual continuations; no push.

Root launches `local-w1a8-modern-reachability-20261103-01` under existing180s-grace
supervisor/local tmux. Its identity.json records live boot/birth/cmdline and stop.
Static/current-prefix-union arms both start at original balanced1920 and continue
to3840 with original19200horizon, sameAdam/RNG/order/all356FIT replay/lambda10/CE4.
Before treatment, static control must reproduce prior nativefit3840 scales,
optimizer,RNG,cursor,exposure and native export exactly. All549masks remain evaluated;
new CE mass/counts logged explicitly. Output on C: in reachability-{static,union}3840-01.
No serving quiet window during this training. No result yet; preserve all states.


## Matched reachability training and native admission complete — November3

Static control exactly equals prior nativefit3840: scales,Adam,RNG,cursor,exposure
and native GGUF. Both stages finish0; control166.90245/treatment166.64348 new trainer
seconds,258.96274/257.52050 stage whole-job wall,517.53099 driver wall. Original
warm-start cumulative costs remain separate. Union adds1272supported labels over
4561static (+27.9%CE mass under fixed4), same3840ordinary windows/1920replay windows/
2071protected exposures. This is a mask-plus-effective-weighting change, not a
normalization-independent causal claim.

Full model readouts: fit prefix1436→1466, oldnative loss2→3; development489→487,
loss42→40. Immutable original actor/native intersection losses2→3FIT and40→38dev.
Do not use after-readout protected fields to count old losses: they are current
intersections; root analysis joins immutable reference02 instead. All549old native
positions remain evaluated. Evidence: reachable-{result,control-exact-parity,
training-analysis}.json and C: reachability-{static,union}3840-01 states.

Actual packed-native replay across all549successes retains507/549union (42lost),
parent549/549,0uncalibrated. Native run81.31783s. Two actualHTTPqualification cells
pass28responses, accepting341static/345union per12dev prompts; no single-pass
speed claim. HTTPqualification wall27.66450s, sourcebound nine-projection receipt
and loaded-image/release evidence retained. Root independently audits all28raw
signatures, stopped/count/cache fields and quiet-loader proof.

Seven-arm repeated-serving first attempt stops BEFORE anyGPUcell/request after
3.50swrapper wall: new proof metadata incorrectly retained log_verbosity instead
of the validator's normalized runtime contract. Failure/packet preserved. New
normalized proof files and modelsquiet02 correct only qualification metadata.
Root now alone owns local-w1a8-modern-reachability-requests-20261103-02 under
180sgrace/localtmuxsocket. identity.json authenticates live boot/birth/cmdline/stop.
Expected420measured+70warmups/35startups. Quiet window forbids CPUtests/builds/
compression/unrelated hashes. Conservative quarter model, initial, target,DFlash
andQ4 controls unchanged; compare full nativefit3840 static versus union explicitly.
No complete result yet. No goal/phase completion or backlog rejection.

Separately, root recomputes all8fresh reasoning expected answers from public
prompts using independent set/modular/path/permutation enumeration. Every frozen
expected answer matches; low1/8target reasoning score reflects actual answer/
instruction failures, not an expected-answer defect or output-limit stop.
Evidence: fresh-quarter-confirmation01/independent-reasoning-oracle-audit.json.
Selected-O sign-training feasibility advice is saved in selected-sign-source-advice.md;
new helper/12tinyCPUfixture tests are prepared by a bounded worker, but actual
real-model selected-sign training/export/native admission remains unproven.


## Reachability serving inconclusive; selected-sign plumbing admitted — November3

Repeated panel02 exits0:420measured/70warmups/35startups, all exact token/text/stop/
repeat matches; root rehashes40images and verifies35releases. Whole runner380.33982s.
Rates target153.395,DFlash193.186,init109.520,quarter114.673,static3840126.783,
union3840125.318,Q4_0154.285tokens/s. Union/control0.988446; whole-prompt95%interval
0.97360–1.00423. Full-prompt/repetition-block resampling widens it to0.93245–1.02817:
late repetition's added HTTPoverhead matters. All measurements retained; no point
removed. Native-reported timing components are diagnostic, not substitutes for full
HTTPwall. More acceptance (345vs341) does not establish total-throughput benefit.
Union is inconclusive at this3840checkpoint, not a permanent slow-learner rejection.
Native preservation still loses42/549old tokens. Original19200horizon/checkpoints
remain. Evidence native-analysis-reachability.json,reachability-timing-robustness.json.
Fresh candidate's joint12family/5repetition resampling also remains positive:
1.052950ratio,95%interval1.043254–1.062092. This supplements the narrower fixed-
repetition family interval without altering pinned original raw results/receipts.

Selected-O helper, qualifier and14CPUtests committed01d36810 and integrated after
Ruff format/check and full14tests pass. Original helpers unchanged. Qualified actual
GPU run selected-sign-qualification-20261103-01 completes52.98314s,3physical/2unique
updates1.57891s total. Both exact resumable checkpoints79,603,493bytes; measured peak
CUDA allocated3,149,585,920bytes. Zero-updateGGUF equals initializer. Finite nonzero
STE gradients move6,553,575of6,553,600latents, meanabsolute displacement.00131586,
max.00199848, ZEROlearned hard-sign flips. Forced diagnostic changes exactly one
whole-file bit, with allscale/other bytes identical; never a trained candidate.
Root independently rehashes all3exports and recomputes whole-file XORone-bit proof.

Actual native admission completes48.93730s across four tiny native processes plus
28fullHTTPresponses. All16reached proposals agree with GPUactor, zero calibrated
old-success loss on these two reused FIT windows; both14response cells equal frozen
target gold. Root audits28raw response/count/cache signatures,2server releases,
31loaded images and nine packed receipts/model. This is bounded plumbing only:
no actual learned sign-capacity, newquality, repeated speed or generalization claim.
Evidence selected-sign-root-admission.json and tracked selected-sign-admission.json;
C:qualified model/checkpoint/source snapshots retained. No job remains live.

A source-only next trainer is being prepared: same originalbalanced1920 migration,
9scaleAdam moments/RNG/order/all356FIT replay/staticCE/CE4/horizon19200. False mode
must discardOgrad beforeclip/step and exactly reproduce prior nativefit3840 before
true mode changes selected signs. SignLR1e-3 is separately declared; zero early
flips must be slow/inconclusive, not a clear model failure. No actual capacity arm
has trained yet. Source/paid checkpoints preserved and no push.

Read-only data audit locates476remaining TRAIN groups (159code/159prose/158reasoning)
after current36. Existing24FIT traces total396,021,761bytes/120.12s. Twelve newFIT
is an estimated189MiB/one capture minute, excluding preparation/geometry overhead.
Root corrects audit's initial C:versusWSL capacity conflation and initialization/
capture-parent conflation; worker original retained. C:direct storage appears
feasible for12or24 at current margins; WSL cannot stage dense arrays. New data
needs versioned roster/collection/count gates and an audited source-index union;
trained-parent captures cannot be silently relabeled as initializer captures.
No new data collected. Audit:fit-expansion-source-audit.json.


## Actual selected-sign capacity comparison live — November3

New isolated trainer/source83006337 integrated after root10focused CPUtests and
Ruff pass; total83related CPUtests. No original actor/data/trainer source changed.
Root launches local-w1a8-selected-sign-training-20261103-01 via180sgrace/localtmux;
identity.json captured immediately atlaunch with verified live boot/birth/cmdline.
Single rootGPUowner, no native serving quiet window. Source/plan/driver pins in
selected-training-plan.json. Both modes start originalbalanced1920, scales/Adam
moments joined BY PATH; cursor/RNG/exposure inherited. StaticCE/fixed4/all356FIT
replay/λ10/order37/19200horizon unchanged. SelectedOsignLR1e-3 versus scales1e-5.
False mode clearsOgrad BEFORE globalclip/step and never createsOmoments; it must
exactly reproduce existing nativefit3840 weights/scaleAdam/RNG/cursor/exposure/
readoutdecisions/nativeGGUF before true mode starts. No DEV/fresh labels.

Control has reached3840 and saved checkpoint metrics with zero sign movement/
flips. Final readouts/control proof pending; treatment has not started at this
checkpoint. Outputs C:/.../w1a8-selected-sign-{control,learn}3840-01, periodic480
compact checkpoints; estimate~0.51GB botharms. Actual costs/results pending.

WSL headroom was near its guard, so root byte-verifies/copies oldretention1920
GGUF toC and atomically aliases its original path. Paid PTfiles retained and all
paths remain readable. WSL actualfree+40,804,352bytes,C:-40,804,352bytes; receipt
retention1920-export-relocation.json. No VHDXcompaction/shared cleanup. New source
worker prepares a separate audited actual-manifest union helper onCPU only,
leaving currentdata/sourcepins immutable. No expandeddata collected or admitted.


## Selected-O training finishes; mature-LR control and union identity pass

Selected-sign comparison exits0 in579.31095sdriverwall: false287.04448s, true
290.98034s. False matches old nativefit3840 rawoffsets/scaleAdam/RNG/cursor/exposure/
allreadoutdecisions/nativeGGUF exactly; Olatentsoriginal and Omomentsabsent. New
trainer seconds172.94550control/175.00124true, same4561CElabels/3840ordinarywindows/
1920replaywindows/2071protected exposures. All80MB/27MBcheckpoint states saved.

True model actualpacked bits570, independently verified from exportedO payload;
source actor metrics0flips at480newupdates,71at960,298at1440,570at1920. Meanlatent
displacement.07862,max.60296,85saturated. FITprefix1436→1469,DEV489→494; old native
model-side losses2→4FIT/42→44DEV. Actual all549native replay loses49 (parent0),
85.02097s. Actual14HTTPresponses exactly equal target,343accepted/1144proposed vs
static341/1148. Single pass is not throughput proof. No new model is promoted or
fresh-confirmed. Zero short-run flips are not a clear capacity failure; source
advice recommends a paired mature-LR continuation before larger capacity changes.

Root launches ORIGINAL strict-resume LR1e-3 control from global3840→4320, all state/
Adam/RNG/masks/data unchanged, horizon19200. This finishes0; FITprefix1529,DEV499,
oldloss2/44,727Obits. Incremental480training47.24414s, wholejob136.11254s. The3e-3
paired treatment needs a NEW explicit migration validator/source and is not yet
launched. Source/case/checkpoint/hashes retained in selected-lr-plan.json and
local-w1a8-selected-lr-control-20261103-01. No GPU job remains live atcheckpoint.

Data union47d87c11 integrates after9rootCPUtests/Ruff. Real idempotence admission
uses originalmanifest and byte-identical COPY of the same capture, not two new
independent captures. Newunion schema/reader preserves2138windows,36groups,2234raw
files/10055rows/1440embeddingIDs,all356FIT+193DEV protections. Allwindow metadata/
masks joined unchanged; actual materialization of36max-anchor windows and1440
embeddinglookups exactly equals original. CLIaudit14.96205s, rootCPUparity1.92223s.
No densearrays copied, no examplesadded, no expandeddata training/native admission.
C:identityqualification sourcecopy/union/proofs retained; viewcontracts explicitly
reject oldV1reader until a new trainer/reference migration is reviewed.

Bridge source audit corrects interpretation:3smokes67/408accepted/proposed in both
original/control and ownership-fix, no measured benefit. Source proves a
conditional borrowed-buffer lifetime bug, but traces do NOT establish this branch
fired. Expected positional bridge/cadence differences and the source lifetime bug
are separate; do not attribute all conditional/serving mismatch to that bug or mix
newcorrected-runtime data into old-lineage training. bridge-smoke-source-audit.json.

Root also records an agent-coordination lesson: a Gitcommit label was called a
sourcepin; worker checked the saved full fileSHA before proceeding. Exactmatching
parent validator remains unchanged; sourcecommit andfiledigest are separatefields.
No wrong source/GPU action occurred. No push and all original checkpoints retained.

## Full user check-in; mature-LR training complete — November 3

User asks to wait and account for everything since ordered continuation. No new
experiment starts during this check-in; the existing paired LR job finished0 at
18:36:58UTC. Current GPU compute list and owned job process query are empty.

New1e-3 migration exactly reproduces original strict-resume4320 model/Adam/RNG/
cursor/exposure/nativeGGUF/FIT+DEV decisions. Paired3e-3 ends with4981Obits versus
727, FITprefix1577versus1529,DEV509versus499; old model-side native prefix losses
2/45versus2/44. Incremental480training47.74737/46.26450s, paired driver288.36723s.
These are actor readouts only: neither mature-LR arm has actual native preservation,
repeated full-request timings or untouched confirmation. No candidate promotion.
Evidence: selected-lr-result.json,selected-lr-control-parity.json,
selected-lr-root-analysis.json. All checkpoints and19200horizon preserved.

Legacy final1920 is present as the previously verified externalgzip archive,
not missing. Root rechecks compressedSHA and full3,887,338,675-byte decodeSHA in
30.25340s with no raw restore written; both pass. Rawlegacy1440 andprotected3504
parent untouched. Receipt: legacy-final-archive-current-verification.json.
This resolves the bounded inventory's initial external-path uncertainty.

Phase1 remains incomplete despite confirmed fresh+5.30% gain (joint95%+4.33–6.21%).
Phase2predictor,98ID/34family screening,combinations andfour-target finalremain
unstarted/gated.99relatedCPUtests pass. Currentfree2,231,816,192WSL and
3,274,629,120C: bytes; storage remains constrained. No push.

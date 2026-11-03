# Current project status

## Active: ordered W1A8 generalization work — November 3, 2026

Phase 1 now has positive untouched-instance transfer on the modern 4B Q4_K_M
track, but remains incomplete: preservation has one known old draft regression,
absolute answer quality is weak, and W1A8 remains far behind DFlash. All other
modern target configurations and the later ordered phases remain unconfirmed.

The frozen conservative candidate completes600new performance requests across
five arms/five repetitions plus120independently judged quality requests. Root
audits every raw response,40loaded images and30server releases. All576speculative
responses equal target tokens/text/stopping; repeated performance outputs agree.
Fresh W1A8 throughput140.317tokens/s versus initializer133.261: +5.30%, 12-family
cluster95% interval+4.55–6.02%. Gains are+5.70%code/+5.07%reasoning/+5.07%prose.
DFlash262.857,target-only176.780,Q4_0EAGLE235.493: candidate reaches only53.38%
of DFlash. All five arms score13/24 independent quality cases (5/8code,1/8reasoning,
7/8prose), with24positive/50semantic-negative judge controls verified. This supports
new synthetic instances within12related task families, not broad task novelty,
pretraining novelty or unrestricted prose quality. Fresh cohort is consumed and
excluded from future confirmation. Whole runner wall907.79s.

Earlier aggressive balanced1920 training gives+13.44%reused-development speed,
but damages more old successes. Conservative original-head/quarter-body choice
preserves548/549native old successes, all356FIT and192/193development; one loss
remains explicit. Checkpoints and isolated runtime/data contracts survive.

The static/current-prefix-union comparison finishes both continuations. Static
control exactly reproduces prior full state/export. Union adds1272supported CE
labels (+27.9%mass): fit prefix1436→1466, development489→487; old native model-side
losses2→3fit and42→40development. Native replay independently loses42/549old
successes;28HTTPqualification responses match target, accepted drafts341→345 on
the12reused development requests. No transfer or speed benefit established yet.

The seven-arm panel completes420measured+70warmups, all exact,35server releases,
40images independently rehashed. Union125.318tokens/s versus static126.783: ratio
0.98845. Joint prompt/repetition resampling gives95%interval0.93245–1.02817, so no
speed benefit is established. All measurements are retained, including late-round
HTTPoverhead variation. Union remains inconclusive at this checkpoint; continuable
states survive. Conservative fresh gain remains positive under joint family/
repetition resampling: +5.30%,95%interval+4.33–6.21%.

Separate attention-output sign-QAT support now passes14CPUfixtures and actual
GPU plumbing: finite nonzero sign gradient, exact compact Adam/RNG/cursor resume,
zero-export identity,79,603,493-byte checkpoints,3.150GBpeak allocated CUDA memory.
Three physical/two unique updates move signs but flip0learned bits. A separate
forced one-bit diagnostic is clearly labelled. Both exports pass16bounded native
proposal comparisons and28exactHTTPresponses. This is NOT a capacity/generalization
result. Matched selected-sign training finishes: control exactly reproduces prior
state/readouts/nativeGGUF; true mode changes570Obits after1920newupdates. Zero flips
at480 becomes71at960/298at1440. FITprefix1436→1469,DEV489→494, but old native model-
side losses2→4FIT/42→44DEV. Actual native replay loses49/549;14HTTPresponses match
but accepted drafts343vsstatic341 establishes no useful benefit or fresh transfer.

The mature-sign paired LR continuation finishes, global3840→4320/horizon19200.
Migration1e-3 exactly reproduces original strict-resume model/Adam/RNG/cursor/
exposure/readouts. Higher3e-3 changes4981Obits versus727; FIT1577versus1529,DEV509
versus499. Both actual native checks lose46/549old successes; all28HTTPresponses
matchtarget. HigherLRaccepts354/1122versus347/1135drafts on12reusedDEVrequests;
no repeatedtiming/promotion. Root evidence `selected-lr-root-native-admission.json`.

After the full check-in, explicit goal continuation resumes Phase1. Root owns
`local-w1a8-modern-fit-expansion-20261103-01` completed0:24newwholeFITgroups8/domain,
407native successes, all exact/native trace audits pass. Old36/fresh48exact/normalized
identities excluded; same6dacparent/target/runtime/80tokens. Fullcapture2079.81s;
summednativegroupstages754.36s omitted substantial setup/integrity/filesystem costs.
Independent frozen-target qualifier passes24new+2warmup exactresponses. Actualviews,
738nativeembeddings and auditedunion complete:48FIT+unchanged12DEV,3603windows,
2135embeddingIDs,763FIT/193DEV masks. Admission731.07s; no modelgainclaimed.
All407newparent successes reproduce natively. FulloldCEcontrol completes3840 and
exactlyreproduces oldstate/Adam/RNG/readouts/nativeGGUF. Whole1268.15s versus165.69s
incrementaltraining exposes file-reader overhead. NewimmutableRAMcache passesall
3807source matrices/60maxanchorwindows/all2135embeddings byteparity, with paidstates
preserved. Queueduncachedtruearm replaced aftercontrolchildcompleted normally;
intentionalcontroller-15 receipt is not a sciencefailure. Originalsource unchanged.
Separate residenttrainer dca0dd02 passes12rootCPUtests/Ruff and unchangednumeric
AST/arithmetic. Resident480update qualification exactlymatches original2400 weights,
Adam/RNG/exposure. Root owns local-w1a8-resident-expanded-ce-20261103-01,20:19:59UTC/
180sgrace, immediateperprocess identities. TrueexpandedCEarm setup islive, samewarm/
objective/old356replay/horizon19200. No expandeddata result/native956/repeatedspeed
yet. `resident-ce-GPU-control-parity.json` and `resident-expanded-ce-plan.json` hold
currentevidence. Readerqualification doesnotvalidate short-to-longprediction.
Data-union helper47d87c11 is integrated with9CPUtests/Ruff; real same-capture
identity check preserves2138windows/549protections,36actualmaterializations and
all1440nativeembedding lookups. This adds0examples and is not expanded-data
training admission. No new fresh/development training labels used.

Source is integrated with161passing related CPUtests and actual native evidence;
pinned worktree is retained for checkpoint/source contracts. Paid states remain;
older exports have verified compact archives or readable relocated aliases.
Storage remains constrained. No push. Quick-screen predictor,98-ID/34-family
backlog screen, combinations and final four-target DFlash confirmation remain gated.
Historical Qwen3 F16 is separate. [Fresh result](../experiments/w1a8-modern-native-20261103/fresh-confirmation.json)
and [detailed progress](../experiments/w1a8-modern-native-20261103/progress.md)
retain scope, costs, failures and next actions under the existing
[goal](goals/nine-model-qat-overnight.md).

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

# Current project status

## Active: ordered W1A8 generalization work — November 3–4, 2026

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

CURRENT GPU JOB: root owns `local-w1a8-full-native956-20261104-01` in localtmux,
180sstopgrace, supervisor/child birth+boot identities recorded. It compares
unchanged5760 and selected replay5760 on all883 fixed-source windows/956 prior
native successes, each with new packet-matched parent calibration. Split output
storage preserves both filesystem floors. This measures full recorded forgetting,
not a new serving timing or a zero-loss gate. Plan/launch/source/results under
`runs/w1a8-modern-protected-20261103-01/natural-replay-confirmation01/`.

The user's preservation-meaning clarification remains unanswered. No zero-loss,
absolute quality-floor or DFlash-win Phase1 exit requirement is inferred from it.
Phase2 short-run predictor,98-ID backlog screen, combinations and final four-target
confirmation remain pending. All paid checkpoints/horizon19200/source pins stay
preserved. Three aliases of a completed4320 derived export now share a verified
lossless archive and exact restore commands; paid states remain raw. This frees
39.80MB on C. Both disks remain constrained. No push.

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

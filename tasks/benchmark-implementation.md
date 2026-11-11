# Training, evaluation and capture implementation

## Goal

Make candidate development answer whether a drafter improves complete interactive
requests over released DFlash Q4KM, preserving the selected target's behavior.
Implement one supervised training/intermittent-evaluation path and one comparison
runner with development and confirmation datasets. Reuse the prompt bank across
compatible targets without confusing prompts, teacher captures or model identities.
This task tracks implementation and observable qualification; individual capabilities
are described as implemented only after their acceptance evidence exists.

## Authorized

Philip authorized completing the entire plan on the RTX 5080 after the existing run
finishes, validating FP16, preparing the prompt bank and generating roughly 2K supplemental
captures focused on missing workloads/lengths. Combine those with the current authenticated
HE4broad corpus and continue its final checkpoint in the new pipeline, preserving optimizer
state and recording the changed dataset/sampling lineage. Leave it running with hourly
health checks/repair. Do not interrupt the predecessor run. The 2080 Ti is available for
additional testing. The 6K design remains the general bank; the legacy hybrid continuation
retains its existing 15,647 sequences plus the supplement rather than silently reducing it.
Philip also authorizes reclaiming old high-volume legacy artifacts when needed. Verify
ownership, live dependencies and evidence retention before retirement. No hosted CI,
new remote or deletion of the active migration parent/data.

## Done when

The implementation is complete only when the phased acceptance criteria below pass,
including a real capture-to-resume-to-native-comparison path. CPU tests alone do not
close this task. Current documentation, command registry, agent navigation and
result closeout must describe the implemented behavior; missing capabilities stay
explicit. Delete this task when implementation and scoped closeout are complete.

## State

Owner: root; RTX 5080 allocation. Original HE4broad finished at update 220095.
Preserve that production parent and its Adam/RNG state. No production supplement or
hybrid continuation has launched yet. All artifacts: `/home/philip/binary-eagle-artifacts`.
No remote configured; checkpoints in Git remain local-only.

Measured qualification: original next-update parity bit-identical; scoped FP16 adoption
passed 64 TRAIN/32 DEV, 1024 updates/arm, 512 array checks, 120 native requests/arm;
public diagnostics plus schema-4 CUDA resume passed. Maximum bank input (14651 tokens)
completed real updates at 13.65 GiB peak allocation. Native runtime v5 is source-complete
and repairs a measured post-EOG sampler bug. Independent greedy oracle matched 664
rounds; both real native sampler algorithms passed probability/EOG tests for both
target precisions. Both 108-request sampled pilots passed; actual settings audited.
Nsight profile-03 verifies intended HE4 integer/packing kernels in all six layers per
round and their absence from controls; profiler timing is excluded.

Bank v3: 6000 TRAIN/600 DEV/nested120, exact native token fingerprints, known aliases
excluded, authentic histories. Semantic audit exposed closing pleasantries in Glaive;
curation-07 selects declared tool request/result/abstention stages. Re-admit and audit
before costly production capture. Old bank/evidence remain immutable. Supplement-v1
mechanical selection is provisional until that review: 2000 new-source prompts,
857/945/178/20 by input band, 600/1000/400 by output cap; maximum FP16 143.60 GiB.
Legacy original corpus is 15647 sequences and remains intact; no heldout bodies read.

Live smoke: 12 tasks ×3 arms ×2 repeats passed identical trajectories/files/checks.
Project-workflow builder now validates deliberate regressions and oracle repairs in
actual project modules; stronger GPU panel remains pending. Phase-specific followup
checks prevent future requirements leaking into initial tests. Full suite and campaign
coordinators exist; actual GPU transitions, failure recovery, global retention and
full-DEV selection still require qualification. See Next for remaining authorized work.
Legacy retirements recovered about 260 GiB with receipts under
`/home/philip/binary-eagle-evidence/retirement-20261111/`; preserve current data, final,
best and recent predecessors. Shared links were excluded from retirement.

### Frozen design requirements

- Two benchmark choices: short120 and full600 +12 live DEV +90 harness requests
  (30 distinct task groups ×3 prefixes). Full starts with two counterbalanced blocks
  per target and all three arms: target-only, released DFlash Q4KM, candidate. Initial
  targets Qwen3.5 4B Q4KM/NVFP4; target qualification is independent. Batch1, warm loaded
  models, greedy/thinking disabled; no cross-request prefix reuse. Live starts cold per
  task and permits prefix reuse inside a session. No model/precision substitution.
- Categories TRAIN/short/full: code1500/30/150; reasoning1000/20/100;
  documents1250/25/125; conversation1000/20/100; structured750/15/75;
  writing500/10/50. Half code editing, at most half reasoning arithmetic, fifth writing
  translation. Input bands <512,512–2047,2048–8191,8192–16384. TRAIN3720/1800/420/60;
  short60/36/18/6; full300/180/90/30. Output caps256/1024/2048 at30/50/20% with natural EOS.
  No artificial padding/truncation. Primarily English; length is not difficulty. Audit
  twenty examples/category across TRAIN/DEV, source exposure, semantic diversity and
  complexity limitations. Six thousand is an initial allocation, not overfit protection.
- Preserve original10K; group all related histories/documents/repository variants in one
  role. No heldout24/aliases in training, calibration or selection. Unknown public
  exposure is not freshness. New targets share prompts only: bind tokenizer/template,
  teacher features/taps, vocabulary/objective, verifier and export compatibility.
- Live tasks: two per family (bug, feature, diagnosis, documentation+followup, structured
  tools, revision). Actual frozen project/document context, sandboxed deterministic
  tools, protected independent phase checks; retain all failures/timeouts. Report task
  success and whole-task/model/tool/overhead clocks independently of isolated throughput.
  Use three authenticated prefixes (minimal, intended project instructions, second
  realistic project-authored harness) on the same30 task groups; this is DEV sensitivity,
  not unseen-harness proof. Disclose single-project and bounded-check limitations.
- Independent accepted/proposed oracle includes rejection/EOS/caps/zero/fallback cases;
  correction and bonus are not accepted draft tokens. Instrumented lane excluded from
  timings; clean counters independently reconcile. Sampled120×3seeds is explicit
  robustness mode after algorithm qualification; no greedy token-equality or checkpoint
  ranking on differing sampled paths. Unseen confirmation requires separately frozen
  600 requests/24 live/90 harness, independent curation and access receipts; no such
  scientific claim is authorized by this implementation task.
- Scores use total emitted tokens/total time; cluster paired uncertainty by source task,
  retaining repeats/variants. Per-target, category, length, harness, EOS/cap/failure reports.
  No pooled score can conceal a target/live regression. Two blocks and600 tasks do not
  guarantee precision or deployment transfer. Multiple finalists need simultaneous
  intervals; recipe claims need training replications. Training selects only provisional
  fast DEV leader; full DEV request incumbent remains distinct from latest recovery and
  from workflow/harness qualification. Ties retain earlier checkpoints.
- DEV diagnostics120 fixed windows are loss/top1, never acceptance. Calibrate native
  cadence near10% overhead by lengthening optimizer-time interval, keeping panels fixed;
  measure zero/interval/final. Dataset-local exposure is not lifetime visits; historical
  exposure remains unknown. Learning/exposure/cost reports must be accessible publicly.
- Captures external, immutable role/target/source identities, partial-tail policy explicit;
  FP16 adoption scoped Q4KM/HE4 only. User planning ceilings400/800 decimal GB per full
  target allocation; no global disk limit. Preserve compact measurements/source to
  regenerate figures; retiring exact arrays does not guarantee exact float regeneration.
- Campaign must survive train/eval/save/restore interruptions without duplicate updates,
  false selection or lost durable state. Retain latest recovery, provisional and full
  incumbent; globally prune unselected closed segments with receipts, never active work.
  Public entries freeze source, validate identities, report actionable refusal; no CI,
  no authored comments, strict typing and complete local checks remain mandatory.

Full historical design/review is preserved in Git checkpoint94c0cbd; current procedures
belong in docs/benchmarks.md, data.md, execution.md and domain.md, durable results in
results/ and FINDINGS. Completion means actual scoped execution, not prepared metadata.

## Next

1. Finish the maximum-bank memory/update proof, validate returned generation settings,
   run the NVFP4 sampled pilot and exercise the 120 × three-seed robustness mode. Record
   independent greedy oracle and sampled repair evidence; profile actual HE4 CUDA kernels.
2. Strengthen live fixtures with genuine project context and actual harness prefixes.
   Complete 600-request counterbalanced comparisons, 90-request harness panels and live
   tasks for both targets; retain failures. Attach full-DEV selection to durable states
   with separate workflow/harness claim status. No unseen-confirmation claim is requested.
3. Exercise campaign zero/interval/final evaluation, exact train/eval restart, evaluator
   failure, interrupted save/restore and retention. Preserve latest, provisional and
   full-DEV pointers separately. Measure overhead and calibrate cadence rather than panel
   size; publish actual data/model/evaluation costs and learning/exposure records.
4. Audit source quality and representative TRAIN/DEV examples, including complexity
   limitations. Freeze the focused 2K selection and actual payload forecast, then acquire
   it in qualified FP16 after the implementation paths have passed. Preserve data roles,
   original inputs, source identities and cleanup receipts. Do not generate the full 6K.
5. Validate the mixture against the original 15,647-capture corpus, continue original
   update 220095 with Adam intact, leave the supervised campaign running and establish
   hourly health/repair checks. Finish current docs, compact results, navigation and full
   gates; remove this task only after the complete authorized outcome is established.

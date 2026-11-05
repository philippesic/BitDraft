# W1A1 EAGLE agent rules

For current DFlash W1A8 candidate work, start with
[docs/CANDIDATE_BENCHMARKING.md](docs/CANDIDATE_BENCHMARKING.md) and its linked
baseline registry. Keep fixed references and the moving incumbent distinct.

This is a bounded research project. Use `docs/PROJECT_OVERVIEW.md` for research
direction, `docs/EVALUATION.md` for measurements, and `docs/AGENT_OPERATIONS.md`
for Git, checkpoint, and GPU procedures. Read the document relevant to the
task; do not load every document for a small edit.

- For modern Qwen3.5 4B/9B × Q4_K_M/NVFP4 work, DFlash Q4_K_M is the
  user-selected primary baseline to beat under the same correctness runtime.
  Use the frozen per-target settings in experiments/w1a8-standing-20261102/baseline.json.
  Target-only remains the correctness reference; Q4_0 EAGLE and FP16 EAGLE are
  diagnostic controls. Historical Qwen3 F16/SM75 experiments retain their frozen
  comparators; do not change target precision or retroactively relabel results.
- Favor development speed and decisive native acceptance/throughput tests over
  bit-exact cross-backend parity. Timebox numerical diagnostics and require each
  one to address a concrete training or deployment risk. Hugging Face versus
  llama.cpp floating-point drift alone is not a blocker when training uses
  captured native target features/logits and bounded native trajectory checks
  pass. Record the discrepancy and a practical numeric gate, then proceed.
  Investigate further if it changes labels, drafter decisions, gradients, or
  conclusions. Preserve exact data ancestry, cache/mask semantics, verifier
  behavior, and held-out evaluation.
- The user owns major research decisions. Continue independent work when a
  decision is pending; record options and evidence in `docs/DECISIONS.md`.
- Keep one active goal at a time. Its durable state lives in `docs/STATUS.md` and
  its own file under `docs/goals/`. Keep checkpoints current as work progresses.
  Start a new goal only when the user requests one. A thread's memory or Goal is
  not the project record. Work autonomously within the active goal.
- Speak to the user briefly and technically, explaining terms
  that require project-specific background. Give periodic milestone updates and
  answer check-ins directly. Save detailed state in the goal file.
- Whenever you notice contention between the user and agents (disagreements,
  cooperation difficulties, or confusion), or a concrete user mistake, record it
  in `docs/USER_LESSONS.md`. Keep entries factual and respectful: include context,
  evidence, agent contributions to the problem, and a practical lesson for
  reminders, behavior, or clearer coordination. Distinguish confirmed errors from
  disagreements or uncertainty; update entries when corrected. Consult relevant
  lessons when needed, without treating them as overriding the user's current
  instructions.
- Work directly in the current thread. Delegate or create another chat only when
  the user explicitly requests it.
- Use temporary worktrees for isolated edits. Integrate reviewed, tested work into
  the integration branch; push when covered by user authorization, then remove the
  merged worktree and branch after checking unpublished work. Never drop unmerged
  work. Do not leave a parent gitlink pointing to an unpublished llama.cpp commit.
- Commit coherent progress regularly; push within existing user authorization.
  Commit subjects are short and simple, with no body or coauthor trailers.
- This project runs locally in WSL on the RTX 5080 machine, from
  `/home/philip/binary-eagle-decoding`. Use the local shell and GPU directly;
  verify hardware with `nvidia-smi` before experiment work. No SSH connection,
  host registry, machine-link card, or separate device checkout is needed.
  Keep experiments in unique run directories and stop owned process groups
  before reporting release. A user request to pause GPU work takes priority.
- Keep model weights, datasets, captures, and raw runs out of Git. Record versions
  and hashes in experiment reports. CPU/Metal checks do not validate SM75
  performance; identify actual hardware and precision in every result.

## Acceptance and continuity

- Continue the existing active goal and its latest checkpoint; do not restart the
  nine-model work, shorten an authorized training allocation, or infer completion
  from a prepared endpoint/metadata packet. Historical README scaffold claims
  and host defaults are not live state. Reconcile STATUS with actual job evidence.
- Avoid competing GPU jobs. Tests/builds must respect any quiet window.
- Preserve healthy training and passive endpoint watchers through unrelated
  configuration edits or context rotation. Record exact run paths, process
  birth/boot identity, source/checkpoint pins, and stop procedures so work can
  continue safely from its checkpoint.
- Require actual drafter/export/admission/native evaluation evidence. Acceptance
  rate or an isolated kernel speedup alone does not prove total-throughput benefit
  over the selected matched baseline. Keep held-out inputs, target/verifier semantics, precision,
  hardware, data ancestry and matched baseline immutable within a comparison.
- Refresh durable state after compaction and continue useful authorized work in
  the current thread.

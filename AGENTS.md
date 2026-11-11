# Binary Eagle

Trains and exports HE4 drafters for speculative decoding: W1A4 weights, A8 down
projection, Q4K head. The central boundary is data roles: heldout24 and its aliases
never enter training, calibration or selection. Names and metrics: [domain](docs/domain.md).

## Where things are

- [README](README.md): setup. [Architecture](docs/architecture.md): packages, owners, entries.
- `w1a1_eagle/`: product. [Data](docs/data.md), [execution](docs/execution.md).
- `tools/`, `policy/`, `tests/cases/`: checks and their rules. [Checks](docs/checks.md).
- [FINDINGS](FINDINGS.md): result index. [Development](docs/development.md): workflow.
- `tasks/`: open multi-session work, one file per task.

## Code

- No comments or docstrings. Put intent in names and types, rationale in the topic doc.
  Machine directives need an entry in `policy/exceptions.toml`.
- Strict typing everywhere; make invalid states unrepresentable at construction.
- Product code never imports `tools` or `tests`. Each kind of state has one owner module.
- Tests are small and deterministic: inject clocks, use small fakes, run on CPU.

## Validation

- While working: `uv run --locked --no-sync --offline python tools/check.py` (about 3 s
  warm). It is partial: no numerical test, required-test roster or rule proofs.
- Commit gate: the same command with `--full` (about 70 s). Enable the `.githooks/` hook with `git config core.hooksPath .githooks`.
- There is no hosted CI; the hook and agents' own runs are the verification. Do not add CI.
- Fix the cause. Never weaken a check, update a digest or add an exception only to pass;
  if a pin or gate blocks legitimate work, stop and report it with the proposed change.

## Working autonomously

- Without asking: repairs, tests, refactors, docs and local commits within the scope a
  task file or Philip authorized. Finish authorized work; do not wait for repeat approval.
- Commit checkpoints without coauthor trailers. Push to the user-owned remote at
  milestones and handoff; report local-only work.
- Needs Philip: weakening a guarantee, changing scientific semantics or data roles,
  new training allocations, data transfer, deleting artifacts you did not create,
  creating remotes or publishing (license is undecided).
- CPU work is autonomous. GPU work follows task scope; there is no blanket prohibition.
  Check GPU activity and task ownership before starting. Preserve other tasks' jobs.
  An established GPU allocation belongs to that task until completion or release;
  coordinate its agents within that allocation. Explicit user limits still apply.
- `/home/philip/binary-eagle-decoding` and `vllm-semantic-cache-ext` are read-only references.
- Parallel work: use a separate worktree. Preserve uncommitted work you did not write,
  and commit it only after confirming the thread that wrote it has finished.

## Documents

- `docs/`: present behavior and procedures; replace obsolete instructions.
- `results/`: durable learning. [Format and retention](docs/results.md).
- FINDINGS.md: short conclusions and result links, not full writeups.
- `evidence/`: immutable existing receipts; large artifacts live outside source.
- `tasks/`: unfinished work; follow [convention](tasks/README.md). Delete when done.
  An open task is context, not authorization to resume it.
- For investigation closeout, read [close-results](skills/close-results/SKILL.md).
  Routine repairs need no result. Git keeps implementation history; no diaries.
- Read relevant findings and tasks before starting. Write plainly.

## Evidence and reporting

- Keep weights, captures and run outputs out of Git.
- Keep failed and unfavorable results. Done means exercised, not written.
- Say what ran and what was reused or skipped. A CPU pass says nothing about native
  execution, hardware or performance. Report operational and scientific health separately.
- Never mix clocks, eras or workloads in a comparison.
- A hash, receipt or editable approval field grants no authority.

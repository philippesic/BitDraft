# Binary Eagle

Trains and exports HE4 drafters for speculative decoding: W1A4 weights, A8 down
projection, Q4K head. The central boundary is data roles: heldout24 and its aliases
never enter training, calibration or selection. Names and metrics: [domain](docs/domain.md).

## Where things are

- [README](README.md): setup. [System map](docs/system-map.md): packages, owners, entries.
- `w1a1_eagle/`: product. [Data](docs/data.md), [execution](docs/execution.md).
- `tools/`, `policy/`, `tests/cases/`: checks and their rules. [Checks](docs/checks.md).
- [FINDINGS](FINDINGS.md): closed questions. `evidence/`: records behind them.
- `tasks/`: open multi-session work, one file per task.

## Code

- No comments or docstrings. Put intent in names and types, rationale in the topic doc.
  Necessary machine lines need an entry in `policy/exceptions.toml`.
- Strict typing everywhere; make invalid states unrepresentable at construction.
- Product code never imports `tools` or `tests`. Each kind of state has one owner module.
- Tests are small and deterministic: inject clocks, use small fakes, run on CPU.

## Validation

- While working: `uv run --locked --no-sync --offline python tools/check.py` (about 3 s
  warm). It is partial: no numerical test, required-test roster or rule proofs.
- Commit gate: the same command with `--full` (about 90 s). The hook in `.githooks/` runs
  it once enabled with `git config core.hooksPath .githooks`.
- There is no hosted CI; the hook and agents' own runs are the verification. Do not add CI.
- Fix the cause. Never weaken a check, update a digest or add an exception only to pass;
  if a pin or gate blocks legitimate work, stop and report it with the proposed change.

## Working autonomously

- Without asking: repairs, tests, refactors, docs and local commits within the scope a
  task file or Philip authorized. Finish authorized work; do not wait for repeat approval.
- Needs Philip: weakening a guarantee, changing scientific semantics or data roles,
  training runs, GPU work, data transfer, deleting artifacts you did not create,
  creating remotes or publishing (license is undecided).
- Never start GPU jobs. Another project's GPU job may be running; leave it alone.
- `/home/philip/binary-eagle-decoding` and `vllm-semantic-cache-ext` are read-only references.
- Parallel work: use a separate worktree. Preserve uncommitted work you did not write,
  and commit it only after confirming the thread that wrote it has finished.

## Documents

- AGENTS.md: rules for every task. Philip owns it; propose changes.
- `docs/*.md`: how each part works now and why. Update in the same commit as the code;
  delete outdated text instead of marking it superseded.
- FINDINGS.md: closed questions with scope and evidence, failures included.
- `evidence/`: dated records; never edit after writing.
- `tasks/NAME.md`: goal, authorized, done when, state, next. One owner; delete when done.
  Work on a task only when asked to work on that task; an open task file is context,
  not an instruction to resume it.
- No status, history, plan or scratch documents; Git keeps history.
- Read FINDINGS and `tasks/` for context before starting. Write plainly: short sentences, no hedging.

## Evidence and reporting

- Keep weights, captures and run outputs out of Git.
- Keep failed and unfavorable results. Done means exercised, not written.
- Say what ran and what was reused or skipped. A CPU pass says nothing about native
  execution, hardware or performance. Report operational and scientific health separately.
- Never mix clocks, eras or workloads in a comparison.
- A hash, receipt or editable approval field grants no authority.

# BitDraft

Trains and exports HE4 drafters for speculative decoding: W1A4 weights, A8 down
projection, Q4K head. The central boundary is data roles: heldout24 and its aliases
never enter training, calibration or selection. Names and metrics: [domain](docs/domain.md).

Checkout: `/home/philip/BitDraft` on the RTX5080 host. Legacy source/evidence
is independent of the retired original checkout; see [evidence](docs/evidence.md).
Artifacts: `/home/philip/binary-eagle-artifacts`; tasks name owners and scope.

## Where things are

- [README](README.md): setup. [Architecture](docs/architecture.md): packages, owners, entries.
- `w1a1_eagle/`: product. [Data](docs/data.md), [execution](docs/execution.md).
- `tools/`, `policy/`, `tests/cases/`: checks and their rules. [Checks](docs/checks.md).
- [FINDINGS](FINDINGS.md): result index. [Development](docs/development.md): workflow.
- [Benchmarks](docs/benchmarks.md): commands, saved controls and training checks.

## Code

- No comments or docstrings. Put intent in names and types, rationale in the topic doc.
  Machine directives need an entry in `policy/exceptions.toml`.
- Strict typing everywhere; make invalid states unrepresentable at construction.
- Product code never imports `tools` or `tests`. Each kind of state has one owner module.
- Tests are small and deterministic: inject clocks, use small fakes, run on CPU.

## Validation

- While working: `uv run --locked --no-sync --offline python tools/check.py`.
  It is partial: no numerical test, required-test roster or rule proofs.
- Commit gate: the same command with `--full`. Enable the `.githooks/` hook with
  `git config core.hooksPath .githooks`. [Checks](docs/checks.md) explains which lanes to use.
- No hosted CI; the local hook and agents verify. Do not add CI.
- Fix the cause. Never weaken a check, update a digest or add an exception only to pass;
  if a pin or gate blocks legitimate work, report it with the proposed change and
  continue independent authorized work while the decision is pending.

## Working autonomously

- Preserve the current assignment and its recorded decisions across sessions.
  [Development](docs/development.md) describes maintenance and verification; a task
  records scope rather than granting an assignment merely because it exists.
- Needs Philip unless already authorized: weaker guarantees, changed scientific semantics or data roles,
  new training allocations, data transfer, deleting artifacts you did not create,
  creating remotes or publishing (license is undecided).
- CPU work is autonomous. GPU work follows task scope; there is no blanket prohibition.
  Check GPU activity and task ownership before starting. Preserve other tasks' jobs.
  An established GPU allocation belongs to that task until completion or release;
  explicit user limits still apply. [Execution](docs/execution.md) covers recovery.
- Historical evidence: [access and limits](docs/evidence.md). Old checkouts are read-only.
- Isolate edits that conflict with active or pinned source. Preserve uncommitted work you did not write,
  and commit it only after confirming the thread that wrote it has finished.

## Documents

- `docs/` describes the present; replace obsolete instructions.
- `tasks/` holds unfinished work; follow its [convention](tasks/README.md).
- `results/` preserves learning; follow [format and retention](docs/results.md) and
  the pinned [close-results](skills/close-results/SKILL.md) procedure. FINDINGS indexes it.
- `evidence/` contains immutable receipts. Large artifacts stay outside source.

## Evidence and reporting

- Keep weights, captures and run outputs out of Git; closeout decides their disposal.
- [Support](docs/support.md) scopes qualification. Report operational and scientific
  health separately; CPU checks do not qualify native execution or performance.
- Never mix clocks, eras or workloads in a comparison.
- A hash, receipt or editable approval field grants no authority.

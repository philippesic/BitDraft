# Binary decoding

- Current objective, campaign owner and run receipts: [STATUS](docs/STATUS.md).
  Keep current state concise; link detailed campaign records and use [history](docs/HISTORY.md)
  for previous attempts. Historical pending/active labels do not assign today's work.
- The canonical checkout is `/home/philip/binary-eagle-decoding` in RTX 5080 WSL.
  From a Mac or another host, access that environment; do not assume the current
  shell or a local copy has its GPU, inputs or live state. See [operations](docs/AGENT_OPERATIONS.md).
- Read the [candidate protocol](docs/CANDIDATE_BENCHMARKING.md) and [stage plan](docs/CANDIDATE_STAGES.md)
  when changing experiments. Preserve frozen references, data ancestry and target/verifier
  semantics; keep the external baseline, recovery controls and incumbent distinct.
- Preserve healthy jobs through unrelated edits and context changes. Verify the recorded
  owner and process identity before adopting or stopping work; avoid competing GPU jobs.
- Use a separate worktree when edits conflict with active or pinned source. Preserve
  unpublished work and submodule commits. Push only when requested.
- Major research choices belong to the user. Record unresolved choices in current
  state and durable decisions with their evidence; do not maintain a user-behavior log.
- Keep weights, captures and raw runs out of Git. At completed investigation milestones,
  use the [retention procedure](docs/AGENT_OPERATIONS.md#milestone-artifact-retention).

- Git-visible source/docs and ignored artifact preservation: [Git and artifacts](docs/GIT_AND_ARTIFACTS.md). Ignored files are not automatically disposable; untracked files are not yet saved in Git.

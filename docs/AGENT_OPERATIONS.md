# Agent operations

## Local repository

On Philip's Linux machine, `/home/philip/binary-eagle-decoding` is the canonical
checkout and the registered T3 project root. This session runs directly in WSL
on the RTX 5080 machine. Use this local shell and GPU for experiment work.
Existing pinned worktrees are historical or isolated source snapshots, not
separate device installations. Keep local artifacts and the Python environment
in place.

## Continuing project work

Work directly in the current thread. Delegate or create another chat only when
explicitly requested by the user. No model-specific roles, task ownership
acknowledgments, or worker rotation are required.

Keep one active goal with a concise checkpoint under `docs/goals/`, linked from
`docs/STATUS.md`. The user makes major research decisions. Record pending options
and evidence in `docs/DECISIONS.md`, and continue independent authorized work.

## Checkpoints and context continuity

At milestones, update the active goal file with actual changes, checks, results,
source/checkpoint pins, active run paths, local tmux sessions, process birth/boot
identities, stop procedures, and the next action. Keep detailed logs in run
artifacts. Refresh durable state after compaction and continue in this thread.
Preserve healthy training through unrelated edits or context rotation. Verify
actual process and run evidence before stopping or resuming an experiment.

## Git and worktrees

`main` is the integration branch. Commit coherent progress without waiting for a
large final batch. Push only when covered by the current user authorization. Use
one short subject line and no coauthors or body. For independent edits, make
temporary worktrees; share a worktree only for read-only work or explicit file
partitioning. Integrate changes into `main` after relevant checks pass, push
`main` when authorized, then remove a merged worktree and its branch. Before
removal check for uncommitted changes, unpublished commits, and live tasks. Do not
run broad `git clean` on a shared checkout.

llama.cpp is a Git submodule whose fetch/push URL is the user's fork. Its
`upstream` remote can track ggml-org. For a runtime change, commit inside the
submodule, push that commit to the fork, then commit the updated gitlink in the
parent repository and push `main`. Keep the upstream revision and experimental
baselines recorded when updating it.

## Local GPU execution

The RTX 5080 is attached to this WSL environment. Before experiment work, check
`hostname`, `nvidia-smi`, current processes and resource headroom locally. No
SSH transport, tmux MCP, host-address lookup, machine linking, code transfer, or
separate device checkout is part of the current workflow. Historical reports'
addresses, transport panes and device checkouts are provenance, not launch
instructions. The RTX 2080 Ti track is historical and outside the current task;
local SM120 evidence does not validate SM75.

Use `/home/philip/binary-eagle-decoding` as the project root. An isolated source
worktree may be used for a candidate; it is still on this same machine. Keep
models, datasets, captures and raw outputs outside Git, under their existing
local paths. Each new experiment has a unique `runs/<run-id>/` directory.

One verified operator owns the GPU lane. Preserve healthy jobs and authenticate
process birth/boot identity, run paths and source pins before adopting or stopping
a recorded job. Reconcile current process/GPU evidence rather than assuming a
historical running status is live. Do not interrupt unrelated processes.

Run long experiments through the existing `scripts/remote_job.py` supervisor
(the filename is retained for compatibility) in a detached **local** Linux tmux
session. For example, from the selected source checkout:

```sh
tmux -L binary-eagle-runtime new-session -d -s RUN_ID -c "$PWD" \
  'exec .venv/bin/python scripts/remote_job.py RUN_ID -- COMMAND ARGUMENTS'
```

Replace the placeholders and use the selected checkout's actual Python path.
Record the local tmux socket/session, supervisor and child process identities,
source/config pins, run directory and stop procedure. The supervisor creates the
child process group, stores `state.json` and `stdout.log`, and keeps its caches
and temporary files inside the run directory. Do not launch detached GPU jobs
outside this supervisor. Preserve checkpoints, accounting and useful artifacts
before removing a completed run directory.

The machine-local pause control remains
`~/.config/binary-eagle-decoding/gpu-control.json`. Existing launchers consume
this control; it does not require a host address. For a requested pause, run
`python3 scripts/agent_env.py pause rtx5080`, stop the authenticated owned
supervisors locally, and verify terminal receipts, process-group absence and
GPU release. If graceful stop fails, terminate the authenticated owned group
and verify again. A flag alone does not stop a job or prove release.

After an explicit resume, use `python3 scripts/agent_env.py resume rtx5080`
and perform fresh local process/resource checks before new work. Back up
shared control files and verify they have not changed before updating them.
Leave other devices' pause state unchanged. No new user confirmation is needed
for work already authorized within the active investigation.

Codex may ask the user to review and trust the repository hook before it runs. The
[Codex hooks reference](https://learn.chatgpt.com/docs/hooks) describes that trust
step and the `SessionStart` behavior. The [subagents
reference](https://learn.chatgpt.com/docs/agent-configuration/subagents) documents
project agent profiles; the [skills
reference](https://learn.chatgpt.com/docs/build-skills) documents repository-local
skills.

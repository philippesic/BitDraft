# Agent operations

## Goal

Decide how agents handle disk, remotes, CI and shared machine resources for this
repository.

## Authorized

Nothing yet beyond the limits below, which stay in force until Philip revises them.

## Done when

Each item below is decided, implemented where needed, and its rule lives in AGENTS.md
or a topic doc; then this file is deleted.

## State

Limits in force:

- Keep aggregate writes for the port within 1 GiB unless Philip approves more.
- Stop before WSL free space falls below 2.5 GiB.
- Delete only disposable files you created, after checking they are unused and
  recoverable.

These were set when WSL had about 3 GiB free; on 2026-11-10 it had 41 GiB free.
Earlier cleanup receipt: [disk reclaim](../evidence/disk-reclaim.json).

Candidates:

1. Replace the fixed limits with a free-space check before writes at a chosen reserve.
2. Create the private remote, then add CI that runs `tools/check.py --full`: install the
   pinned uv, sync from the lock, run the check. Add a container only if CI needs the
   numerical environment. The never-run workflow and container file were deleted.
3. Backup for retained states and captures; local presence is not a backup.
4. Coexisting with GPU jobs from other projects on the same machine.
5. Retention and cleanup policy for run outputs outside Git.

## Next

Philip picks which candidate to take first.

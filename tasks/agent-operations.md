# Agent operations

## Goal

Decide how this repository handles its remote, CI, backups and shared machine resources.
Disk space is handled by a separate operations system, not by this repository.

## Authorized

Nothing yet.

## Done when

Each candidate below is decided, implemented where needed, and its rule lives in
AGENTS.md or a topic doc; then this file is deleted.

## State

Candidates:

1. Create the private remote, then add CI that runs `tools/check.py --full`: install the
   pinned uv, sync from the lock, run the check. Add a container only if CI needs the
   numerical environment. The never-run workflow and container file were deleted.
2. Backup for retained states and captures; local presence is not a backup.
3. Coexisting with GPU jobs from other projects on the same machine.

## Next

Philip picks which candidate to take first.

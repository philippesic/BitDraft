# Agent operations

## Goal

Decide how this repository handles its remote, backups and shared machine resources.
Disk space is handled by a separate operations system, not by this repository.

## Authorized

Nothing yet.

## Done when

Each candidate below is decided, implemented where needed, and its rule lives in
AGENTS.md or a topic doc; then this file is deleted.

## State

Candidates:

1. Create the private remote. It is storage only; there is no hosted CI.
2. Backup for retained states and captures; local presence is not a backup.
3. Coexisting with GPU jobs from other projects on the same machine.

## Next

Philip picks which candidate to take first.

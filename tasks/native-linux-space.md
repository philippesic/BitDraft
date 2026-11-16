# Release disk space for native Linux

## Goal

Retire the approved obsolete WSL data and environments, reclaim the Windows disk image
and release as much partition space as practical for a native Linux installation,
targeting one decimal terabyte while keeping Windows usable.

## Authorized

Philip explicitly approved the discussed cleanup candidates and volume shrinking on
November 15, 2026. Preserve current corpora, selected/recovery states, active agent
binaries, T3 user data and worktrees, and unpublished source. Preserve source/evidence
from retired environments before disposal. Do not format a partition or reboot Windows.

## Done when

Approved cleanup is verified, WSL disk compaction is measured, and Windows partition
shrinking is measured or its remaining blocker is recorded. Remove this task after
reconciling the Windows completion records.

## State

Owner: this storage cleanup session. Exact authorization, executable operation snapshots,
per-file retirement plans/journals and verified environment source/evidence archives are
under `artifacts/receipts/native-linux-space-20261115/`. Corpus and kernel-array
retirement runs through the maintained retirement command. The full CPU commit gate
passed before environment removal. Original corpus logs and completed semantic project
output records have lossless, verified archives in the receipt folder. The original final archive
and GGUF are protected; obsolete corpus tensors lose exact training replay.

Elevated Windows inspection confirmed disk 0, partition 3, C:, a 2,000,398,934,016-byte
GPT disk with BitLocker off. The initial shrink analysis was limited by the NTFS USN
change journal to about 17 GB. Journal settings are backed up: maximum 33,554,432 bytes,
allocation delta 8,388,608 bytes, range tracking disabled. Resetting it discards old
change notifications and may require indexing or synchronization services to rescan.

The Windows continuation is `C:\Users\philip\BitDraft-space-recovery\recover-space.ps1`.
It requires administrator rights and an explicit `ready.json` marker before shutting
down WSL. It compacts the inspected VHD, consolidates free space, resets the change
journal only if it still blocks shrinking, restores its settings, and shrinks C: as far
as supported while leaving 32 GiB free for Windows. It does not delete restore points
or alter pagefile settings. `status.json`, `recovery.log`, `compact.log`, `before.json`,
`after-compaction.json` and `after.json` in that Windows folder record the outcome.

## Next

Finish and verify Linux cleanup, push the documentation, remove the approved rebuildable
BitDraft Python environment after the commit gate, trim the Linux filesystem, then release
the Windows readiness marker. WSL shutdown ends
this agent session. On reconnect, inspect the Windows records, copy them into the
project receipts, report the measured unallocated extent and remove this task when
all work is reconciled. Do not repeat deletion or shrinking without checking receipts.

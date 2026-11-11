# Completed checkpoint retirement

The user authorized purging old checkpoints/artifacts not explicitly needed on
November 6. This supersedes older blanket preservation statements only for the
exact paths in [the compact retirement manifest](summary.json).

Removed 34 optimizer-state files:

- 17 superseded precision-search states inside WSL: 68,258,710,002 bytes
  (63.57 GiB). All native exports, reconstruction dictionary bases, configurations,
  training/readout measurements and source records remain.
- 16 obsolete October 28 six/eight-update qualification/feature checkpoints on
  Windows, plus the rejected optimizer-aliasing diagnostic state from the DFlash2
  follow-up: 66,131,426,450 bytes (61.59 GiB). Failure evidence, regression source,
  measurements and the valid before/after states remain. The retired full states
  were deliberately discarded, not newly archived or backed up.

After cleanup, observed free space is approximately **66.20 GiB in WSL** and
**63.16 GiB on Windows C:**. These are separate filesystems. WSL logical reclaim
does not establish the same amount of host reclaim; the Windows deletions directly
restored host space. No VHD compaction, trim, host configuration or unrelated
archive transfer was changed.

Five protected full states passed fresh SHA256 checks before deletion and retained
their exact device/inode/size/mtime identities afterward: HE4, original Csmallbroad,
its accepted extra-30-minute continuation, B0 and the essential foundation. The
three selected native exports were fully reconstructed/read and authenticated
after cleanup. Original dense model inputs, target/baseline models, captures,
source snapshots and current dictionary dependencies remain protected.

The exact plan and per-file before/after deletion journal are under
`runs/storage-cleanup-20261106-01/`. Selection was explicit, not an age-based
recursive purge. Each file's identity and single-link regular-file status was
rechecked through its parent directory descriptor immediately before unlink.
Local process descriptors showed no selected file open; no unrelated process was
adopted or stopped. Inventories used the existing milestone helper; the historical
Windows inventory reached its 10,000-entry bound and is not a full-tree audit.

The deferred mixed FFN ternary/2-bit experiment is recorded in
[DECISIONS](../../docs/DECISIONS.md#2026-11-06-revisit-mixed-ffn-weight-precision-later).
No experiment or new training was launched during this cleanup.

[Raw plan](../../runs/storage-cleanup-20261106-01/plan.json) ·
[Verified outcome](../../runs/storage-cleanup-20261106-01/verified.json) ·
[Purpose-bound retirement command](retire.py).

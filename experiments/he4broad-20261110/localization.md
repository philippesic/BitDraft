# Local-data migration and historical model cleanup

November 10: **complete; sustained HE4broad training remains paused** at 20,671
updates. The saved parameters, Adam moments/group steps, RNG, shuffled order,
cursor and cumulative clocks are unchanged. The dedicated health event remains
disabled during Philip's explicit pause. No production continuation is claimed.

All training captures and all current-run checkpoints/exports/accounting are on
local ext4. The directories retain the old `teacher-mount/` and `archive-mount/`
names to preserve checkpoint configuration identity; neither is a mount. The
full transfer completed successfully. One-time admission verified 265,999 capture
files / 314,226,348,433 bytes across all 15,647 sequences and 1,568,435 windows,
including checksum, feature shape/finiteness, label/token alignment, window/EOG
boundaries and the pinned data index/manifest. Extra producer/source/evidence
files were copied with the complete tree as well.

The local runtime reads eight feature files and reuses labels/tokens from the
authenticated index. It no longer repeats SHA-256, small label-array parsing or
full feature finite scans on every materialization. Cheap length checks remain;
training math, objective, data/order, optimizer, target and budget are unchanged.
The two-sequence RAM cache remains bounded. Six CPU loader checks pass, including
exact old/new tensors/labels, omitted rehashing and truncated-file rejection.

## Measured short GPU diagnostic

Both arms restored the **same complete paused checkpoint**, used the same next
32 windows and took 32 actual optimizer updates. The old arm used the original
remote SSHFS loader; the new arm used local ext4 with the lean loader. Checkpoint
restore was local in both arms and outside the training-loop timing. Both arms
produced matching losses and **exactly equal final parameters, Adam state and
RNG digest**. Neither scratch state was adopted; the production recovery receipt
remained byte-identical.

| Measured quantity | Old remote | New local | Speedup |
| --- | ---: | ---: | ---: |
| Data loader, all 32 updates | 26.695 s | 0.485 s | 55.06× |
| Full loop, all 32 updates | 46.821 s | 15.214 s | 3.08× |
| Loader + update, steps 2–32 | 29.986 s | 4.480 s | 6.69× |
| Model setup + full-state restore, separate | 75.521 s | 71.257 s | — |

The first update's GPU/model time differed (15.799 versus 10.601 seconds), so
steps 2–32 are reported separately to reduce startup distortion. Their sums
exclude tiny loop bookkeeping. This is one short, serial old-then-new diagnostic;
OS caching and workload size limit extrapolation. It does not measure inference
acceptance or prove sustained 12-hour speed. In the new arm file reads cost
0.435 s; array parsing/copying 0.043 s; metadata/Python 0.006 s; repeated hashing
and full feature finite checks cost zero. File reads dominated the original
loader; removing validation alone would not have solved the remote-I/O problem.

[Interactive comparison](local-loader-comparison.html) ·
[Machine-readable timings and step records](local-loader-comparison.json) ·
[Migration / source / retirement receipts](localization.json).

## Retention and final disposition

The obsolete-model cleanup retired **443 local model artifacts, 543.36 GiB
logical size**, plus inactive reproducible CUDA caches. Logical size overstates
physical reclaim where files were hard-linked. HE4 originals/current work,
Csmallbroad's useful original and accepted continuation states, foundation/B0,
fixed target/base/comparator models and representative W1A8/DSpark baselines
remain. Results, curves, configs and compact decision evidence remain even when
an intermediate optimizer checkpoint was intentionally retired.
[Detailed retention evidence](../storage-cleanup-20261110/report.md).
[Storage inventory before cleanup](storage-breakdown.html) is a historical
snapshot, not current free space.

After local admission and the GPU diagnostic passed, the exact old
`pstack-2080:/home/philip/he4-capture-2080ti` tree was removed under Philip's
explicit authorization. No readable same-user process held its files; known
protected login/key-service processes and the retirement SSH connection were
recorded separately. Two conservative scan aborts were preserved before deletion.
The remote current-run checkpoint backup was **preserved**. This recovered about
300.45 GiB on the 2080 Ti. The 5080 ext4 filesystem had **45.34 GiB free** at the
final check, after dataset and checkpoint migration.

Both scratch workers exited successfully, the temporary comparison SSHFS mount
is gone, and all three GPU locks and the GPU are free. Local training is qualified
through `active-source.json`; the original frozen source is preserved separately.
A future sustained resume must use the complete own-run checkpoint and original
remaining allocation, never a diagnostic state or fresh optimizer.

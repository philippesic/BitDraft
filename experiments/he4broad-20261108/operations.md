# HE4broad execution and recovery

Authorized by Philip Pesic, November 8, 2026 at about 10:01 America/Los_Angeles:
28,800 seconds of actual synchronized forward/backward/optimizer updates, from
source BF16 tensors with the existing broader TRAIN manifest. This is a new recipe.

The local tmux socket is `binary-eagle-runtime`, session `he4broad-20261108-01`.
The supervisor is `scripts/remote_job.py`, run ID `he4broad-supervisor-20261108-02`,
with 600 seconds of shutdown grace. Its `runs/he4broad-supervisor-20261108-02/state.json`
records the live supervisor/child process identities. The workflow owner and
boot/birth identity are in `runs/he4broad-20261108-01/gpu-owner.json`.
The workflow holds `runs/contender-gpu.lock` across all serial training/native work.

Live progress: `runs/he4broad-20261108-01/candidate/live.json`.
Completed updates: `candidate/updates.jsonl`. Readouts and recoveries:
`candidate/readouts.json`, `candidate/recovery.json`. Segment logs:
`runs/he4broad-20261108-01/train-*.log`. Native panels:
`smoke-0`, `smoke-14400`, `smoke-28800`, `formal-0`, `formal-28800` in that run.
The source closure and its hash manifest are frozen in the run's `source/` directory.
The existing supervisor sets canonical checkout cwd for its child. Actual imported
canonical files are verified identical to the retained source closure in
`runtime-source-pin-check.json`; preserve those source files during this run.
Use a separate worktree for edits that would conflict with this pinned code.
The first setup-only attempt was stopped before model construction or any update
to replace slow WSL zero-fill allocation with verified native Windows allocation;
`setup-abort01.json` records ownership, cleanup and zero optimizer exposure.

Full checkpoints and physically preallocated slots live only in the new directory
`/mnt/c/Users/philip/binary-eagle-decoding-artifacts/he4broad-20261108-01`. A slot is removed immediately before
its checkpoint is written. Checkpoints and exports are retained; only unused
reservation files and reproducible duplicate export temporaries are removed.
No older retained state/source/export is deleted or overwritten.

To stop, verify supervisor PID, `/proc/PID/stat` birth identity and boot ID against
its current state receipt, then send SIGTERM to that authenticated supervisor.
The supervisor forwards to its owned process group. Training finishes the current
update and saves a separate paused checkpoint/export. Allow the full shutdown
grace for checkpoint I/O. The workflow releases its lock in `finally`; the
`gpu-release.json` receipt and actual processes/GPU must be checked afterward.
The machine's existing GPU pause control is checked before stages and updates.

Normal hourly resume is automatic: construct the same source model and install
the same components, load only the last HE4broad checkpoint, verify parameters,
Adam groups/moments/counters, restore RNG and preserve order/cursor/update time.
An exception during an update is uncounted; the last good full state remains.
Manual recovery after a terminal interruption must reconcile the update-log tail
against the last good checkpoint before any further work; never treat an
uncheckpointed completed tail as recoverable optimizer state. The wrapper does
not automatically restart a failed process or silently reload B0/HE4.

The initial qualification update is the first budgeted update and is retained in
the trajectory. No scratch update is discarded or used as hidden extra training.
Final verification restores GPU parameters, Adam and RNG and reproduces native
bytes without an extra optimizer step beyond the authorized budget.

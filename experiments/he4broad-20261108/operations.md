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

After the GPU supervisor exits, an owned session hook starts the one-shot CPU
`scripts/he4broad/finish.py` through `scripts/remote_job.py` with run ID
`he4broad-finalize-20261108-01`. It inventories all retained artifacts, verifies
completed-run evidence, records retention and commits/pushes only the HE4broad
report files and status section. It refuses to stage concurrent user work or
unrelated STATUS changes. `completion-callback.json` records success or a blocked
finalization/push; blocked work stays local and is explicitly labeled. The hook
then removes this run's dead tmux session. It performs no GPU updates or restart.

## Independent timer watchdog

User-authorized watchdog-only addition, November 8. The separate CPU loop
`scripts/he4broad/watchdog.py` runs through the existing `remote_job.py` supervisor,
run ID and tmux session `he4broad-watchdog-20261108-01`, on socket
`binary-eagle-runtime`. It performs an immediate check, then wakes every **600
seconds** independently of this chat turn and the training completion hook.
The recorded workflow/supervisor birth identities, actual trainer process,
`live.json` updates and update seconds, and kernel `/proc/locks` ownership of the
existing contender lock are checked. An advancing run is observed without signals,
lock acquisition, parameter/data/config edits, or relaunch.

**1,200 seconds without progress** while the process remains alive raises a stall
alarm, detected at the next timer tick. Unexpected trainer/workflow/supervisor
death, lock loss, explicit OOM evidence and failed finalization also raise alarms.
Hourly checkpoint/native transitions and a clean full-budget lock release are
handled separately from an unexpected trainer death. The timer survives the chat
turn while this WSL host stays up; it is not a reboot-start service.

Durable heartbeat and next check: `watchdog/state.json`; append-only checks:
`watchdog/checks.jsonl`; authenticated monitor/process configuration:
`watchdog/config.json` and `watchdog/owner.json`. The watchdog's own supervisor
receipts/log are under `runs/he4broad-watchdog-20261108-01/`.
Failures are retained in `watchdog/failure.json` / `latest-failure.json` and shown
prominently in this experiment's `report.md`. After any alarm, the loop checks
every 30 seconds and restores the report note if an hourly publisher replaces it.
It continues observing a stalled process without killing it; if progress recovers,
the incident stays recorded. Clean completion or an explicit user stop ends the
watchdog and its own session without touching the trainer or its clean-exit hook.

Automatic recovery attempts are **zero**: the existing supervisor has no safe
post-failure resume entry point. Its full-workflow entry point reserves a new
directory and opens zero/hourly logs exclusively, so replaying it is unsupported.
The watchdog records the last good HE4broad checkpoint receipt and leaves an
unmistakable `FAILURE_NO_AUTOMATIC_RESUME` status. It never loads B0, HE4-4096,
the overnight continuation, or random tensors. No resume mechanism or training
algorithm is added by this monitoring task.

## Hourly T3 heartbeat

The user-requested **HE4broad hourly heartbeat** is enabled with an interval of
3,600,000 ms and bound to the current thread. Its exact task ID, initial next-run
time and complete prompt are tracked in [hourly-heartbeat.json](hourly-heartbeat.json)
and copied to `runs/he4broad-20261108-01/hourly-t3-schedule/schedule.json`.
The first run is November 8, 2026 at 12:20:09 PM America/Los_Angeles.

Each invocation reads host-visible process/lock evidence, current progress and
detached-watchdog health. A healthy run posts one heartbeat line and stops. A
failure writes a prominent report note and reports it in the bound thread; it
performs no automatic resume and never starts a trainer. After clean completion
or a terminal outcome from the existing completion hook, the task posts final
status and disables/deletes itself using its saved exact ID. This hourly task
does not replace or alter the independent 10-minute watchdog or completion hook.

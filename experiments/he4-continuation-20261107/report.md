# HE4 overnight continuation, November 7–8, 2026

The user-authorized unchanged HE4 continuation was launched through the existing contender pipeline at 23:41 PST on November 7. Parent archive/raw/native verification and actual next GPU update passed; the main continuation is starting. No continuation result or headroom claim exists yet.

The only training allocation is 21,600 additional synchronized optimizer-update seconds from the exact retained HE4-confirm state at 4,096 updates. The HE4 recipe, Adam state, constant learning rates, captures, order/cursor and RNG remain unchanged. Source is frozen in the new run; the contender GPU lock and existing local supervisor serialize training, recovery qualification and native requests. All outputs use `runs/he4-continuation-20261107-01/`; the parent archive and retained frontier are preserved.

Full recoverable checkpoints and DEV loss/agreement/native exports are planned at approximately 300, 1,800, 3,600, 7,200, 10,800, 14,400, 18,000 and 21,600 update seconds. Native measurements follow trainer exit: final comparison with HE4-4096, external DFlash Q4_K_M and target-only, using existing consumed DEV3, 256-token output reserve, two warmups and five alternating repetitions. Three intermediate 64-token smoke panels remain separate. No heldout inputs are opened.

The final report will include acceptance, complete-request throughput, actual updates/time, memory, recovery proof, hashes and output paths. External Q4 remains the deployment incumbent. No mixed-weight work is authorized or performed.

## Live operation

[Launch and authenticated process identities](../../runs/he4-continuation-20261107-01/launch.json), [pipeline state](../../runs/he4-continuation-20261107-01/pipeline/state.json), and [supervisor state](../../runs/he4-continuation-20261107-01-supervisor/state.json) are the live evidence. Tmux socket: `binary-eagle-runtime`; session: `he4-continuation-20261107-01-supervisor`. Frozen source, plan, commands and per-stage logs are under the run directory. The lock is `runs/contender-gpu.lock`.

For an authorized stop, authenticate current boot/start identity against the launch and pipeline receipts, send SIGTERM to the supervisor, and allow the recorded 960-second cleanup grace. The runner forwards SIGTERM to the authenticated trainer group; the trainer completes its current update and saves its full recovery state. Verify terminal supervisor/pipeline receipts, process-group absence and GPU release. Do not use a stale recorded PID. A GPU pause also stops the existing runner at a safe checkpoint boundary.

The raw HE4 parent GGUF is intentionally archived in the original campaign. Its existing archive/dependencies are authenticated, and this run requires byte-identical regeneration in its own output directory; retained native artifacts are not replaced. Preparation used no GPU optimizer updates. A recurring follow-up monitor was rejected by automatic approval review, so no persistent schedule was installed; monitoring remains in this thread.

[Exact parent native-byte and GPU resume proof](../../runs/he4-continuation-20261107-01/candidate/zero-resume-proof.json) passed before the update budget: saved Adam step 4096, scratch next step 4097, caller optimizer unchanged.

## First continuation milestone

At 00:19 PST on November 8, the 1,800-second checkpoint/readout was complete: 14,083 additional updates, 4,096 → 18,179 total updates, 1800.101063 synchronized update seconds. The measured update log reconciles exactly through this milestone. Both 300-second and 1,800-second full recovery archives are retained, and training continues toward the authorized 21,600-second bound. [Compact progress and checkpoint/native hashes](progress.json). No continuation native acceptance or throughput result exists yet.

## One-hour continuation milestone

At 00:51 PST, the 3,600-second checkpoint/readout was complete: 28,174 additional updates, 32,270 total updates and 3600.050516 synchronized update seconds. The update log reconciles through this boundary. Three new full recovery states and four native exports (including the parent) are retained. The main first update exactly matched the qualification probe in capture, anchor, loss and gradient norm. Training continues toward 21,600 seconds. [Fresh GPU ownership/resource receipt](../../runs/he4-continuation-20261107-01/gpu-ownership-hour01.json).

## Two-hour continuation milestone

At 01:54 PST, the 7,200-second checkpoint/readout was complete: 56,359 additional updates, 60,455 total updates and 7200.119015 synchronized update seconds. The complete update log reconciles through this boundary. Four new full recovery states and five native exports (including the parent) are retained; training continues toward the unchanged 21,600-second bound. [Fresh ownership/resource receipt](../../runs/he4-continuation-20261107-01/gpu-ownership-hour02.json). No native continuation acceptance/throughput result exists yet.

## Halfway continuation milestone

At 02:57 PST, the 10,800-second checkpoint/readout was complete: 84,549 additional updates, 88,645 total updates and 10800.035910 synchronized update seconds. The complete update log reconciles through this boundary. Five new full recovery states and six native exports (including the parent) are retained. Three additional update hours remain in the original allocation. [Fresh ownership/resource receipt](../../runs/he4-continuation-20261107-01/gpu-ownership-hour03.json). Native request evaluation remains queued after trainer exit.

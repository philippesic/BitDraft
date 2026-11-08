# HE4 overnight continuation, November 7–8, 2026

The user-authorized unchanged HE4 continuation was launched through the existing contender pipeline at 23:41 PST on November 7. Preflight is running. No continuation result or headroom claim exists yet.

The only training allocation is 21,600 additional synchronized optimizer-update seconds from the exact retained HE4-confirm state at 4,096 updates. The HE4 recipe, Adam state, constant learning rates, captures, order/cursor and RNG remain unchanged. Source is frozen in the new run; the contender GPU lock and existing local supervisor serialize training, recovery qualification and native requests. All outputs use `runs/he4-continuation-20261107-01/`; the parent archive and retained frontier are preserved.

Full recoverable checkpoints and DEV loss/agreement/native exports are planned at approximately 300, 1,800, 3,600, 7,200, 10,800, 14,400, 18,000 and 21,600 update seconds. Native measurements follow trainer exit: final comparison with HE4-4096, external DFlash Q4_K_M and target-only, using existing consumed DEV3, 256-token output reserve, two warmups and five alternating repetitions. Three intermediate 64-token smoke panels remain separate. No heldout inputs are opened.

The final report will include acceptance, complete-request throughput, actual updates/time, memory, recovery proof, hashes and output paths. External Q4 remains the deployment incumbent. No mixed-weight work is authorized or performed.

## Live operation

[Launch and authenticated process identities](../../runs/he4-continuation-20261107-01/launch.json), [pipeline state](../../runs/he4-continuation-20261107-01/pipeline/state.json), and [supervisor state](../../runs/he4-continuation-20261107-01-supervisor/state.json) are the live evidence. Tmux socket: `binary-eagle-runtime`; session: `he4-continuation-20261107-01-supervisor`. Frozen source, plan, commands and per-stage logs are under the run directory. The lock is `runs/contender-gpu.lock`.

For an authorized stop, authenticate current boot/start identity against the launch and pipeline receipts, send SIGTERM to the supervisor, and allow the recorded 960-second cleanup grace. The runner forwards SIGTERM to the authenticated trainer group; the trainer completes its current update and saves its full recovery state. Verify terminal supervisor/pipeline receipts, process-group absence and GPU release. Do not use a stale recorded PID. A GPU pause also stops the existing runner at a safe checkpoint boundary.

The raw HE4 parent GGUF is intentionally archived in the original campaign. Its existing archive/dependencies are authenticated, and this run requires byte-identical regeneration in its own output directory; retained native artifacts are not replaced. Preparation used no GPU optimizer updates. A recurring follow-up monitor was rejected by automatic approval review, so no persistent schedule was installed; monitoring remains in this thread.

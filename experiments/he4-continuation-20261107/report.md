# HE4 overnight continuation, November 7–8, 2026

Preparation for the user-authorized unchanged HE4 continuation is underway. No continuation result or headroom claim exists yet.

The only training allocation is 21,600 additional synchronized optimizer-update seconds from the exact retained HE4-confirm state at 4,096 updates. The HE4 recipe, Adam state, constant learning rates, captures, order/cursor and RNG remain unchanged. Source is frozen in the new run; the contender GPU lock and existing local supervisor serialize training, recovery qualification and native requests. All outputs use `runs/he4-continuation-20261107-01/`; the parent archive and retained frontier are preserved.

Full recoverable checkpoints and DEV loss/agreement/native exports are planned at approximately 300, 1,800, 3,600, 7,200, 10,800, 14,400, 18,000 and 21,600 update seconds. Native measurements follow trainer exit: final comparison with HE4-4096, external DFlash Q4_K_M and target-only, using existing consumed DEV3, 256-token output reserve, two warmups and five alternating repetitions. Three intermediate 64-token smoke panels remain separate. No heldout inputs are opened.

The final report will include acceptance, complete-request throughput, actual updates/time, memory, recovery proof, hashes and output paths. External Q4 remains the deployment incumbent. No mixed-weight work is authorized or performed.

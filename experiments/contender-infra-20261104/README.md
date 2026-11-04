# Contender infrastructure qualification

Work in progress. User authorized direct-thread implementation and extensive
training/evaluation workflow tests; architecture/precision choices remain joint.

Raw evidence: `/home/philip/binary-eagle-decoding/runs/contender-infra-20261104-01/`.
Models: `/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01/`.

Current experiment uses the existing fixed-sign row W1A8 candidate, its original
paid5760 parent, unchanged target/runtime and consumed DEV. Original trainer and
historical frozen source pins remain intact. Separate wrapper changes only
workflow I/O when independently qualified.

The initial GPU rehearsal trains48 updates continuously, sends a real SIGTERM
during a second run, resumes its interrupted checkpoint to the same endpoint,
compares exact parameters/Adam/RNG/exposure and exports, then runs matched native
comparisons. Supervised launches and every task identity are in raw evidence.

CPU tests cover budget accounting, stage resume, stale/missing/wrong evidence,
changed inputs, timeouts, signal stop, log bounds, concurrent locks, hard-crash
cleanup and PID reuse protection; complete paired-panel validation, confirmation
consumption, data-shard eviction/corruption and sampler resume across epochs.

No new architecture, precision or final-confirmation input is selected here.

# Csmallbroad continuation showed no consistent throughput gain

## Outcome

Not consistently. It ran 9,998 updates in 1,800.073 update seconds. Request throughput
changed by -2.19% on the Q4 target and +1.54% on NVFP4. Proxy cross-entropy worsened
while top-1 rose. Two failed attempts used another 2,100.023 update seconds; in one,
aliased scalar Adam state invalidated the attempt.

## Scope and limitations

two process repetitions and DEV3 smoke. No consistent gain or ranking. The
regression source was read, not executed.

## Decision

Unresolved: retain both target results and failed attempts; do not promote the continuation as a consistent improvement.

## Evidence and reproduction

[experiments/dflash2-smoke-20261106/report.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/report.md) [experiments/dflash2-smoke-20261106/parent-review.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/parent-review.md) [experiments/dflash2-smoke-20261106/attempts.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/dflash2-smoke-20261106/attempts.md) [tests/test_gpu_resume_optimizer_state.py](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/tests/test_gpu_resume_optimizer_state.py)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. Missing reproduction details must be recovered before artifact
retirement; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

# Binary decoding

Investigate whether lower-precision speculative drafters deliver useful end-to-end
inference savings while preserving the target model's verification semantics.
The project has evolved from W1A1 EAGLE through DSpark/DFlash and broader precision
comparisons; the old architecture, hardware and time-budget assumptions are
[historical context](history/PROJECT_OVERVIEW_20261105.md), not today's instructions.

- [Current work and ownership](STATUS.md): remaining component precision and recovery.
- [Candidate comparison protocol](CANDIDATE_BENCHMARKING.md) and [stage sequence](CANDIDATE_STAGES.md).
- [Operations](AGENT_OPERATIONS.md): actual host, jobs and cleanup.
- [Decisions](DECISIONS.md) and [history](HISTORY.md): why we reached this direction.

A faster isolated kernel or better acceptance does not by itself answer the
end-to-end question. Historical studies keep their original scope and comparators.
This documentation cleanup does not change the active scientific plan or allocation.

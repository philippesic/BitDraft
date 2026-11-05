# Quick binary pipeline profiling

`bench.py` reuses the screened shared library and independent correctness checks
on historical DSpark FFN shapes, at1/7positions, three graph-timing repetitions.
`profile_ops.py` launches the corresponding operations for Nsight Compute;
counters were denied by the host driver in the recorded run.
`native.py` preserves the historical DSpark protocol and compares imma_mix versus
planes7 for one reused DEV prompt, one warmup each. Run under Nsight Systems with
CUDA graph node tracing; profiler injection must reach the child and the server
must exit gracefully. `analyze.py RUN_DIR` reads exported native-trace3.sqlite and
request UNIX bounds; it verifies native token/counter equality and writes analysis.

Use scripts/remote_job.py for each invocation, fresh run directories and the GPU
pause/ownership checks in docs/AGENT_OPERATIONS.md. Historical command receipts
live under runs/binary-pipeline-{micro,counters,native,native2,native3}-20261104.
No model or production-runtime edits. See the experiment report for limitations.

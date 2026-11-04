# Matched Q4 state-update diagnostic

`compare.py` compares the available modern Qwen3.5 4B DSpark and DFlash
Q4_K_M drafts on the same frozen target/runtime, both at depth2. It reuses
the admitted DEV12 requests and exact target responses, with two excluded
warmups per startup and five alternating repetitions. Plain and host-traced
requests must preserve all output signatures and draft/round counters.

The local `DSRequests.cell` copies the qualified request lifecycle with one
explicit DSpark CLI mapping. `DSPanel` admits only the three declared arms and
requires the same frozen runtime for all of them. No W1A8 model is run.
Run only through the project GPU supervisor using an authenticated plan.

`analyze.py RUN_DIRECTORY` recomputes costs from trace summaries and counts
state-update calls from raw spans; paired prompt and repetition resampling
provides exploratory intervals. Host spans do not isolate GPU critical paths.
The comparison is not a historical Qwen3 DSpark W1A8 diagnosis, an architecture
optimum search, or a precision ablation. The DSpark production depth3 setting
is deliberately not compared against DFlash depth2 in this matched-depth test.

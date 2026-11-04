# Original DSpark W1A8 versus DFlash Q4

`run.py` uses the retained parent3504 DSpark W1A8 export, original Qwen3-4B
F16 target, retained same-family DSpark Q4_K_M, and a downloaded compatible
DFlash Q4_K_M. No EAGLE or modern Qwen3.5 candidate is substituted.

The immutable config pins models, runtime, protocol and consumed prompt ancestry.
Six prompts (first two in each code/prose/reasoning domain) run at greedy D5,
128 output tokens, five repetitions, two warmups per startup. Plain and native
round-traced runs alternate; target-only is included. Exact historical target
IDs and freshly checked target text/finish signatures are required for every
response. Trace emissions and draft counters must match the native HTTP metrics,
and tracing must preserve the plain run's output and all counters.

The existing native request helpers preserve the historical context2048,
batch32, F16 KV and target width mask3. DSpark retains its author-layout seven
body positions and full Markov head. DFlash uses its standard block path with
five proposals. This is matched maximum proposal count, not identical body work
or independently optimized draft-length policies.

Run through the local supervisor with the pinned plan; use `analyze.py RUN_DIR`
after successful completion. Component values are native host spans without
extra synchronization, not isolated GPU critical paths or removable costs.
Small reused-prompt bootstrap intervals remain exploratory; this is not an
untouched confirmation or a win against the modern Qwen3.5 standing baseline.

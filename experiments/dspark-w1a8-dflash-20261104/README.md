# Original DSpark W1A8 versus DFlash Q4_K_M

**The retained DSpark W1A8 parent3504 is16.2% faster than DFlash Q4_K_M in this
quick matched-depth screen, and7.2% slower than its DSpark Q4_K_M control.**
No large state-update penalty appears. This is the original Qwen3-4B F16 target,
not the modern Qwen3.5 standing-baseline campaign. No EAGLE candidate is used.

## Actual comparison

RTX5080; retained native runtime and original full-F16 target/KV; target width
mask3; context2048; batch/ubatch32; concurrency1; greedy seed42; thinking off;
128-token cap; no prompt cache. Both families propose at most5tokens. DSpark
retains its author-layout seven-position body and full Markov head. DFlash uses
its standard block implementation; its trained block size is16. This is not
identical internal work or an independently optimized policy comparison.

Six already-consumed DEV prompts: first two each in code/prose/reasoning.
Five alternating repetitions, with plain/traced order alternating and two
excluded warmups per startup. No checkpoint selection using these results.

| Model | Plain full-request tokens/s | Emitted tokens/round | Guess work ms/round | State update ms/round |
| --- | ---: | ---: | ---: | ---: |
| DSpark W1A8 parent3504 |207.42|3.747|2.354|0.358|
| DFlash Q4_K_M |178.49|3.131|2.256|0.346|
| DSpark Q4_K_M control |223.45|4.240|2.677|0.354|
| Target only |94.00|—|—|—|

W1A8 versus DFlash throughput:+16.21%, exploratory paired95%interval
+9.87–+24.20%. W1A8 versus its same-family Q4 control:−7.17%, interval
−13.82–−0.46%. These are six reused prompts, not broad or untouched confirmation;
repetitions reduce timing noise but do not increase independent prompt count.

W1A8 state-update cost is+3.41% versus DFlash (interval−0.89–+8.68%) and
+1.14% versus DSpark Q4 (−3.78–+6.99%): neither difference is established.
W1A8 complete draft work including state update/acceptance is2.712ms per round,
versus DFlash2.602ms and DSpark Q4 3.031ms. W1A8 needs fewer rounds than DFlash
for the same outputs, despite its slightly higher draft work per round.

The familiar gap to same-family DSpark Q4 is reproduced: W1A8 draft work is
cheaper but produces fewer useful tokens per round. The expensive EAGLE
catch-up measurement from the modern campaign does not apply to this candidate.

## Correctness and timing scope

All320responses match historical target IDs and freshly checked target text,
finish reason and token counts, including80warmups and120measured requests per
plain/traced mode. Traced emissions map to the actual response, trace counts
match native HTTP metrics, and plain/traced outputs and all speculative counts
match exactly. Forty server releases and43owned identities are verified gone.

Component times are native CPU-wall trace spans without additional explicit
GPU synchronization. GPU work can be charged at a later wait; they are not
isolated kernel costs or guaranteed removable time. Full-request throughput
comes from the separate uninstrumented runs. Instrumentation wall changes are
unresolved: W1A8+0.40% (−1.11–+1.96%), DFlash+0.003% (−0.87–+0.59%), DSpark
Q4−1.01% (−3.12–+0.48%), target+0.23% (−0.37–+1.36%).

No training, runtime edits, precision changes or proposal-length search. Total
supervised job wall338.05s includes startup, loading and input hashing, not
pure GPU execution. GPU released; no push. This is not a claim of beating the
selected modern Qwen3.5 DFlash baseline or the best possible DFlash configuration.

## Provenance

Candidate: unchanged retained parent3504 from the October31 matched comparison.
Models/runtime/protocol and raw source pins: [config](../../runs/dspark-w1a8-dflash-20261104-01/config.json).
DFlash: [released Qwen3-4B-DFlash-b16 Q4_K_M](https://huggingface.co/Anbeeld/Qwen3-4B-DFlash-b16-GGUF/blob/5cbf9d2073be26ca6ce6ecaf1ec71731e818af11/Qwen3-4B-DFlash-b16-Q4_K_M.gguf),
322,542,144bytes, downloaded at the pinned revision and checked against its LFS
SHA256. Its source model is z-lab/Qwen3-4B-DFlash-b16. Download and server loader
receipts identify it separately from Qwen3.5 models.

[Recomputed analysis](analysis.json), [raw responses and traces](../../runs/dspark-w1a8-dflash-20261104-01/paired),
[reusable adapter](../../scripts/dspark_w1a8_compare/README.md),
[release](../../runs/dspark-w1a8-dflash-20261104-01/release.json).

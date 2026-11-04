# DSpark versus DFlash Q4 state-update cost

**The proposed DSpark catch-up penalty did not appear.** On the same modern
Qwen3.5 4B Q4_K_M target, DSpark Q4_K_M catch-up costs about25% less per round
than DFlash Q4_K_M. This does not support attributing the prior EAGLE W1A8
catch-up penalty to DSpark architecture. It also does not isolate precision's
contribution: no W1A8 arm is part of this diagnostic.

## Matched measurement

RTX5080; same frozen native runtime/libraries; correctness mode32; both drafts
Q4_K_M and depth2; F16 KV; context4096; batch512; concurrency1; greedy decoding;
same consumed DEV12 prompts,80-token cap, two excluded warmups per startup.
Five repetitions alternate arm and plain/traced order. DSpark's previously
selected depth3 is deliberately not used: this is a matched-depth diagnostic,
not a comparison of each architecture's optimized configuration.

| Generation work | DSpark Q4_K_M | DFlash Q4_K_M |
| --- | ---: | ---: |
| State update per speculative round |0.268ms|0.355ms|
| State update per actual update call |0.264ms|0.352ms|
| Guess generation per round |1.786ms|1.771ms|
| Complete draft work per round |2.054ms|2.127ms|
| Prompt state initialization per request |13.54ms|14.25ms|
| Emitted tokens per round |2.064|2.466|
| Rounds for identical4,685 emitted tokens |2,270|1,900|
| Uninstrumented full-request throughput |174.81tok/s|191.72tok/s|

DSpark catch-up is24.58% cheaper per round; exploratory paired95% interval
22.36–26.90% cheaper. Complete draft work is3.42% cheaper (2.51–4.39%). Guess
work differs+0.82%, unresolved (−0.16–+1.70%). DSpark still has lower request
throughput because fewer tokens are emitted per round and more rounds are
needed; these timings do not establish an exhaustive causal decomposition.

The host process hook is shared by DSpark and DFlash in this runtime, with
model-specific graphs underneath. These disjoint outer host spans contain
asynchronous GPU effects and are not isolated GPU critical paths. Different
models still have different target-feature widths, layers, heads and cache
work; matching Q4 does not make them otherwise identical.

## Correctness, instrumentation and limits

All420responses match target token IDs, text, stopping and truncation, including
60warmups and180traced/180plain measured requests. Traced and untraced draft
proposals, acceptance and round counts match exactly. Raw response/counter and
loaded-image receipts are retained. Catch-up call counts are recomputed from
raw trace timestamps, separate from speculative-round counters.

Instrumentation changes full-request wall by+1.73% for DSpark (paired interval
+0.36–+3.40%); DFlash−0.96% (−3.49–+2.06%) and target−0.58% (−2.21–+1.28%)
are unresolved. Therefore component figures remain instrumented diagnostics,
not zero-overhead serving measurements. Uninstrumented request rates are
reported separately. Paired prompt/repetition bootstrap intervals are
exploratory, unadjusted and based on reused inputs, not fresh confirmation.

This is **modern Qwen3.5 DSpark**, not the historical Qwen3 DSpark W1A8 candidate
that was roughly5–10% behind its same-family Q4 control. The previous0.806ms
W1A8 catch-up number belonged to modern **EAGLE**, not DSpark. The architecture
labels must remain separate. No precision-wide inevitability follows here.

No training or downloads. Supervised job wall257.28s includes loading and all
30server startups, not pure GPU execution. All33recorded owned identities are
gone; all30native release receipts exist; GPU compute is empty. No push.

[Recomputed evidence](analysis.json), [raw run](../../runs/dspark-dflash-catchup-20261104-01),
[adapter](../../scripts/dspark_catchup/README.md), and
[release](../../runs/dspark-dflash-catchup-20261104-01/release.json).

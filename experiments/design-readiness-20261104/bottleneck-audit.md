# Current fixed-candidate bottleneck audit

The remaining request-time gap falls predominantly inside the server's generation interval. The retained natural24 study does **not** separate packed drafter kernels from target verification and catch-up/cache work, so a current component diagnostic remains warranted.

This independent CPU audit authenticated all **840 measured records and1,680 raw request/response files**, verified frozen request settings/counts/timings, and checked exact tokens, text and stopping across every arm within each prompt/repetition. No GPU work or source change was performed. All rows below are the **same24 natural prompts × five repetitions**, each arm producing25,815 output tokens. Hardware/model/runtime: RTX5080/SM120; Qwen3.5-4B Q4_K_M; original mode32; context4096, batch/ubatch512, threads4, FAon, F16KV, depth2, temperature0, seed1234, no prompt cache, max256. The successful candidate uses fixed binary signs and per-output-row scale vectors over nine trainable tensors.

| Arm | Full request s | Server generation s | Server prompt s | Residual s | Request tok/s | Accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target_only | 146.219 | 132.255 | 4.784 | 9.180 | 176.550 | 0/0 |
| dflash_Q4KM | 102.163 | 85.759 | 7.210 | 9.193 | 252.685 | 15520/20265 |
| w1a8_initial | 192.564 | 172.797 | 10.522 | 9.245 | 134.060 | 5590/40020 |
| w1a8_quarter | 184.340 | 164.535 | 10.541 | 9.263 | 140.041 | 6595/38050 |
| w1a8_control5760 | 156.133 | 136.357 | 10.526 | 9.251 | 165.340 | 9995/31305 |
| w1a8_added_replay5760 | 155.619 | 135.958 | 10.501 | 9.159 | 165.886 | 10015/31280 |
| eagle_Q4_0 | 116.219 | 97.022 | 10.089 | 9.108 | 222.123 | 14905/21485 |

Selected versus paired DFlash differs by53.456s full request time, of which50.199s (**93.91%**) is server generation; prompt contributes3.292s and the residual difference is−0.034s. Selected acceptance is32.02%, initializer13.97%, DFlash76.59%. Training's previously confirmed+23.74% versus initializer is reproduced by this accounting. These are recorded accepted/proposed counters; exact emitted-draft counts and native verification rounds were not retained. Server generation includes drafting, verification, feature catch-up, cache operations and host scheduling. It is not a pure GPU-verifier timer. Residual request-minus-server time is descriptive and includes timing-boundary/transport overhead; it is not evidence of an independently removable kernel.

[Raw evidence and independent audit](bottleneck-audit.json) records exact pins, settings, methods and source links. [Existing natural confirmation](../w1a8-modern-native-20261103/natural-confirmation.json) retains bootstrap intervals and limits. No rate is compared across cohorts.

## Cheapest decisive native measurement

The unchanged server already has two useful diagnostic interfaces. At `-lv4`, `common_speculative_print_stats` reports cumulative `#calls(b,g,a)` and `dur(b,g,a)` for begin/generate/accept; subtract consecutive request snapshots. `process(batch)` is **untimed**, so the generation total omits target-feature/catch-up work outside `draft()`. All spans are host wall time with incomplete synchronization attribution; they cannot be subtracted from a separate clean run.

More usefully, `/metrics` already exports exact `llamacpp:spec_decode_num_drafts_total` (native verification rounds), `spec_decode_num_draft_tokens_total` and `spec_decode_num_accepted_tokens_total`. Capture before/after each completed request **outside** its measured HTTP clock. Per-position `llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"}` counters are also exported. This gives precise accepted/round and emitted/round diagnostics with no rebuild. Reconcile emitted-token boundaries separately; do not infer them from accepted counters. Definitions are in frozen `tools/server/server-task.cpp:1583` and `server-context.cpp:5016`.

Root can copy the selected arm's exact command from retained `natural-confirmation01/rep-00-w1a8_added_replay5760/command.json`, changing only an owned port and `-lv3`→`-lv4`, preserving model/library/settings pins. Use3–6 already-consumed DEV prompts, two excluded warmups, and a prospective10-minute cap including release. One separate diagnostic pass closes attribution gaps; use five alternating clean repetitions for any resulting throughput claim. Existing `modern_requests.py` supports explicit verbosity3/4 and should retain its five-repetition contract. Run only through the local supervisor under root's GPU ownership. This audit launches nothing.

If this still leaves material ambiguity, add bounded same-pass `process`/decode stage timing with explicit synchronization, correct target/draft context identity and no tensor output. Preserve a clean uninstrumented control. The existing `modern_trace.cpp` interposer explicitly synchronizes and fsyncs tensors, and declares its timings invalid. It is a semantics instrument, not a clean profiler.

## What source and older tests establish

Frozen `common/speculative.cpp:505` documents real waste: EAGLE `process()` runs feature encoder and decoder on the **full verify batch**, including rows rejected afterward. Current decoder also retains an unused LM-head/full-vocabulary tail on all-false-output catch-up calls (`src/models/eagle3.cpp:420`). Its measured importance needs attribution. Preserve shifted `(token[P+1],g[P])` inputs, deferred boundary and all K/V effects in any optimization.

An earlier head-only skip was actually built/admitted/tested on a different12-DEV/80-token1920-checkpoint panel: trained ratio1.003803, paired95%0.994399–1.012352; initializer1.004295,0.997829–1.011391. Both are unresolved. This is useful counterevidence against assuming the unused head dominates total cost, not a universal rejection of head or catch-up savings. Do not use those rates as current natural24 measurements. Evidence: `runs/w1a8-modern-actual-training-20261103-01/native-analysis1920-headskip.json` and `runs/w1a8-modern-headskip-20261103-01/headskip.patch`.

`shortlist_pilots/cost_bound.py` is historical corrected-D5/DSpark trace accounting and explicitly refuses causal full-request subtraction. It cannot consume current EAGLE d2 records. `binary/component_cost.py` benchmarks seven-row PyTorch F32 references, excluding native packed reads/unpacking/dispatch and full request cost. Neither supplies the missing native decomposition.

Acceptance/data work and measured per-round savings are both relevant; evidence does not yet select which kernel/runtime lever dominates. There is no justified zero-draft critical-path bound, general early-pruning rule, cross-design effect, or requirement for a DFlash win/zero draft-success loss before preparation proceeds. CPU audit time was not prospectively recorded; GPU/training cost is zero for this audit.

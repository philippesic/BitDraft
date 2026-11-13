# Drafter prompt overhead is host-side copying and state checkpoints, not drafter compute

Attaching either drafter roughly doubles the server's prompt-processing time and
cancelled the drafters' decoding gain on the live tool tasks
([standing](../he4broad-final-standing/result.md)). The extra time comes from two
runtime mechanisms around the drafter, not from its arithmetic. The target's own
kernels run at the same speed with or without a drafter, and the drafter's
context-encoding kernels take about 2% of the prefill. Both mechanisms are the same for
Baseline and HE4, so neither model can fix them. Both can be removed without changing
any target output.

1. **Feature copying through host memory (per prompt token).** For every prompt batch
   the target hands the drafter eight hidden-state rows per token (80 KB of FP32 per
   token, 42 MB per 512-token batch). The runtime copies them to host memory, waits for
   the target, copies them about four more times on the host (gather, batch object,
   batch allocator, micro-batch), then uploads them again. The GPU sits idle meanwhile.
   This costs about 0.057 ms per prompt token: most of the overhead on long single
   requests.
2. **Context checkpoints (per call, growing with context).** The target is a hybrid
   recurrent model, so the server saves state checkpoints near the end of every prompt,
   usually two per call. Both drafters declare a 4,096-token sliding window on five of
   their six layers. For a windowed cache a "partial" checkpoint copies every cached
   key/value inside the window. Each checkpoint therefore grows from 50 MiB (target
   recurrent state) to 130 MiB, copied at about 1.4 GB/s. With a drafter, a short cached
   call spends about 200 ms on checkpoints instead of about 70 ms. This dominates the
   live tool tasks, where calls add about 644 new tokens to about 4,100 cached ones.

## Outcome

### Where the time goes

Clean server timings on synthetic token prompts, RTX 5080, benchmark server arguments.
The cells are server `prompt_ms`, median of three repeats. `ckpt0` disables context
checkpoints as a diagnostic only, because live calls need them (below).

| Prompt tokens | Target-only | Baseline | HE4 | Target-only, ckpt0 | Baseline, ckpt0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 60.8 | 91.2 | 91.5 | 43.7 | 70.2 |
| 2,048 | 191.1 | 360.7 | 359.3 | 166.0 | 280.1 |
| 4,096 | 358.5 | 667.7 | 664.5 | 335.2 | 563.4 |
| 8,192 | 702.2 | 1,243.7 | 1,249.4 | 680.6 | 1,142.7 |
| 16,000 | 1,419.8 | 2,422.1 | 2,436.3 | 1,399.0 | 2,309.4 |
| Slope, ms/token | 0.088 | 0.149 | 0.150 | 0.088 | 0.146 |

At 16,000 tokens Baseline adds 1,002 ms. Checkpoints account for 92 ms of that: the
drafter's share of the 113 ms Baseline loses without them, minus the 21 ms target-only
loses. The other 910 ms (0.057 ms/token) remains with checkpoints off. HE4 matches
Baseline within 1% at every length.

Live-like cached calls: build a context, then send calls that append the previous 8
generated tokens plus new tokens. Cells are `prompt_ms`, median of calls 2–4.

| Cached context + new tokens | Target-only | Baseline | HE4 | Target-only, ckpt0 | Baseline, ckpt0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 + 32 | 83.7 | 223.3 | 228.2 | 14.7 | 19.7 |
| 4,096 + 512 | 84.3 | 182.8 | 184.4 | 55.4 | 87.9 |
| 12,000 + 32 | 60.4 | 162.8 | 162.2 | 15.8 | 18.1 |
| 12,000 + 512 | 91.5 | 193.2 | 197.5 | 62.4 | 93.6 |

With checkpoints off, a 32-token call costs 15–20 ms in every arm. With them on, the
drafter arms pay 100–140 ms more than target-only, and target-only itself pays
29–69 ms. The server's trace log confirms two checkpoints per call: 50.3 MiB each for
target-only, 130.3 MiB with a drafter once the context reaches the 4,096-token window.
Creating one took 18–27 ms without a drafter and 74–108 ms with one.

An Nsight trace of one 16,000-token prefill (instrumented, so the absolute times are
slightly inflated) gave this breakdown:

| | Target-only | Baseline |
| --- | ---: | ---: |
| Wall, ms | 1,467 | 2,430 |
| Target kernels, ms | 1,274 | 1,269 |
| Drafter kernels, ms | — | 29 |
| Feature download / upload (GPU time), ms | — | 23 / 47 |
| Main-thread host gaps over 2 ms, total ms | 142 (33 gaps) | 827 (69 gaps) |

A steady-state 512-token batch took about 47 ms without a drafter and about 70 ms with
Baseline. The main thread waits for the target batch (target kernels 38–41 ms) and
then spends 18.3 ms in host-only work before the feature upload. That upload takes
1.2 ms of GPU time and the drafter kernels take 0.9 ms. A 4.6 ms gap before the next
target batch occurs in both arms. The 18.3 ms gap matches four host copies of 42 MB at
about 9 GB/s, which is what the runtime source does
(`common/speculative.cpp` `process`, `llama_batch_ext::set_token_embd`,
`llama_batch_allocr::init`/`ubatch_add`). The trace cannot assign that time to
individual copies.

### Hypotheses tested and rejected

- **Target slowdown from exporting features:** target kernel time was unchanged
  (1,269 vs 1,274 ms).
- **Drafter compute:** 29 ms per 16,000 tokens, about 2% of its added time.
- **Batching:** `-b/-ub 2048` left Baseline at 2,395 ms vs 2,422 ms at 16,000 tokens and
  slightly slowed short prompts. The cost is per token, not per batch.
- **`GGML_CUDA_DECODE_CONSISTENT=32`:** it does reach prefill, because any nonzero value
  disables CUDA graph fusion for every batch size (`ggml_cuda_fusion_disabled`).
  Unsetting it sped target-only by 7.8% and Baseline by 5.2% at 16,000 tokens, about
  110–127 ms in each. So it is a cost on every arm, not part of the drafter's extra.
  Its single-token arithmetic for small batches also touches the final 4-token prompt
  batch and short cached segments in all arms.

### Stored data agrees

Raw response timings from `full-qualification-01` (1,212 Q4KM requests per arm):

- **Prompt slopes:** a linear fit gives 0.084 ms/token target-only and 0.147 for either
  drafter, with 17–20 ms more per request.
- **By input length:** the paired drafter excess per token was 0.07–0.12 ms below
  8,192 tokens (highest for short prompts, where per-request costs weigh more) and
  0.059–0.063 above.
- **Live calls:** paired with target-only, the drafter excess fits about
  45 ms per call + 0.08 ms per new token + 0.005 ms per cached token. Calls adding
  under 64 tokens to about 4,800 cached tokens took 67 ms target-only and 139–140 ms
  with either drafter.
- **Cache reuse:** in one live repeat, 80 of 109 follow-up calls extended the cached
  context exactly. 29 diverged, by a median of 327 tokens, and those calls need a
  checkpoint to avoid reprocessing the whole context.

### Options, ranked by expected gain

Estimates for Baseline; HE4 is the same. Live prompt time is 82.2 s against 39.7 s
target-only. I split the 42.4 s excess into a per-token part (0.057 ms × 233,682 new
tokens ≈ 13 s) and a remainder of about 29 s, which I attribute to checkpoints. That
split is inferred from the clean measurements, not measured on the live panel.

| Option | Live prompt s saved (inferred) | 16k-token request | Outputs / exactness |
| --- | ---: | ---: | --- |
| A. Keep features on the GPU: copy the target's layer inputs device-to-device into the drafter's input instead of through host memory | ~10–13 | about −600 to −850 ms of the 1,002 ms excess | Identical values; target outputs and acceptance unchanged |
| B. Take the drafter out of checkpoint copies: give its windowed layers a full, non-evicting cache (about +650 MB GPU memory at 32k context) so rollback needs no saved drafter cells, or skip the drafter cells and accept a stale window after a restore | ~20–29 | −80 to −90 ms | Full cache: identical. Skipping: target outputs unchanged, acceptance may drop after the ~27% of calls that restore |
| C. Make every checkpoint copy fast: reused pinned host buffers instead of a freshly allocated, zero-filled vector per checkpoint | most of B's gain, plus target-only's own ~30–70 ms per short call | about −80 ms (all arms gain) | Identical |
| D. Remove the redundant host copies only: gather straight into one buffer handed to the drafter | ~5–8 | about −400 ms | Identical |
| E. FP16 features | about half of A's per-token gain | about −400 ms | Drafter inputs change: acceptance may change, target outputs cannot |
| F. Narrow `DECODE_CONSISTENT` so fusion stays on for prompt batches over 32 tokens | — (all arms ~5–8% faster prefill) | −110 to −130 ms, all arms | Verification stays exact, but target prefill arithmetic changes. Target-only outputs move after near-ties ([decode consistency](../cuda-decode-consistency-cost/result.md)), so saved references must be re-established |

Not worth pursuing: larger batches (measured no gain); lazily computed or fewer
features (drafter compute is about 2%, and its one full-attention layer needs every
position); deferring drafter encoding to needed positions (same reason).

With A plus B or C, drafter prompt cost would fall to roughly its kernel time. Applying
the inferred live savings to the standing result, Baseline would go from 259 to about
217 client model seconds (target-only 257) and HE4 from 278 to about 236. Those are
projections, not measurements.

### Side finding: the request benchmark pays a prompt-cache save in every arm

Before each new independent request the server saves the previous slot's full state to
its RAM prompt cache. The benchmark sends `cache_prompt: false`, so that saved state is
never reused, but every request pays for it. The save sits outside `prompt_ms` but
inside the request wall time. The trace log measured 54–550 ms target-only and
65–700 ms with Baseline, growing with the previous prompt's length. In the stored
1,186-request logs the gap between slot selection and task start averaged 92 ms for
target-only and 105–107 ms for the drafters: 110–126 s per arm. That inflates all
request times and slightly penalizes drafters. Turning the cache off (`--cache-ram 0`)
for independent-request panels would remove it, but it changes measured times and
would need new controls.

## Scope and limitations

- **Hardware and models:** one RTX 5080 under WSL2, one runtime build (runtime-v5), Q4KM
  target, draft depth 2, F16 KV. Host copy and checkpoint speeds are specific to this
  host; native Linux with faster pageable copies would show smaller costs.
- **Prompts:** clean timings used synthetic token sequences, not DEV text. Prefill cost
  does not depend on content, but checkpoint placement at user-message boundaries only
  occurs with chat-template prompts, which these were not. Three repeats estimate
  timing noise only.
- **Instrumented runs:** the Nsight and trace-log runs were separate from the clean
  timing runs.
- **Inferred, not measured:** the per-copy split of the 18 ms host gap; every gain in the
  options table; the live split between per-token and checkpoint costs.
- **No prototypes:** no fix was built or measured. The runtime would need rebuilding,
  and each option is a proposal.

## Decision

Unresolved, proposed only. The drafter prompt overhead is a runtime problem, shared by
Baseline and HE4, and removable without changing target outputs. If live or long-input
speed matters, the next step is to prototype B or C (checkpoint copies) and A (features
on the GPU) in a pinned runtime build. Then rerun the live and long-input panels with
fresh controls. Until then, drafter evaluations should keep reporting prompt and
prediction time separately. Drafter training cannot recover this cost.

## Evidence and reproduction

- [result.json](result.json) holds every number above, the stored-data fits, the trace
  sizes and the source identities.
- [executed-prefill-probe.txt](executed-prefill-probe.txt) is the measurement script. It
  takes the project GPU lease, launches `llama-server` with
  `benchmarks.runtime_policy.arguments` and `native_environment`, and adds per-config
  flags.
- [executed-analysis.txt](executed-analysis.txt) holds the summarizers for timings,
  Nsight exports and host gaps.
- To rerun, save the script as a `.py` file under ignored `artifacts/` and run it from a
  checkout root: `PYTHONPATH=. uv run --locked --no-sync --offline python -B
  PROBE.py OUT target,baseline,he4,...`, adding `profile` for the Nsight runs.
- Executed binary `llama-server` sha256 `c83a7965…`, with libraries as recorded for
  `he4-220095-q4km-requests-01`. Target sha256 `38742092…`; HE4 export `99d2be79…`;
  Baseline `artifacts/models/dflash-4b-q4km.gguf`.
- Stored inputs (read-only): `/home/philip/BitDraft/artifacts/benchmarks/full-qualification-01/`
  (`q4km-requests`, `q4km-live`).
- GPU use: 4.3 minutes under the project lease (12:13–12:18 on 2026-11-13).

## Artifact disposition

Raw outputs (server logs, per-request rows, two Nsight reports and their SQLite exports,
about 50 MB) remain in this worktree's ignored
`artifacts/benchmarks/prefill-overhead-01/`. They were not copied into the main
checkout. The compact values here were extracted and checked against them, so they are
disposable. The traces can be regenerated by rerunning the probe (about 5 GPU minutes).
No input was modified and nothing in the main checkout was touched or deleted.

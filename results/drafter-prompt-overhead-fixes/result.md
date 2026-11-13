# Two exact runtime fixes remove most of the drafter prompt overhead

Attaching a drafter roughly doubled prompt processing and cancelled the drafters'
decoding gain on live tool tasks ([cause](../speculative-prefill-overhead/result.md)).
Runtime-v6 applies two fixes to the pinned runtime-v5 source. Both keep every output
identical to runtime-v5. They remove 64–91% of the extra prompt time that a drafter
adds, depending on the workload. On the 12 live DEV tasks Baseline's model time drops 17%,
and Baseline now beats target-only by 16%.

1. **Target features stay on the GPU.** After each target batch, the runtime copies the
   eight layer inputs the drafter needs into a persistent 42 MB GPU buffer, GPU to GPU.
   The drafter's graph reads that buffer directly and reorders it into the layout it
   used before with one exact copy. Nothing passes through host memory. A GPU event
   stops the next target batch from overwriting the buffer before the drafter has read
   it, so the host never waits.
2. **Checkpoint copies go into warm host memory.** The drafter's sliding-window cache stays
   in the context checkpoints byte for byte. Checkpoint buffers are reused instead of
   freshly allocated, and a background thread pre-faults a reserve of four blocks. Fresh
   memory, not the copy, was the slow part: on this WSL2 host, writing 130 MiB into new
   memory took about 43 ms (the kernel zero-fills each page), and into reused memory about
   8 ms. This fix also speeds up target-only, which takes checkpoints too.

## Outcome

### Outputs are identical

For each comparison below, every item named matched runtime-v5 exactly:

| Check | Target-only | Baseline | HE4 |
| --- | --- | --- | --- |
| Synthetic prompts, 512–16,000 tokens, 96-token answers and 12 cached calls | 17 / 17 rows | 17 / 17 | 17 / 17 |
| DEV screen, 60 requests + warmup: text, token IDs, stop, prompt and output counts, proposed/accepted | 61 / 61 | 61 / 61 | 61 / 61 |
| Live DEV tasks: 121 model calls (token IDs, text, cached/evaluated counts, proposed/accepted) | 121 / 121 | 121 / 121 | 121 / 121 |
| Live DEV tasks: final workspace hash, turns, termination, totals | 12 / 12 | 12 / 12 | 12 / 12 |

The probe totals were 411 proposed / 344 accepted for Baseline and 660 / 220 for HE4 on
both runtimes. On the screen they were 5,446 / 3,794 and 6,755 / 3,134. Live totals were
8,521 / 6,756 and 10,263 / 5,880.

### Prompt time

Server `prompt_ms` on synthetic token prompts, median of three repeats; one server per
cell, all cells run in one lease session. "Host features" is runtime-v6 with the GPU
handoff disabled (`LLAMA_SPEC_HOST_FEATURES=1`), so it isolates the checkpoint fix.

| Prompt tokens | Target v5 | Target v6 | Baseline v5 | Baseline v6 | Baseline v6, host features | HE4 v5 | HE4 v6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 61.5 | 64.6 | 94.2 | 64.6 | 93.0 | 93.1 | 65.5 |
| 2,048 | 193.8 | 194.5 | 369.0 | 212.7 | 306.2 | 364.9 | 211.6 |
| 4,096 | 362.3 | 363.3 | 688.1 | 406.0 | 623.4 | 678.5 | 409.0 |
| 8,192 | 712.1 | 713.1 | 1,286.4 | 779.5 | 1,186.6 | 1,256.5 | 778.5 |
| 16,000 | 1,439.0 | 1,430.3 | 2,465.7 | 1,549.5 | 2,381.2 | 2,436.0 | 1,552.7 |
| Slope, ms/token | 0.090 | 0.088 | 0.151 | 0.096 | 0.147 | 0.150 | 0.096 |

Cached calls, as in live tasks: a context call, then four calls that each append the
previous 8 generated tokens plus new tokens. Cells are the median of calls 2–4.

| Cached + new tokens | Target v5 | Target v6 | Baseline v5 | Baseline v6 | Baseline v6, host features | HE4 v5 | HE4 v6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 + 32 | 65.1 | 46.1 | 210.1 | 54.6 | 56.2 | 218.9 | 54.5 |
| 4,096 + 512 | 87.7 | 74.7 | 184.2 | 90.0 | 114.6 | 197.7 | 90.1 |
| 12,000 + 32 | 62.3 | 46.4 | 137.5 | 59.9 | 65.9 | 157.6 | 62.2 |

At 16,000 tokens Baseline's extra prompt time over target-only fell from 1,027 ms to
119 ms. The checkpoint fix alone saved 85 ms, and the GPU handoff another 832 ms. The
investigation projected 80–90 ms and 600–850 ms. The 4,096 + 32 call fell from 210 ms to
55 ms; target-only on v6 takes 46 ms. About 0.008 ms per prompt token of drafter cost
remains, roughly four times the drafter kernel time the investigation measured (29 ms per
16,000 tokens). The rest is the reorder copy and per-batch synchronization. It was not
profiled.

### DEV requests and live tasks

Lean DEV screen (60 requests, 128-token caps), server seconds summed over requests:

| Arm | Prompt v5 | Prompt v6 | Prediction v5 | Prediction v6 |
| --- | ---: | ---: | ---: | ---: |
| Target-only | 10.5 | 9.8 | 36.0 | 35.3 |
| Baseline | 19.3 | 10.6 | 25.6 | 24.9 |
| HE4 | 17.0 | 10.6 | 30.3 | 29.8 |

Lean live panel: 12 DEV tool tasks, one round, warmups excluded, runtime-v5 run
immediately before runtime-v6:

| Arm | Model s v5 | Model s v6 | Prompt s v5 | Prompt s v6 | Prediction s v5 | Prediction s v6 | Completed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Target-only | 89.5 | 83.8 | 14.1 | 10.6 | 66.5 | 65.7 | 8 / 12 both |
| Baseline | 84.6 | 70.1 | 28.9 | 14.3 | 42.8 | 42.9 | 8 / 12 both |
| HE4 | 91.4 | 83.9 | 28.5 | 15.8 | 50.1 | 50.2 | 8 / 12 both |

The drafters' extra live prompt time over target-only fell from 14.8 to 3.7 s (Baseline)
and from 14.4 to 5.2 s (HE4). On the screen it fell from 8.8 to 0.8 s and from 6.5 to
0.8 s. The investigation projected Baseline's live model time to fall 16% (259 to about 217 s
over three repeats of the full panel, target-only unchanged at 257 s). The measured fall
is 17%. Target-only also fell 6%, because its own checkpoints got cheaper. On runtime-v6
Baseline is 16% faster than target-only on these tasks, and HE4 is level with
target-only (83.9 vs 83.8 s), where it was 2% slower on v5. One round on the lean panel
does not give an interval. The arms ran in sequence, not interleaved.

### Measured along the way

- **Blocking the host after each drafter injection cost decode time.** The first v6 build
  synchronized the drafter after every feature injection. That was exact, but per-token
  prediction time rose by a median 1.7% for Baseline and 4.4% for HE4 on the 60 screen
  requests, since each verification round stalled the host. Replacing it with a GPU event
  brought HE4 to 0.992 of v5 in a replay of the same requests. The final build's screen
  and live prediction times above match v5.
- **Pinned host memory lost.** Copying 80 MiB into pinned memory took 2.3 ms against
  5 ms into reused pageable memory. But allocating it took 40 ms and freeing it 17 ms,
  more than a fresh pageable allocation costs. Live checkpoints accumulate (up to 32 per
  slot), so most would need a new block. Transparent huge pages made fresh memory slower
  on this kernel: 55–61 ms against 43 ms for 130 MiB.
- **The investigation's other checkpoint options were not built.** A full-size drafter
  window cache keeps cache cells in different slots once the context exceeds the window.
  Attention then sums in a different order, so drafts could change after near ties. That
  is inferred from the cache layout, not measured. Skipping the drafter's cells changes
  acceptance after restores, as the investigation said. Keeping the bytes and making the
  copy cheap is exact and needs no extra GPU memory.

### Costs

- GPU memory: a 42 MB feature buffer in the target context (8 layers × 512 rows × 2,560
  FP32). The drafter's reordered copy uses the space its uploaded input used before.
- Host memory: up to four pre-faulted blocks (about 0.4 GB with a drafter) stay resident.
  Freed checkpoint blocks are kept for reuse up to 2 GiB of idle memory. Peak use does not
  exceed the checkpoints alive at the peak plus that reserve. One background thread
  zero-fills reserve blocks while the GPU works.
- The GPU path is used only for DFlash drafters whose graph supports it (plain DFlash
  layout, ascending target layers, not LiliCorr). Others keep the host path.

## Scope and limitations

- One RTX 5080 under WSL2, Q4KM target, depth 2, F16 KV, 32,768 context, benchmark server
  arguments. NVFP4 and other hosts were not run. The reuse gain depends on how slowly this
  kernel faults fresh pages; native Linux may gain less from the checkpoint fix.
- Exactness was checked on 17 synthetic rows, 61 DEV screen requests and 12 DEV live tasks
  per arm. The 120-request panel, harness panel and long-input DEV requests were not run.
  The synthetic rows cover prompts up to 16,000 tokens. `verify-greedy` was not rerun;
  each comparison already checked greedy outputs against the target arm, and every v6
  output matched v5 exactly.
- Timing comes from one run per cell. The probe paired v5 and v6 within one session; the
  live and screen arms ran v5 then v6 within about 25 minutes. No intervals are given.
- The drafter's checkpoint copy remains about 5 ms per checkpoint once the reserve is warm.
  A cold reserve right after a burst of checkpoints falls back to fresh memory.
- The RAM prompt-cache save, `GGML_CUDA_DECODE_CONSISTENT` fusion, FP16 features and depth
  are unchanged.

## Decision

Retain runtime-v6 as a qualified candidate runtime; adopting it is proposed, not done.
The benchmark default stays runtime-v5 and no saved reference was re-recorded. Adopting it
would need:

1. `native-build` with `artifacts/native/runtime-v6-build-spec.json` in the main checkout
   after merging. Build results bind absolute paths, so the worktree build cannot be moved.
   Then a short probe or screen to confirm the rebuilt binary is still exact.
2. New saved references for the runtime-v6 protocol: run the `prepare-checks` establish
   specs (screen, requests, harness, live) for each target precision used, then
   `record-reference`. Runtime-v5 references cannot serve v6 candidates.
3. Pointing current comparison and campaign specifications at runtime-v6, and stating in
   results which runtime produced each timing. Historical v5 timings stay valid for v5.

## Evidence and reproduction

- [result.json](result.json) holds every number above: per-repeat prompt times, cached-call
  values, identity counts, panel summaries, projections and runtime/model identities.
- [executed-probes.txt](executed-probes.txt) holds the timing probe and the screen replay;
  [executed-analysis.txt](executed-analysis.txt) holds the panel runner, the identity
  checks and the extractor.
- Runtime-v6: `native-build` of `artifacts/native/runtime-v6-build-spec.json`. That is the
  runtime-v1 bundle (sha256 `696db0c1…`) plus the sampled-EOG repair and the edit sets
  `checkpoint_warm_buffers_v1` and `device_feature_handoff_v1` in
  [`benchmarks/runtime_edits.py`](../../benchmarks/runtime_edits.py), applied by exact
  match. The build froze code `84724b9a…`; `llama-server` sha256 `1938a8b0…`.
- Runtime-v5 `llama-server` `c83a7965…`; target `38742092…`; Baseline `83586a63…`; HE4
  export `99d2be79…`.
- Specifications: the lean Q4KM `screen-establish.json` and `live-establish.json`
  unchanged for v5. For v6, the same files with only `source`, `binary` and `libraries`
  replaced.
- GPU use: about 32 minutes under the project lease between 14:19 and 15:24 on
  2026-11-13. This includes copy benchmarks (2 s), dev smoke runs (5 min), a screen and
  probe on the intermediate build (6.5 min), the HE4 replay (2 min), final probes
  (3.5 min), screens (5.5 min) and live runs (10 min).

## Artifact disposition

In this worktree's ignored `artifacts/`:

- **Retained:** `native/runtime-v6/` (504 MB, the measured build) and its build spec, until
  adoption is decided. Also the final paired outputs in `benchmarks/prefill-fix/`:
  `probe-02`, `screen-v5`, `screen-v6final`, `live-v5` and `live-v6final` (about 45 MB).
  They are complete authenticated v6 panels, which a new v6 reference could use.
- **Retired** with the retirement command (receipts under `artifacts/receipts/`): the dev
  source/build copy `native/dev-prefill/` and the intermediate host-sync build
  `native/runtime-v6-hostsync/`. Their outputs (`dev-smoke-01..03`, `replay-01`,
  `probe-01`, `screen-v6`) were retired too. The numbers above that come from them are in
  result.json. The source state of each is reproducible from Git history of
  `benchmarks/runtime_edits.py`. The tool removes only listed regular files. Afterwards
  the leftover library symlinks, empty directories and the scratch Git repository of the
  dev copy were deleted. That repository held only the unmodified pinned source as its
  base commit.
- The CUDA kernel cache in `cache/cuda` (256 MB) is disposable.
- Read-only links from this worktree's `artifacts/` into the main checkout (models,
  prompt sources, lean checks, project workflows, original export, runtime-v5, the source
  bundle) were removed after use. No input or main-checkout file was modified.

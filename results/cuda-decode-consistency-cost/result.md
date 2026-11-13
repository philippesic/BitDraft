# Exact verification costs about 5% of decoding speed and caps useful draft depth at three

Every benchmark sets `GGML_CUDA_DECODE_CONSISTENT=32`, which makes a speculative run's
verification batch use the same arithmetic as decoding one token, so speculative greedy
output equals target-only greedy output. This investigation measured what that guarantee
costs on the Q4KM target, what goes wrong without it, and where speed could be recovered
without giving it up. Exactness itself held in every measured run: all 20 runs with the
setting matched target-only exactly, at every draft depth from 2 to 15.

## Outcome

**Cost today.** The setting costs every arm about 4–6% of decoding speed. On the screen it
cost target-only 7% of complete-request throughput. Baseline, which pays extra for its own
prompt processing, lost 4%. HE4 ran 10% *faster* with the setting, because it had stalls
of up to 1.8 s per request without it (see below). The cost grows with context: target-only
decoding at 8,192–16,384 input tokens is 19% slower with the setting, on both panels.

| 60-request screen, depth 2, mean of 2 runs | Setting on (tok/s) | Setting off (tok/s) | Change from setting | Decode on | Decode off |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 127.5 | 137.2 | −7.1% | 182.2 | 194.5 |
| Baseline | 132.7 | 137.9 | −3.8% | 256.5 | 274.0 |
| HE4 | 120.8 | 109.7 | +10.1% | 214.4 | 223.2 |

Complete-request tok/s is emitted tokens over total HTTP request time. Decode tok/s uses
the server's own token-prediction time and excludes prompt processing. The two repetitions
ran in reversed arm order. Complete-request rates varied by up to 5% between repetitions
(Baseline with the setting: 136.0 and 129.4); decode rates by at most 3%.

| 120-request panel, one run each | Complete tok/s | Decode tok/s | Accepted per round | ms per round | × target-only (on) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only, setting on | 167.1 | 188.2 | — | — | 1.00 |
| Target-only, setting off | 175.6 | 197.0 | — | — | 1.05 |
| Baseline, depth 2, on (maintained) | 204.3 | 260.1 | 1.28 | 8.75 | 1.22 |
| Baseline, depth 3, on | 217.7 | 281.7 | 1.60 | 9.22 | 1.30 |
| Baseline, depth 4, off | 224.0 | 289.9 | 1.78 | 9.60 | 1.34 |

**The setting caps depth at three, not two.** Each round verifies the depth plus one
token. With the setting, the target's Q4_K/Q5_K/Q6_K matrix products process four tokens
per pass over the weights, so a round costs about the same up to depth 3 (four tokens) and
jumps 28% at depth 4. Without the setting, rounds grow smoothly. The sweep below is on the
60-request screen, one run per cell; depth 2 averages two runs.

| Depth | Baseline on: tok/s | decode | ms/round | Baseline off: tok/s | decode | ms/round | Baseline accepted/round | HE4 on: decode | ms/round | HE4 off: decode | ms/round | HE4 accepted/round |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 132.7 | 256.5 | 9.28 | 137.9 | 274.0 | 8.67 | 1.38 | 214.4 | 8.95 | 223.2 | 8.60 | 0.92 |
| 3 | 132.5 | 282.2 | 9.90 | 135.1 | 273.9 | 10.07 | 1.78 | 217.1 | 9.55 | 224.0 | 9.27 | 1.08 |
| 4 | 122.9 | 237.4 | 12.69 | 148.2 | 295.4 | 10.14 | 2.00 | 176.5 | 12.32 | 220.4 | 9.82 | 1.17 |
| 6 | 122.0 | 230.2 | 14.39 | 138.9 | 291.4 | 11.17 | 2.28 | 163.9 | 13.78 | 215.4 | 10.45 | 1.26 |
| 8 | 119.6 | 206.8 | 16.76 | 146.2 | 287.0 | 11.80 | 2.43 | 141.5 | 16.23 | 198.5 | 11.43 | 1.28 |
| 15 | 101.0 | 159.2 | 22.65 | 136.9 | 257.9 | 13.78 | 2.58 | 104.2 | 22.22 | 170.9 | 13.41 | 1.30 |

Accepted per round differs by up to 0.08 between modes and is shown as their mean. With
the setting, Baseline decodes fastest at depth 3, and depth 3 gained 6.6% complete-request
throughput over depth 2 on the 120-request panel with exact output. Without the setting,
the best depth is 4–8. HE4 gains little from depth in either mode: its drafts are rarely
accepted beyond the second position.

The drafter's own forward is a small, depth-dependent part of the round. Its traced drafting
time per round, with versus without the setting: 1.88 vs 1.87 ms at depth 2, 1.97 vs 1.99 at
depth 3, 2.33 vs 2.05 at depth 4, 3.37 vs 2.39 at depth 8 and 5.22 vs 2.81 at depth 15
(Baseline). Nearly all of the depth-4 jump is the target's verification.

**What goes wrong without it.** Near-tie flips, never a wrongly accepted draft.

| Without the setting (60-request screen) | Requests differing from target-only (setting off) | First differences per 1,000 tokens | Reference logit margin at first difference, median / max |
| --- | ---: | ---: | --- |
| Target-only (vs target-only with the setting) | 28 / 60 | 5.9 | 0.048 / 0.216 |
| Baseline or HE4, depth 2 | 16 / 15 of 60 | 2.8 / 2.6 | 0.033 / 0.114; 0.028 / 0.114 |
| Baseline, depth 4 | 22 / 60 | 4.0 | 0.022 / 0.056 |
| Baseline, depth 8 | 32 / 60 | 8.0 | 0.062 / 0.397 |
| 120-request panel: target-only, setting off vs on | 74 / 120 | 5.0 | not measured |
| 120-request panel: Baseline depth 4 vs target-only, both off | 66 / 120 | 3.8 | not measured |

- In every one of these first differences, the speculative or normal-dispatch run chose
  the reference target's second-ranked token. The gap between the top two log-probabilities
  (equal to their logit difference) never exceeded 0.40. Only 1.6% of all generated positions
  have a margin below 0.1; the median margin is 5.9.
- The diagnostic verifier-trace runtime checked every round independently for Baseline and
  HE4 at depths 2 and 8 without the setting: 11,055 rounds, 49,971 proposals. Every
  accepted prefix and correction token equalled the target's own argmax over that same
  verification batch. No accepted draft disagreed with the target's choice.
- Without the setting, runs are deterministic: repeated runs and the trace build reproduce
  identical tokens. Output depends mainly on the verification width; Baseline and HE4 at
  the same depth differ in only 1–2 of 60 requests.
- The target-only greedy output itself depends on the setting, because the setting also
  disables kernel fusion and forces single-partition attention for one-token decoding.
  Target-only with and without it differ in 28 of 60 screen requests and 74 of 120 panel
  requests, more often than speculative and target-only runs differ without it.

So the divergences are a tolerance question: a different greedy continuation after a
near-tie, caused by different floating-point summation order. They are not a verifier bug.

**Acceptance beyond position 2.** Share of rounds whose first *i* drafts were all
accepted, from depth-15 runs with the setting (3-decimal runtime rates; depth 8 agrees
within about 0.02):

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.732 | 0.484 | 0.321 | 0.223 | 0.174 | 0.135 | 0.107 | 0.088 | 0.066 | 0.050 | 0.029 |
| HE4 | 0.526 | 0.289 | 0.163 | 0.112 | 0.078 | 0.051 | 0.037 | 0.025 | 0.012 | 0.004 | 0.000 |

Baseline's acceptance at each later position, given that all earlier drafts were accepted,
rises from 0.66 to about 0.86. Its expected accepted drafts per round are 1.38 at depth 2
and 2.58 at depth 15. HE4's tail falls quickly, so depth cannot rescue it.

## Ways to recover speed without giving up exactness

Ranked by expected gain and risk. Gains are estimates from the measurements above; none
of these changes was implemented or measured.

1. **Draft at depth 3 instead of 2 (Baseline).** Measured: +6.6% complete-request and
   +8.3% decode throughput on the 120-request panel, exact output, no code change. It is a
   benchmark default change, so it alters the saved-reference protocol (depth is bound into
   references). It does not change the correctness contract. HE4 gains about 1% in decoding.
2. **Exact attention that keeps split-KV parallelism, plus batch-invariant fusions.** The
   setting forces one attention partition for every batch of up to 32 tokens, including
   one-token decoding. That likely explains target-only's 19% decode loss at 8K–16K inputs
   and part of its 2–5% loss at short inputs. This attribution is inferred from the
   per-length pattern, not isolated. Partitioning keys at fixed positions, independent of
   batch width and padded length, keeps single-token and batched results identical. It
   could recover most of the 4–6% average and 19% long-context loss for every arm,
   target-only included. Exactness is preserved, but target-only arithmetic changes, so
   every saved reference must be re-established. This needs native kernel work and a new
   runtime build.
3. **Share weight reads across more than four columns.** The SM120 group kernel works
   because the four-warp MMVQ reduction is identical for one to four columns; five or more
   switch to two warps. A variant that keeps four warps for 5–16 columns would compute each
   column exactly as one-token decoding does. If its rounds then grew like normal
   dispatch's, Baseline with the setting would decode about 300 tok/s at depths 4–8, versus
   282 at depth 3: about 3–7%. It needs the same native qualification at widths 5–16. The
   contract is unchanged. HE4 gains nothing.
4. **Exempt the drafter's forward.** Its outputs are only proposals, so its arithmetic
   need not match anything, and target exactness is unaffected. The measured drafter-time
   difference is zero at depths 2–3, 0.3 ms per round at depth 4 and 1.0 ms at depth 8.
   This is only worth doing together with option 3. It needs a per-context switch, because
   the setting is process-wide today.
5. **Choose depth per round from drafter confidence.** With the setting, rounds of up to
   three drafts cost almost the same, so truncating within that range saves little.
   Without option 3, confident rounds cannot go deeper cheaply. The expected gain is small
   (inferred, unmeasured); exactness is unaffected.
6. **Accept normal dispatch with a measured divergence rate.** Measured: Baseline at depth
   4 without the setting ran 224.0 tok/s on the panel, against 217.7 for the exact depth-3
   option (+2.9%) and 204.3 for today's default (+9.6%). The cost: 55% of panel requests
   (3.8 per 1,000 tokens) no longer match target-only greedy output. That breaks the rule
   in [domain](../../docs/domain.md) and [benchmarks](../../docs/benchmarks.md) that
   "Greedy arms must match target token IDs, text and stopping", which is the cheap check
   that catches real verifier bugs. The replacement guarantee would be the trace oracle
   (accepted prefix equals the batched target argmax), which needs the slower diagnostic
   build. References would be per depth, and slightly per drafter. After options 1–3,
   little speed remains to justify this.

## Scope and limitations

- One RTX 5080, runtime v5 (and its identical-GPU-source trace build), Qwen3.5-4B Q4KM
  target, F16 KV, context 32,768, greedy decoding, concurrency one. NVFP4 and other
  hardware were not measured.
- DEV prompts only: the 60-request screen (128-token caps) and the 120-request panel. Both
  are reused DEV cohorts, not traffic.
- Most depth-sweep cells and every panel row are single runs. Complete-request rates
  varied up to 5% between repeated runs, mostly in prompt time. Close differences, such as
  depth 2 versus 3 on the screen's complete-request clock, are not resolved.
  The panel's depth-3 gain is one run per arm.
- HE4's stalls without the setting add 4–5 s per 60 requests outside the server's prompt
  and decode timers. Their cause is unexplained; decode-only, HE4 is 4% faster without
  the setting.
- Margins were measured on the screen only, from the target's top-2 log-probabilities;
  requesting them did not change any token. Panel divergence margins were not measured.
- Per-position rates carry the runtime's 3-decimal rounding (about ±1.5 rounds in ~3,000).
- Options 2–5 are design estimates from these measurements and source reading of
  `ggml-cuda.cu`, `fattn.cu`, `fattn-common.cuh` and `mmvq.cu`. None was built or tested.

## Decision

Unresolved; proposed, not decided:

- Keep the exact contract and the setting.
- Consider depth 3 for Baseline-style drafters, which needs new saved controls.
- Pursue fixed-position attention partitioning (option 2), then four-warp wide groups
  (option 3), as runtime work with native exactness qualification.

Normal dispatch with a tolerance is not recommended: its measured gain over exact depth 3
is 3%, and it would replace an exact check with a statistical one. No default,
saved reference or runtime was changed.

## Evidence and reproduction

[result.json](result.json) holds every run's totals, length-band rates, drafter statistics,
per-position acceptance, divergence positions with margins, trace verdicts, the executed
run plans, input identities and script hashes. The depth sweep is in `screen_runs`
(fields `arm`, `mode`, `depth`). Scripts:
[harness](executed-harness.txt), [analysis](executed-analysis.txt) and
[result builder](executed-result-builder.txt).

The harness reuses `benchmarks.runtime_policy.arguments`/`payload`, `benchmarks.server_child`,
`benchmarks.measurement.request`, `verify_generation`/`valid_sample` and the project
`device_lease`. It changes only `GGML_CUDA_DECODE_CONSISTENT` (32 or 0) and records
mismatches instead of refusing them. The analysis uses `benchmarks.verifier.reduce_verifier_trace`
for the trace oracle. Run with the small check environment from a checkout containing those modules:
`PYTHONPATH=. uv run --locked --no-sync --offline python -B harness.py PLAN OUTPUT`, then
`analyze.py OUTPUT` and the builder. The `target-c-r0` arm reproduced the saved Q4KM screen
target reference token-for-token (60/60) at 128.4 vs 129.0 tok/s.

GPU use was 59 minutes of arm time, executed 2026-11-13 under the project lease. Batches:
A, depth 2 in two reversed-order repetitions; B, the depth sweep; C, diagnostic
probability and trace runs, which are excluded from timing; D/E, the panel. Inputs
(read-only, main checkout): `artifacts/models/qwen35-4b-q4km.gguf`,
`artifacts/models/dflash-4b-q4km.gguf`,
`artifacts/qualification/original-zero-export-02/exports/99d2be79….bin` (HE4 220,095),
`artifacts/native/runtime-v5`, `artifacts/native/verifier-trace-runtime-01`, and the
lean-checks Q4KM workloads. All hashes are in result.json and matched the saved
specifications.

## Artifact disposition

- **Raw outputs, retained for now.** The raw run outputs (13 MB: per-request tokens,
  probabilities, server logs) are in the executing worktree's ignored
  `artifacts/consistency/`, not in the main checkout. They are disposable after review:
  result.json holds every value reported here, and rerunning the plans reproduces them.
  Exact raw server logs and per-token probabilities are lost if that worktree is removed.
- **Read-only inputs.** No models, saved references or runtimes were created, changed or
  deleted.

# Protocol v3: Baseline runs 1.34–1.72× target-only, S4 trails it by 7–12% on requests and ties it on live tasks

The shared benchmark protocol changed in one step, so saved controls were re-recorded
once. Request policy v3 serves with runtime-v6 (the exact prompt-overhead fixes), drafts
at depth 3 with exact verification, and turns off the server's unused RAM prompt cache
(`--cache-ram 0`). Depth 2 had flattered HE4 against Baseline by about 8%
([depth fairness](../draft-depth-fairness/result.md)). Runtime-v5 had doubled drafter
prompt time ([fixes](../drafter-prompt-overhead-fixes/result.md)). The RAM cache added
about 90 ms to every independent request in every arm
([prefill overhead](../speculative-prefill-overhead/result.md)).

Every output that should match runtime-v5 did. The rebuilt runtime reproduced target-only
token IDs, text and stopping on all 198 requests and 12 live tasks of both precisions, and
reproduced runtime-v5's depth-3 accepted/proposed counts for Baseline, S4 and HE4 220,095
request by request. New target and Baseline references exist for Q4KM and NVFP4 on all
four routine cohorts, and S4 is the prior leader on the three request cohorts.
One exactness gap surfaced: on live tasks S4's trajectory left target-only's on one of 12
Q4KM tasks and two of 12 NVFP4 tasks. The cause lies in cached-prefix reuse, not in
verification, and S4 has no live reference.

## Outcome

### Standing under protocol v3

Complete-request tokens/s over summed HTTP request time; speed relative to target-only in
brackets, then acceptance and accepted drafts per round. Live cells are model seconds
summed over the 12 tasks (one round, warmups excluded) and completed tasks. Every request
arm ran once.

| Q4KM | Target-only | Baseline | S4 | HE4 220,095 |
| --- | ---: | ---: | ---: | ---: |
| Screen, 60 requests | 148.1 | 202.9 (1.37×) 60.6% / 1.80 | 187.9 (1.27×) 49.1% / 1.45 | |
| Panel, 120 requests | 174.7 | 250.1 (1.43×) 53.3% / 1.60 | 222.6 (1.27×) 41.4% / 1.24 | 197.5 (1.13×) 31.3% / 0.94 |
| Prefix harness, 18 requests | 174.5 | 296.4 (1.70×) 74.1% / 2.22 | 274.0 (1.57×) 62.8% / 1.89 | |
| Live, 12 tasks | 75.9 s, 8 done | 51.4 s (1.48× faster), 8 done, 70.1% | 49.8 s, 9 done, 72.8%; diverged on 1 task | |
| Live, 11 agreeing tasks | 71.8 s | 47.6 s, 72.1% | 47.3 s, 73.1% | |

| NVFP4 | Target-only | Baseline | S4 |
| --- | ---: | ---: | ---: |
| Screen, 60 requests | 161.6 | 219.7 (1.36×) 58.2% / 1.72 | 202.5 (1.25×) 47.4% / 1.40 |
| Panel, 120 requests | 184.4 | 263.0 (1.43×) 53.0% / 1.59 | 231.6 (1.26×) 40.5% / 1.21 |
| Prefix harness, 18 requests | 185.4 | 319.0 (1.72×) 75.5% / 2.27 | 292.0 (1.58×) 63.3% / 1.90 |
| Live, 12 tasks | 48.7 s, 8 done | 36.3 s (1.34× faster), 8 done, 59.7% | 35.9 s, 8 done, 58.1%; diverged on 2 tasks |
| Live, 10 agreeing tasks | 41.1 s | 30.0 s, 60.7% | 30.5 s, 57.1% | |

S4 against Baseline, paired source-group bootstrap within the same run (2,000 draws):

| Cohort | Q4KM acceptance points | Q4KM speed ratio | NVFP4 acceptance points | NVFP4 speed ratio |
| --- | --- | --- | --- | --- |
| Screen | −11.6 (−13.4, −9.7) | 0.926 (0.910, 0.940) | −10.8 (−12.4, −9.2) | 0.922 (0.908, 0.935) |
| Panel | −12.0 (−12.6, −11.2) | 0.890 (0.882, 0.899) | −12.5 (−13.3, −11.8) | 0.881 (0.873, 0.890) |
| Prefix harness | −11.3 (−13.9, −7.5) | 0.925 (0.898, 0.957) | −12.2 (−16.7, −5.6) | 0.916 (0.880, 0.964) |

On the Q4KM panel the 220,095 export accepts 10.1 points less than S4 (−11.0, −9.2); its
speed ratio to S4 is 0.887, descriptive because the two ran in separate executions.

- **Baseline now clearly beats target-only everywhere,** including the live tool tasks,
  where under runtime-v5 at depth 2 no drafter measurably helped (F015).
- **S4 trails Baseline by 7–12% on request throughput** on both precisions, with an
  acceptance gap of 11–13 points.
- **On live tasks S4 ties Baseline.** On the tasks where its trajectory matched, S4 took
  0.8% less model time than Baseline on Q4KM and 1.7% more on NVFP4. This is one round,
  arms in sequence, with no interval. Live calls repeat much file and tool text, which suits
  short drafts.

### Exactness

| Check | Result |
| --- | --- |
| Target-only signatures (token IDs, text, stop) against the runtime-v5 target references: screen, panel, harness | Q4KM 60/60, 120/120, 18/18; NVFP4 60/60, 120/120, 18/18 |
| Live target-only final workspace against the runtime-v5 references | Q4KM 12/12, NVFP4 12/12 |
| Live target-only per call, tokens and cached/evaluated counts | Q4KM 121/121 against runtime-v6 with the RAM cache and against two runtime-v5 runs; NVFP4 97/97 against runtime-v5 |
| Depth-3 accepted, proposed and emitted per request against runtime-v5 | Baseline screen 60/60 and panel 120/120, S4 screen 60/60, HE4 220,095 panel 120/120 |
| Drafter request arms against target-only | All identical, both precisions |

**The live divergence.** When a call ends with the end-of-turn token (248046) and that
token was an accepted draft, the drafter arm's slot cache holds it; target-only never decodes
its final token. The next call then reuses one more cached token (1,662 against 1,661) and
splits its new prompt into batches at a different offset. Prompt batches over 32 tokens use
fused arithmetic, so the logits move slightly and a near tie can flip. Every S4 divergence
followed exactly this pattern, at generated token 27 (Q4KM), 14 and 62 (NVFP4) of the
following call. A third longer prefix (Q4KM, `project-percentile-interval`) did not flip.
Baseline's cached prefix never differed (121 and 97 calls), nor did any arm's in the five
earlier depth-2 live rounds checked. The Q4KM divergence repeated exactly in a second run.
The mechanism is inferred from the cached counts; the server's draft rounds were not traced.

**The RAM prompt cache.** Live sessions do not use it: they reuse the slot's own cache, and
with `--cache-ram 0` every live call matched the earlier runs, including cached and evaluated
counts. For independent requests the time outside the server's prompt and prediction timers
fell from 87.5 to 1.1 ms per request for target-only on the screen (runtime-v6 with and
without the cache). On the panel it fell from 11.6–14.3 s per arm in the runtime-v5
depth-fairness runs to 0.2 s.

### Not comparable with earlier ratios

Depth, runtime and request overhead all changed. On the Q4KM panel Baseline went from 1.20×
target-only (runtime-v5, depth 2, mean of two runs) to 1.43×, and the 220,095 export from 0.87 of Baseline to
0.79. Ratios and acceptance from depth-2 or runtime-v5 panels, including F014, F015 and the
depth-2 screens of F016 and F022, describe their own protocol only.

## Scope and limitations

- One RTX 5080 under WSL2, Qwen3.5-4B Q4KM and NVFP4 targets, F16 KV, 32,768 context,
  greedy, concurrency one, DEV cohorts only. S4 is trained on Q4KM teacher features; its
  NVFP4 scores use the same export.
- Each panel ran once. Arms ran in sequence within a block, so the paired speed intervals
  cover request-to-request variation, not drift between sessions (earlier repeats moved
  0.4–3.5%). Live has 12 tasks from one project and no interval.
- Live completion is not comparable on the diverged tasks: S4 finished
  `project-partial-labels` in 4 turns on Q4KM where target-only hit the 16-turn limit.
- The same cache behaviour exists in runtime-v5, so any drafter could trigger it; only S4
  did in the runs checked. The runtime does not drop an accepted final end-of-turn draft from the cache; removing it
  would need a runtime change and new controls.
- Depth 3 is fair on today's runtime; a change to verification cost needs the panel sweep
  that [depth fairness](../draft-depth-fairness/result.md) proposes.

## Decision

Decided by the owner and done: protocol v3 is the benchmark default. Runtime-v6 is built in
the main checkout, `--cache-ram 0` applies to every server including live and depth
calibration, and request policies are named `native-greedy-request-v3` and
`native-live-v3`. Comparisons refuse references from other protocols, so runtime-v5
controls cannot serve v3 runs; those records stay valid for their own protocol.

Decided here: Q4KM and NVFP4 live controls come from target+Baseline runs of the prepared
live specification, because a panel containing S4's diverged arm cannot be registered.
S4 is the prior leader on screen, panel and harness for both precisions and has no live
reference.

Proposed, not decided: a runtime edit that keeps an accepted final end-of-turn draft out
of the slot cache, or ends the cached prefix at a fixed boundary, so live trajectories stay
exact. It would need a new runtime and new live controls.

## Evidence and reproduction

[result.json](result.json) holds every arm summary, the paired comparisons, live rows,
exactness counts and divergence events, the RAM-cache measurement, GPU seconds per run,
runtime identities and the new reference names. [Executed scripts](executed-scripts.txt)
generated the inputs, ran each command from a frozen snapshot, registered references and
wrote the specifications. [The result builder](executed-result-builder.txt) regenerates
result.json:

```sh
python3 build_result.py RESULTS_DIR result.json
```

- Code: branch commit `ab3976ac`, source identity `68210f1e…`, frozen in
  `artifacts/benchmarks/references/runtime-v6-depth3/executed-sources/` of the main checkout.
  Commands ran from that snapshot so outputs landed in the main checkout's artifacts.
- Runtime-v6: `native-build` of `artifacts/native/runtime-v6-build-spec.json` (`9ccc4338…`)
  from the main checkout at `913891eb` (code `ff39cc12…`); repairs
  `rejection_eog_boundary_v1`, `checkpoint_warm_buffers_v1`, `device_feature_handoff_v1`;
  `llama-server` `190b49eb…`.
- Models: Q4KM target `38742092…`, NVFP4 target `5447bdb8…`, Baseline
  `83586a63…`, S4 export `10ade33a…`, HE4 220,095 export `99d2be79…`.
- Workloads: `prepare-checks` reproduced the existing lean workloads byte for byte (screen
  `feaeeae0…`, panel `f25c7504…`, harness `a3308c85…`, live `4c0b0b49…`).
- New references in `artifacts/benchmarks/references/` (file name prefixes):

| Cohort | Q4KM target / Baseline / S4 | NVFP4 target / Baseline / S4 |
| --- | --- | --- |
| Screen | `a0f13eeb` / `4aa9b044` / `35150f3e` | `6d792e91` / `28690888` / `fc21f5cb` |
| Panel | `7602a2e9` / `25924e17` / `b17ad391` | `18d99b01` / `e0996d9d` / `d428d75d` |
| Prefix harness | `c3b4b42b` / `824c14cc` / `d0c7cbe7` | `115f82c2` / `ecdec409` / `e3848867` |
| Live | `efa6a096` / `e60f52af` / none | `5f901b49` / `fde4e9c9` / none |

- GPU: 42.3 minutes of command wall time under the project lease, including model loads,
  between 17:37 and 18:24 on 2026-11-13 (per-run seconds in result.json). The build took
  about 7 minutes of CPU time.

## Artifact disposition

All in the main checkout's ignored `artifacts/`; nothing existing was changed or deleted.

- **Retained, current inputs:** `native/runtime-v6/` (504 MB), its build spec and log, and
  the build's frozen code under `native/executed-sources/ff39cc12…`.
- **Retained, reference dependencies:** `benchmarks/references/runtime-v6-depth3/` (103 MB):
  inputs, lean workloads and establish specifications, the 13 source runs, registrations,
  candidate-only specifications, logs and the frozen code. The 22 new registry records
  authenticate these runs, so the runs stay while their references are in use.
- **Retained, divergence evidence:** the refused three-arm `runs/q4km-live` and its three
  refused registrations, which result.json summarizes; also `q4km-live-s4` and
  `nvfp4-live-s4`.
- **Disposable:** the CUDA kernel cache growth under `cache/cuda`.
- Exact replay of every run remains possible from the retained runtime, models and specifications.

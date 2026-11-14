# Runtime-v7 keeps live drafter trajectories exact; S4 now has live references and ties Baseline

Under protocol v3 the S4 drafter's live tool-task trajectory left target-only's on 1 of 12
Q4KM tasks and 2 of 12 NVFP4 tasks ([protocol v3](../benchmark-protocol-v3/result.md)).
The cause was cache state, not verification. When a call ended on an end-of-turn token
that the target had accepted as a draft, that token stayed in the server's per-session
cache. The next call then reused one more cached token than target-only. It split its new
prompt into differently sized batches, the arithmetic moved slightly and a near tie
flipped. The registry refuses a panel with a diverged arm, so S4 had no live reference.

Runtime-v7 is runtime-v6 plus one edit to the server's speculative loop, `stop_token_cache_v1`.
When generation stops on a token from the accepted part of a round, the server now
removes that token, and anything accepted after it, from the session cache before it
answers. It does this in the single cache rollback that each round already makes, because
the target's recurrent layers allow only one rollback per round. The cache then holds
exactly what target-only leaves: everything before the stopping token. With runtime-v7, S4
matches target-only on all 12 live tasks for both precisions, call by call, including
cached and evaluated token counts.

## Outcome

### Exactness

| Check, runtime-v7 | Q4KM | NVFP4 |
| --- | --- | --- |
| S4 live calls identical to target-only (tokens, text, cached, evaluated and prompt counts) | 121 / 121, 12 / 12 tasks | 97 / 97, 12 / 12 tasks |
| S4 final workspace identical to target-only | 12 / 12 | 12 / 12 |
| Target-only and Baseline live calls identical to runtime-v6 | 121 / 121 each | 97 / 97 each |
| Target-only, Baseline, S4 request outputs (token IDs, text, stop) identical to runtime-v6: screen, panel, harness incl. warmups | 61, 123, 21 each | 61, 123, 21 each |
| Same responses: proposed, accepted, evaluated, cached-prefix and predicted counts identical to runtime-v6 | all | all |
| Drafter arms' outputs identical to target-only | all | all |

Under runtime-v6 the same S4 export matched target-only on 10 of 12 tasks for each
precision when cached counts are included. One of the Q4KM differences
(`project-percentile-interval`) shifted the cache without flipping a token, as the protocol
result recorded.

One response field changes. `tokens_cached` reports the session cache size after a request.
Runtime-v6 reported one more than target-only for 54 of 820 drafter request responses
(Baseline 18, S4 36): those ended on an accepted end-of-turn draft. Baseline did this on
requests too, but never at the end of a live call. Runtime-v7 reports the same value as
target-only for every response. No reference or comparison reads this field.

### Live standing with S4

Model seconds summed over the 12 tasks, one round, warmups excluded; completed tasks;
acceptance. All arms ran in one panel per precision.

| Arm | Q4KM | NVFP4 |
| --- | --- | --- |
| Target-only | 78.1 s, 8 done | 51.6 s, 8 done |
| Baseline | 55.4 s (1.41× faster), 8 done, 70.1% | 38.9 s (1.33×), 8 done, 59.7% |
| S4 | 55.0 s (1.42×), 8 done, 72.0% | 38.2 s (1.35×), 8 done, 57.7% |

S4 took 0.7% (Q4KM) and 1.8% (NVFP4) less model time than Baseline. There are 12 tasks
from one project and no interval, so this is a tie. Under runtime-v6, S4 finished one
diverged Q4KM task in 4 turns that target-only does not finish. With exact trajectories,
every arm completes the same 8 tasks.

### Requests are unchanged

Outputs and draft counts are identical to runtime-v6, so acceptance is too. Paired S4 against
Baseline speed ratios, source-group bootstrap within each panel (2,000 draws):

| Cohort | Q4KM speed ratio | NVFP4 speed ratio | Acceptance points, Q4KM / NVFP4 |
| --- | --- | --- | --- |
| Screen, 60 | 0.945 (0.928, 0.961) | 0.925 (0.907, 0.941) | −11.6 / −10.8 |
| Panel, 120 | 0.887 (0.877, 0.898) | 0.888 (0.880, 0.898) | −12.0 / −12.5 |
| Prefix harness, 18 | 0.920 (0.895, 0.954) | 0.918 (0.883, 0.969) | −11.3 / −12.2 |

Runtime-v6 gave 0.926, 0.890, 0.925 (Q4KM) and 0.922, 0.881, 0.916 (NVFP4).

### Speed cost: none measured, but the saved live controls ran on a busy host

The runtime-v7 live panels took 3–8% longer than runtime-v6's for every arm, including
target-only, whose code path the edit does not touch. Q4KM live controls were rerun
alternately on both runtimes (target-only / Baseline model seconds):

| Run | Started | Runtime-v6 | Runtime-v7 |
| --- | --- | --- | --- |
| Original panels | 17:59 / 19:04 | 75.9 / 51.4 | 78.1 / 55.4 |
| Recheck 1 | 19:43 / 19:46 | 76.1 / 51.8 | 77.5 / 54.9 |
| Recheck 2 | 20:15 / 20:12 | 75.8 / 51.2 | 76.1 / 52.9 |
| Recheck 3 | 20:20 / 20:17 | 75.9 / 51.1 | 75.7 / 51.2 |

Another agent's native build compiled on the CPU during recheck 1. Once the host was quiet,
runtime-v7 matched runtime-v6 within 0.4%. This fits the code: in Baseline's live calls
runtime-v7 performs the same operations as runtime-v6, in a different order within the
host-side loop. The original v7 panels, and the references recorded from them, probably ran
under similar CPU load. The v7 screen panels also ran 1.6–4.8% slower than v6's for every arm;
the 120-request panels 0.4–1.5% slower and the harness panels within 0.7%. Ratios within a panel are paired and unaffected,
but times compared against these saved controls are descriptive, as the protocol already
states for historical controls.

## Scope and limitations

- One RTX 5080 under WSL2, Qwen3.5-4B Q4KM and NVFP4 targets, depth 3, exact verification,
  F16 KV, 32,768 context, greedy, DEV cohorts only. Each panel ran once.
- The edit applies only when the target can roll back the dropped tokens: no full-state
  checkpoint context, and a draft no longer than the recurrent rollback window. That always
  holds at the benchmark depth on these hybrid targets. Contexts that restore checkpoints
  keep runtime-v6 behaviour. The edit covers any stop inside an accepted run (end-of-turn,
  stop strings), but only end-of-turn stops were observed.
- Live exactness is checked on 12 tasks from one project; other drafters or longer sessions
  could still meet a divergence from another cause.
- The live timing rechecks cover Q4KM target-only and Baseline only.

## Decision

Done here, as the task assigned: runtime-v7 replaces runtime-v6 for protocol v3. Request
policies and server arguments are unchanged.

The existing v3 references could not be reused. A reference's protocol includes the
runtime build record, server binary and library hashes. A comparison refuses any reference
whose protocol differs, and the registry keys records by protocol. The request cohorts did
reproduce runtime-v6 for every arm, but reusing their v6 records under v7 would need a rule
that treats two runtime identities as equivalent. That would weaken the binding.

Re-recording needed no extra GPU time: the eight exactness panels are complete three-arm
panels under the new protocol, and their target, Baseline and S4 arms became the 24
references. S4 is the prior leader on all four cohorts for both precisions, including live.
Runtime-v6 records stay valid for runtime-v6.

## Evidence and reproduction

[result.json](result.json) holds every arm summary (runtime-v7 and runtime-v6), the paired
comparisons, the call- and response-level exactness counts, the timing rechecks, GPU
seconds per run, runtime and model identities and the reference names.
[Executed scripts](executed-scripts.txt) list every script in order of use; their
`build_result.py` regenerates result.json:

```sh
python3 build_result.py result.json
```

- Edit: `stop_token_cache_v1` in [`benchmarks/runtime_edits.py`](../../benchmarks/runtime_edits.py).
  The unpacked runtime-v7 source differs from runtime-v6's only in `tools/server/server-context.cpp`.
- Runtime-v7: `native-build` of `artifacts/native/runtime-v7-build-spec.json`
  (`61d34dd0…`) in the main checkout, from branch commit `996fadcb` frozen as code
  `8abfc54c…` under `artifacts/native/executed-sources/`. It is runtime-v6's spec plus the new
  edit. Build record `5898a4fb…`, `llama-server` `537d4a86…`, repairs
  `rejection_eog_boundary_v1`, `checkpoint_warm_buffers_v1`, `device_feature_handoff_v1`,
  `stop_token_cache_v1`.
- Panels: code `8abfc54c…`, frozen under `artifacts/benchmarks/runtime-v7-depth3/executed-sources/`.
  Specifications in `establish/` are runtime-v6's lean `*-establish.json` with only
  `source`, `binary` and `libraries` replaced. The workloads are unchanged.
- Models: Q4KM target `38742092…`, NVFP4 target `5447bdb8…`, Baseline `83586a63…`, S4
  export `10ade33a…` (`artifacts/training/block-test-s4-01`).
- New references in `artifacts/benchmarks/references/` (file name prefixes):

| Cohort | Q4KM target / Baseline / S4 | NVFP4 target / Baseline / S4 |
| --- | --- | --- |
| Screen | `4e854be6` / `27adb2a2` / `b7d1775b` | `d93a8b87` / `4cb023da` / `b2302b51` |
| Panel | `685d2c3a` / `6b392635` / `fedb59a2` | `6e6ce7cd` / `43f79238` / `5d53146b` |
| Prefix harness | `c75dc88a` / `45a2e5f0` / `9d589b87` | `993ad5ab` / `6d2ebcf5` / `8cdda967` |
| Live | `0271a8a1` / `513b5396` / `ae2bda4b` | `5b575c57` / `34cf48b6` / `3357b6f6` |

- All 16 candidate-only and leader specifications passed `verify-inputs` with their
  historical arms authenticated.
- GPU: 50.9 minutes of command wall time under the project lease, including model loads.
  That is 35.8 minutes for the eight panels (19:04–19:41) and 15.1 minutes for the six
  rechecks (19:43–20:23) on 2026-11-13. The build took 7 minutes of CPU time.

## Artifact disposition

All in the main checkout's ignored `artifacts/`; nothing existing there was changed or deleted.

- **Retained, current inputs:** `native/runtime-v7/` (504 MB, the build with its unpacked source),
  `native/runtime-v7-build-spec.json`, `native/runtime-v7-build.log` and the build's
  frozen code under `native/executed-sources/8abfc54c…`.
- **Retained, reference dependencies:** `benchmarks/runtime-v7-depth3/` (140 MB with the rechecks): `establish/`,
  the eight source panels in `runs/`, `registrations/`, `specs/`, `logs/` and the frozen
  code. The 24 new registry records authenticate these runs by path, so they stay while
  the references are in use.
- **Retained, timing evidence:** the six `runs/v*-q4km-live-recheck*` runs and
  `recheck/q4km-live-controls.json` (66 MB), summarized in result.json. They could
  be retired with the retirement command once nobody needs to re-inspect their transcripts.
- **Retained, previous protocol:** `native/runtime-v6/` and `benchmarks/references/runtime-v6-depth3/`.
  Their records remain valid for runtime-v6 and their lean workloads are inputs to the v7
  specifications. Retiring runtime-v6 would end exact replay of the v6 panels.
- **Disposable:** CUDA kernel cache growth under `cache/cuda`.
- Exact replay of every v7 run remains possible from the retained runtime, models and
  specifications.

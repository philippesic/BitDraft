# On native Linux the leader runs at 0.96–1.00 of Baseline's speed; acceptance, not drafter time, now limits it

Test-list item 9, 2026-11-16. The host moved from WSL2 to native Linux, so every saved
control had to be re-recorded. This records the new controls with runtime-v8, which adds
the recovered tensor-core one-bit kernel (`kernel_recovered_imma_v1`) to runtime-v7's edits,
and profiles where a round's time goes on the new host.

- Native Linux shortens every round by 1.0–1.3 ms (about 11%) for Baseline and S4 alike.
  The drafter's host-side wait fell from 1.0–1.4 ms to 0.4–0.6 ms; the rest came off the
  target's side. Target-only requests run 9–12% faster. The gain is generic: it helps
  Baseline as much as the candidate.
- The leader (export `5749a05f…`) runs at 0.96–1.00 of Baseline's request speed on every
  panel, against S4's 0.89–0.95 on WSL. Its acceptance stays 4.6–7.9 points below Baseline's.
- At depth 3 the leader's drafter takes 1.20 ms against Baseline's 1.54 ms (short prompts),
  but its round is only 4% shorter, because target verification is about 80% of the round.
  It proposes fewer accepted tokens per round (1.98 against 2.09), which cancels the gain.

## Outcome

Request panels, runtime-v8, depth 3, greedy, exact outputs on all 24 arm-panels. Paired
intervals are source-group bootstraps from the same three-arm run.

| Panel | Baseline / target | Leader / Baseline (95%) | Acceptance Baseline → leader | WSL: Baseline / target, S4 / Baseline |
| --- | ---: | --- | --- | --- |
| Q4KM screen | 1.416 | 0.992 (0.980, 1.005) | 60.6% → 55.2% | 1.347, 0.945 |
| Q4KM broader requests | 1.470 | 0.965 (0.956, 0.974) | 53.3% → 46.2% | 1.422, 0.887 |
| Q4KM prefix | 1.741 | 0.989 (0.952, 1.028) | 74.1% → 67.7% | 1.688, 0.920 |
| Q4KM live | — | — | 70.1% → 73.1% | — |
| NVFP4 screen | 1.416 | 0.998 (0.983, 1.013) | 58.2% → 53.6% | 1.347, 0.925 |
| NVFP4 broader requests | 1.461 | 0.958 (0.949, 0.968) | 53.0% → 45.1% | 1.416, 0.888 |
| NVFP4 prefix | 1.773 | 0.993 (0.968, 1.023) | 75.5% → 70.1% | 1.712, 0.918 |
| NVFP4 live | — | — | 59.7% → 60.6% | — |

Target-only throughput, Q4KM screen: 161.3 tokens/s on Linux against 143.4 saved on WSL.

Round profile, clean clock, ms per round (drafter / rest), same requests, runtime-v7
diagnostic and Nsight Systems 2025.5.2 on both hosts:

| Drafter, depth, prompts | WSL | Native Linux | Drafter host or idle, WSL → Linux |
| --- | --- | --- | --- |
| Baseline, 3, short | 9.05 (1.80 / 7.25) | 8.06 (1.54 / 6.52) | 1.18 → 0.44 |
| Baseline, 3, long | 11.83 (2.05 / 9.78) | 10.47 (1.72 / 8.75) | 1.20 → 0.49 |
| S4, 3, short | 8.96 (1.72 / 7.24) | 7.90 (1.47 / 6.43) | 1.10 → 0.47 |
| S4, 3, long | 11.56 (1.94 / 9.61) | 10.30 (1.62 / 8.67) | 1.19 → 0.52 |

Runtime-v8 on Linux, depth 3: Baseline 8.03 ms short / 10.47 long, leader 7.71 / 10.05.
Accepted tokens per round: Baseline 2.09 / 1.62, leader 1.98 / 1.44. Leader drafter
components (short): fused gate/up 0.069 ms, down 0.088 ms (Baseline's 4-bit feed-forward
0.373 ms), vocabulary scoring 0.414 ms (Baseline 0.584 ms), attention projections 0.200 ms,
feature handoff 0.146 ms. Depths 1 and 7 and all cells are in [result.json](result.json).

## Scope and limitations

- One run per panel and arm; timing intervals come from arms run one after another in one
  process, not interleaved trials.
- The WSL columns used a different candidate (S4) and runtime-v7, so the leader's change
  against Baseline mixes host, kernel, frozen-table format and model state. Only the round
  profile isolates the host.
- Thinking, sampled and agentic panels were not re-recorded; Level 2 and 3 decisions on
  this host still need them.
- Live panels report acceptance only.
- The candidate-only specifications were exercised only for the Q4KM screen (item 10).
- Nsight Systems was unpacked from the Ubuntu package into `~/.local/opt/nsight-root`, not
  installed system-wide.

## Decision

Decided by the owner on 2026-11-14: runtime-v8 and these records are the current protocol
on native Linux. WSL records stay valid for their own host and cannot serve Linux runs.

Proposed: treat acceptance as the main lever. Even a drafter that took no time would shorten
a round by only about 19%, while each acceptance point is worth about 1% of speed.

## Evidence and reproduction

- **Runtime.** `artifacts/native/runtime-v8/result.json` (`d613abbe…`) and the diagnostic
  `artifacts/native/runtime-v8-round-profile/`, both built natively from executed source
  `3587d764…` (main `7af81924`) with the specifications in
  `artifacts/benchmarks/runtime-v8-linux-01/specs/`.
- **Panels and references.** `artifacts/benchmarks/runtime-v8-linux-01/`: `establish/`,
  `runs/`, `registrations/`, `references.json` (registry records per panel and arm),
  candidate-only `specs/*-candidate.json` and `*-leader.json`, `logs/`, `preflight.md`.
- **Profiles.** `profile-v7-host/` (unchanged runtime-v7 specification, comparable with
  `artifacts/benchmarks/round-profile-baseline-s4-v7-01/`) and `profile-v8-leader/`.
- **Models.** Baseline `83586a63…`, leader `5749a05f…`, S4 `10ade33a…`, targets Q4KM
  `3874209241c9…` and NVFP4.

## Artifact disposition

Retained: the runtime-v8 builds, all eight panel runs (the references authenticate them),
registrations, specifications, both profiles with their session traces, logs and the frozen
executed source. Nothing was removed.

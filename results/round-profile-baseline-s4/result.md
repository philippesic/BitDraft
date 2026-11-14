# S4's one-bit feed-forward does not save round time at depth 3

On the WSL2 host, target work and host waiting dominate a speculative round.
The small difference in drafter GPU time does not establish a request-speed advantage.

## Outcome

The primary November 13 record gives clean depth-3 short rounds of 9.186 ms
(Baseline) and 9.409 ms (S4). On the separate phases clock, target batch work
is 6.559 / 6.790 ms and round wall minus GPU busy time is about 3 ms.
That gap is host or idle time, not proof that WSL itself caused it.
Target batch work is distinct from the additional sampling/acceptance `verify` phase.
Feature handoff takes 0.421 / 0.494 ms on that clock and is excluded from
`draft_ms`; that counter is not the complete drafter cost.

Nodes-clock drafter GPU time is 1.514 / 1.469 ms. Baseline / S4 component
milliseconds at depth 3 on short prompts are: vocabulary scoring 0.573 / 0.401;
feed-forward including norm 0.370 / 0.361; attention projections 0.178 / 0.274;
handoff and context-cache update 0.135 / 0.173; token selection 0.100 / 0.098;
attention small steps 0.066 / 0.065; context attention 0.045 / 0.045
(0.197 / 0.193 on long prompts). Other elementwise and copy work remains in
the compact record; these selected components are not an exhaustive sum.

S4 feed-forward costs 0.221 / 0.361 / 0.647 ms at depths 1 / 3 / 7,
against Baseline 0.341 / 0.370 / 0.594. Per-slot work erodes the shallow-depth
saving. Vocabulary scoring follows table bytes at shallow depth: 497 / 341 MiB,
about 900 GB/s, nearly flat through depth 3 and roughly doubled at depth 7.
These are effective table-byte rates, not hardware bandwidth measurements.

Attention projections plus handoff cost S4 about 0.135 ms more at depth 3.
The runtime source explains a plausible cause: its exact-decoding shared-read
path groups four slots for Q4_K, Q5_K, Q6_K and NVFP4, with Blackwell, shape and
contiguity guards; Q4_0 does not qualify. The tables have comparable shapes,
not identical encoded byte sizes. Reading S4’s GGUF tensor metadata confirms
Q4_0 attention projection and fusion weights. This is an attribution supported by source,
not a measured Q4_K conversion speedup; item 8 must test causality and acceptance.

Correction to the preliminary summary: primary phases-wall overhead spans
**2.63–13.68%**, not 4–14%, relative to clean round counters. Nodes walls must
not be included in that range. Between runs the largest component-group delta
is 0.00283 ms and clean round time differs by up to 4.66% relative to primary.
Two runs do not resolve a small Baseline–S4 request-speed difference.

## Scope and limitations

External Baseline and S4 export `10ade33a…`, Q4KM target, runtime-v6, exact
verification, depths 1/3/7, three short and two long DEV prompts, 128-token
outputs, one RTX 5080 under WSL2. Each run has 12 drafter/depth/band groups;
all report identical outputs across clocks. Counts and traced-round denominators
are retained per group in [result.json](result.json). No heldout24 selection.
Run 02 is primary; run 01 is a repeat with a reported overlapping commit gate.
No new execution was performed for this closeout.

Runtime-v7 is now the default; reported identical request outputs do not prove
identical timing. Its separate profile is outside this checkout's closeout.
The planned native Linux move requires fresh controls and measurements; these
numbers describe only this WSL host.

## Decision

Retain the reusable profiler and use the component split to prioritize tests.
The owner has authorized speedup-list items 1 (kernel benchmark), 2 (zero-code
and slot-overlap measurement), 3 (RSR kernel test), and 8 (frozen tables to Q4_K).
Other listed experiments remain proposals. This result does not execute them.

## Evidence and reproduction

Paths below are relative to `/home/philip/BitDraft/`. Primary and repeat:
`artifacts/benchmarks/round-profile-baseline-s4-02/round-profile.json` and
`artifacts/benchmarks/round-profile-baseline-s4-01/round-profile.json`, with their
text summaries. The compact JSON extracts every arm's groups, clocks, request
and traced-round counts, output checks, reconciliation, model/runtime identities
and settings, checked by field equality against both records. Summary ranges
are recomputed from those groups; repeat deltas use matching group order.

Recipe: `artifacts/benchmarks/round-profile-specs/baseline-s4-v3.json`;
repeat via `python -m benchmarks profile-rounds SPEC NEW_OUTPUT` under the
project's GPU allocation rules. This closeout does not authorize that rerun.
The records identify runtime-v6 and diagnostic build results, servers, target,
drafters, workload and profiler by SHA-256. Diagnostic source is retained at
`artifacts/native/runtime-v6-round-profile/source/`; the shared-read branch is
in `ggml/src/ggml-cuda/ggml-cuda.cu` around line 2160; per-slot integer
dot products are in `ggml/src/ggml-cuda/w1a1.cu`. Build spec, log, source
bundle and binaries remain. Session commands, library maps and request rows
remain in both run folders. The report's `code_sha256` alone does not establish
an immutable snapshot of the Python process at execution: no such snapshot is
identified by these records. This provenance limit does not affect reading the
saved component measurements but limits exact reconstruction of the runner.

## Artifact disposition

Retain both JSON/text records, all non-trace session records, specification,
executed runtime source and dependencies, diagnostic runtime with build spec/log,
and model/data inputs. They support assessment and future reruns. Retain the
compact JSON and retirement plans/receipts with this result.

Retired only the 24 `.nsys-rep` files in the two named run folders through
`tools.retention`; no exports were present. Per-run authenticated plans and
receipts ([run 01](run-01-receipt.json), [run 02](run-02-receipt.json))
record 400,178,540 and 400,199,677 bytes removed, respectively: **800,378,217 bytes**
in total. Originals are stored under `artifacts/receipts/round-profile-closeout/` and preserved as JSON entry arrays beside this result. Once retired, the component split cannot be regenerated
from those raw traces. Future measurement requires a new profiling run;
retained summaries are not replacements for raw trace replay.

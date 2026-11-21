# The 284 MB file beats stock Baseline on all six protected request suites

On the frozen Q4KM confirmation requests, the 284 MB release file was 1.9–5.6%
faster than stock Baseline and 25.5% smaller on disk. The request criterion passed.
The separate agentic run produced zero completed assignments in every arm, leaving
agentic speed for useful work unconfirmed.

## Outcome

### Protected request confirmation

The four arms ran on the same RTX 5080, native Linux, runtime-v10, depth 3 and
32,768 context. Each suite has 20 requests. Sampled requests used three locked seeds.
Throughput is generated tokens divided by request seconds.

| Suite | 284 MB / Baseline | Paired 95% interval |
| --- | ---: | ---: |
| Chat and writing | 1.056× | 1.038–1.070× |
| Coding requests | 1.039× | 1.025–1.055× |
| Long documents | 1.025× | 1.017–1.033× |
| Reasoning | 1.032× | 1.011–1.052× |
| Thinking-on | 1.028× | 1.017–1.038× |
| Sampled | 1.019× | 1.011–1.026× |

Greedy outputs were exact against target-only. Sampled responses passed the
authenticated sampler qualification and output validity checks. The trained 4-bit
control used its usual Baseline layout. Its 21,000 training updates and the leader's
335,812 updates are different budgets. All arms and counters are in the compact record.
The protected campaign took about 51 minutes including live tasks.

### Protected agentic suite

All arms followed the target-only trajectory and completed 0/15 on Boltons.
Eleven assignments hit the repeated-assistant rule and four hit the turn limit.
Eight target-only final workspaces passed their checks but were still scored as
failures under the repeated-assistant rule. The observed candidate/Baseline rate
for assigned tasks was 1.121×. It measures time spent on these unsuccessful
assignments. The suite has one project, so a between-project interval is unavailable.
The maintained stricter verdict remains `not-established`, with all six request
scenarios passing and the agentic scenario insufficient.

### File forms and alternate tool calls on DEV

The 284 MB and 185 MB forms generated identical exact outputs and identical draft
counts on all 120 DEV requests, on the same runtime-v11 build. The 185 MB form had
the expected derived-row hash at all four starts.

| File form | Request tok/s | Time to model loaded | Resident GPU memory |
| --- | ---: | ---: | ---: |
| 284 MB | 298.19 | 0.80–0.81 s | 5,992 MiB |
| 185 MB | 298.43 | 6.40–6.42 s | 5,992 MiB |
| Stock Baseline | 277.48 | | 6,052 MiB |

The live crossover used the same 12 DEV tasks, tools and checks on runtime-v10.
The JSON format used JSON objects inside the native tool-call delimiters and carried
the JSON into assistant history. The pilot proved real calls and tool responses.
With native XML, every arm completed 5/12, and the candidate was 13.6% faster than
stock Baseline by assigned-task rate. With JSON, every arm completed 3/12, and
the candidate was 3.2% slower. The trained 4-bit control was 11.3% faster than
Baseline on XML and 0.7% faster on JSON. Candidate acceptance moved from 79.65%
to 46.74%, versus 61.78% to 59.94% for stock Baseline.

## Scope and limitations

The confirmation is a one-time independent check of the fixed 284 MB build.
Source families were screened against project TRAIN and DEV, with one RTX 5080,
one target format and one repetition per panel. The paired intervals measure
variation across request groups. The small tool crossover measures sensitivity to
one alternate convention. The 185 MB result covers DEV request parity, with
the startup measurements descriptive for this host.

## Decision

Lead with the 284 MB file, as decided by the owner on 2026-11-21. The release
throughput claim can cover the six protected request suites. The 185 MB form
qualifies as an optional download on runtime-v11 with about 5.6 seconds of
additional loading time. Scope agentic performance to the measured DEV harness
and withhold a protected agentic speed claim. The protected allocation is spent.

## Evidence and reproduction

Use the [compact evidence](result.json) for exact release hashes, runtime,
frozen source, measured arms, criteria, and paired inputs. The report and immutable
raw outputs live in `artifacts/release-qualification-01/runs/`, locked plans in
`specs/`, and execution receipts and preflight in `logs/`. The file-form
comparison used `benchmarks --frozen compare`. The crossover used
`benchmarks --frozen live` on the recorded source with the opt-in DEV mode.
The protected suites used the maintained comparison and live entry points.
`scenario-report` produced the strict verdict. The compact record includes
pseudonymous request and source-group labels in the original sort order and
enough counters to recompute timing and acceptance. Protected response bodies
remain in the private artifact output.

## Artifact disposition

Retain the measured release files, runtime dependencies, frozen specifications,
and protected receipts as release provenance while these claims are cited.
The project owns the auxiliary DEV results and JSON pilot evidence and retains
them through the release decision and any harness redesign. Keep past
references and original evidence unchanged. No raw outputs or input artifacts
were retired.

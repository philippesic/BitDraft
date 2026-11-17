# Benchmarks

Native runs measure acceptance/speed; loss/top1 diagnose learning. Routine training
checks execute only the candidate against saved target/baseline/leader records. The
complete pause must fit 120 seconds every 1800–3600 optimizer seconds; qualify the actual
pause before production. Request scores and whole application tasks stay separate.

```sh
uv run --locked --no-sync --offline python -m benchmarks COMMAND INPUT [OUTPUT]
```

Writing commands default to the matching folder under ignored `artifacts/` and reject
destinations outside it. [Storage](storage.md#new-output-locations) defines relative names,
explicit paths and the shared saved-reference registry.

| Commands | Purpose |
| --- | --- |
| `plan`, `verify-inputs` | Native comparison units/warmups; optional model/runtime hashing |
| `native-build`, `native-helpers` | Authenticated runtime/helper builds |
| `curate`, `admit-bank`, `select-supplement`, `select-expansion` | Role-safe prompt preparation/selection |
| `generate-sessions` | Target-only greedy TRAIN tool sessions for session captures |
| `plan-long-projects`, `prepare-long-projects` | Long-context TRAIN session tasks from full repository snapshots (CPU) |
| `prepare-project-workflows`, `prepare-replays`, `prepare-panels` | Project controls, measured histories, live/harness inputs |
| `capture`, `evaluate` | Teacher features; matched teacher-forced DEV loss |
| `replay-tokens`, `replay-agreement`, `write-batched` | Rebuild captures from stored token IDs, compare them, measure parallel writing ([data](data.md#token-streams-and-replay)) |
| `compare`, `live` | Isolated native requests; executing tool episodes |
| `campaign-preflight`, `campaign`, `campaign-report` | CPU admission; training; read-only health |
| `prepare-checks`, `record-reference` | Frozen lean cohorts; authenticated reusable references |
| `scenario-report` | Per-scenario paired comparisons and Level 2/3 decisions |
| `policy-report` | Owner tolerance policy on five release scenarios, beside the stricter verdict |
| `prepare-initialization` | Reusable authenticated model basis; no training or capture |
| `suite` | Broader panels and optional state selection |
| `verify-greedy`, `verify-sampler`, `profile-native` | Independent decisions, distributions, kernel activation |
| `reduce-trace` | Structural round accounting, not an independent verifier |
| `calibrate-depth` | Serving draft depth for one target, drafter, runtime and GPU |
| `build-kernel-benchmark`, `capture-kernel-activations`, `benchmark-kernels` | Exact kernel comparisons on supplied DEV activations |
| `profile-rounds`, `round-profile-report` | Where a drafter's time goes inside one drafting round |

Execution commands freeze source and use [the numerical environment](development.md)
where needed. Preserve inputs, logs, ownership and failures. Comparison outputs must be
new; capture resumes verified records. Resume campaigns/suites with their recorded source
and specification. Checkout edits do not alter active runs. CPU preflight is not runtime
qualification. Run [local checks](development.md); there is no hosted CI.

## Three evaluation levels

Level 1 checks training progress: the unchanged 60-request candidate-only screen and
120 fixed teacher-forced DEV windows, plus separate 20-window agentic-history and
20-window thinking groups. Loss/top1 are diagnostics, not speed evidence. Preserve the
120-second complete training pause budget; a standalone diagnostic timing does not
qualify a production campaign interruption.

Level 2 guides development decisions. Run the four routine cohorts below, the fixed
12-request thinking slice, the same 12 sampled requests with two seeds, and the
12-task `agentic-coding` slice. Every primary scenario must be visible. If any slice
is worse than Baseline or the prior leader beyond its paired 95% interval, or its
speed direction disagrees with the broader request panel, Level 3 is required before
a decision. Saved timing is descriptive; acceptance regression against a saved leader
also triggers escalation. Historical speed direction still participates in the disagreement
check. The report lists unresolved timing when no paired interval exists; absence of an
escalation trigger is not speed qualification. Invalid outputs or an agentic completion
regression also block clearance. Missing scenarios or the broader-request comparison
prevent a Level 2 clearance.

Level 3 is required to claim **faster**: all routine request coverage, all 40 thinking
requests, the same 40 sampled requests with three seeds, and the comprehensive
agentic-coding suite. For every primary scenario, the lower bound of the paired 95%
speed interval must exceed 1 against target-only and be at least 1 against Baseline.
Agentic coding decides: a request-panel gain cannot compensate for a workflow failure
or regression. Missing, invalid, sparse or inconclusive evidence cannot pass. A passed
DEV suite reports `confirmation-required`; the claim needs one successful run on
fresh, protected, source-disjoint confirmation data. Reused DEV is selection data.
heldout24 and its aliases remain the final reserve and are not consumed here.

Primary scenarios are `agentic-coding`, `coding-requests`, `chat-writing`, `reasoning`,
`long-documents`, `thinking-on`, and `sampled`. Structured/tool requests remain separately
reported as `structured-tools`; their regressions must also be inspected. Request/task
metadata may carry plain `scenario`, `project`, `style` and `task_kind` fields. An agentic project
uses `group: project:owner/repository` across its tasks and instruction styles, preserving project-level
uncertainty. Scenario tags do not manufacture independent observations.

## Evaluation contents and cadence

| Level | Panel | Units per candidate/target | Execution |
| --- | --- | ---: | --- |
| 1 | Teacher-forced diagnostics | Original 120 DEV windows + 20 agentic-history + 20 thinking | Separate metrics; no native speed claim |
| 1, 2, 3 | Training screen | 60 requests, 128-token caps | Candidate pass; saved controls |
| 2, 3 | Broader requests | 120 requests, existing longer caps | One pass; four ordinary primary scenarios plus structured/tools |
| 2, 3 | Prefix sensitivity | 6 groups × 3 prefixes = 18 requests | Supplementary sensitivity check |
| 2, 3 | Legacy live | 12 tasks, two per family | Supplementary application check; failures stay denominated |
| 2 | Thinking slice | 12 requests, thinking enabled, 2,048-token cap | One greedy pass |
| 2 | Sampled slice | Same 12 requests, thinking disabled × 2 seeds | Temperature 0.7; paired by request and seed |
| 2 | Agentic slice | 12 tasks, two per project | One greedy pass; [scope](#agentic-coding-panels) |
| 3 | Thinking suite | 40 requests, thinking enabled, 2,048-token cap | One greedy pass |
| 3 | Sampled suite | Same 40 requests, thinking disabled × 3 seeds | Same sampler as slice |
| 3 | Agentic suite | 60 tasks across six projects | One greedy pass; project-group intervals; failures stay denominated |

Level 2 is the complete development comparison, not just the broader120 panel. Level 3
replaces the three small slices with their full sets; nested rows are not independent
extra observations. The agentic slice completes only 5/12 Q4KM and 7/12 NVFP4, versus
37/60 in each full suite, identically across all three arms. Keep this coverage limitation
visible instead of selecting a more favorable slice after seeing results.
Measured costs are below for [routine cohorts](#routine-dev-cohorts),
[thinking, sampled and diagnostics](#measured-cost-of-thinking-and-sampled-coverage),
and [agentic panels](#agentic-coding-panels). Each distinguishes request/task time from
whole-arm allocation and projected slices from standalone execution.

## Routine DEV cohorts

Screen categories: 15 code, 12 documents, 10 reasoning, 10 conversation, 8 structured/tool,
5 writing. Input bands: 30 below 512 tokens, 18 at 512–2047, 9 at 2048–8191, 3 at 8192–16384.
The early-answer cap makes this a progress indicator; broader requests retain longer
answers. Category estimates are noisy. Broader categories 30/25/20/20/15/10 and input
bands 60/36/18/6 preserve breadth. Prefix groups split 3 code/3 structured, each under
minimal/coding/tools. Eight measured application histories remain in request coverage.

`prepare-checks` accepts schema 1 with authenticated `requests`, `harness`, `live`
comparison specifications, producing workloads and multi-arm `*-establish.json` specs.
Use them to establish missing controls, then bind references for candidate-only execution.
One-time multi-arm control establishment is timed separately; the 120-second screen
deadline applies to routine candidate-only execution and complete training interruptions.
The [bank definition](../benchmarks/definitions/prompt-bank-v1.json) owns source identities
and the 6K TRAIN bank; smaller evals do not change training data. Targets remain Qwen3.5
4B Q4KM/NVFP4, with separate reference records.

Legacy short120/full600/harness90/live×3/sample120×3 evidence keeps its original meaning;
these are supported readers, not routine defaults. Sampled slices use seeds 8101/8102;
comprehensive sampled sets add 8103. Extra repetitions answer explicit uncertainty.

These are reused DEV cohorts. Independent confirmation requires fresh curation and
access receipts; a new label supplies no unseen data. Never consume heldout24 aliases.
Public dialogue is partly synthetic, planning dominates non-arithmetic reasoning, and
long inputs are mostly papers. Length is not difficulty; this is not measured traffic.
One-project trials cannot establish universal transfer. Workflow regressions cannot be
rescued by request scores. Use the measured per-panel costs below to budget the complete
Level 2 comparison; the broader request panel alone does not represent its cost.

The four shared routine cohorts have these measured costs for final export `405e326b…`.
Target-only and Baseline columns are saved historical request/task seconds; HE4 was
executed candidate-only. Final panel wall includes its loading, warmup and bookkeeping.
These costs describe each format separately and do not imply matched timing intervals.

| Format / cohort | Saved target s | Saved Baseline s | Final HE4 s | Final panel wall s |
| --- | ---: | ---: | ---: | ---: |
| Q4KM / screen | 45.9 | 34.1 | 33.5 | 38.5 |
| Q4KM / broader120 | 261.3 | 183.8 | 195.5 | 202.7 |
| Q4KM / prefix18 | 23.5 | 13.9 | 14.4 | 21.4 |
| Q4KM / live12 | 81.3 | 58.8 | 53.5 | 60.4 |
| NVFP4 / screen | 41.5 | 30.8 | 30.9 | 35.3 |
| NVFP4 / broader120 | 243.9 | 172.2 | 182.3 | 188.2 |
| NVFP4 / prefix18 | 22.3 | 13.0 | 13.5 | 17.1 |
| NVFP4 / live12 | 54.8 | 42.1 | 39.7 | 48.8 |

### What each panel resembles

Measured on the Q4KM target-only arm; the training column is the frozen 6K TRAIN bank.
The earlier 220,095-update state was trained on the 15,647-capture corpus.

| Property | 6K TRAIN bank | Training screen | DEV requests | Live tasks |
| --- | ---: | ---: | ---: | ---: |
| Mean input tokens | 625 | 1,556 | 1,366 | 4,747 context, 644 new per call |
| Inputs of 2,048 tokens or more | 8% | 20% | 20% | 87% of calls |
| Output per request | to EOS or 256–2,048 cap | 110 mean; 47 of 60 hit 128 | 400 mean | 92 per call, 1,024 cap |
| Project `tools` harness prefix | 200 of 6,000 groups | none | none; harness panel separate | every call |

The screen sees only answer openings, and inputs there are longer than in training. Live
calls reuse a cached context and evaluate about seven new prompt tokens per generated
token, so drafter prompt cost matters most there. Function-calling prompts (Glaive) count
as structured requests without the project harness. None of these is measured user traffic.
The [TRAIN expansion](../results/train-bank-expansion/result.md) adds 874 whole tool
sessions in the live harness format (1.6K–6.8K tokens, about 12% of positions supervised)
on separate TRAIN projects; it does not change these DEV panels.

The new agentic panels resemble focused library maintenance with a checked follow-up:
44 seeded bug fixes, eight test-writing tasks, four features and four refactors.
Initial context averages 16,304 tokens (11,523–19,865), drawn from relevant source.
Each style appears twenty times; the slice uses two styles per project. The 1,024-token
cap applies per assistant call, with at most sixteen calls per task. These small
authored patches exercise real sandboxed
tools and growing context, but do not model hours-long production issue resolution.

## Reusable references

`record-reference REGISTRATION REGISTRY` accepts schema 1 with authenticated `comparison`
(the desired cohort) and completed `result`, plus source `arm` and zero-based `rep`.
It authenticates whole source-panel completeness, selected raw responses/counters and
exact unchanged task definitions. Projection from a larger complete panel is explicit;
partial/invalid panels cannot establish references. Changed caps require new evidence.

The `artifacts/benchmarks/references` registry keys records by protocol and model content identity. Its first
canonical observation is reused, never overwritten. Separate observations remain under
`observations/`; deliberate repeats cannot silently replace the reference. Raw result,
specification and response/transcript receipts remain attached. Model identities persist
even if obsolete weights are later retired with receipts; executing weights still requires
full authentication.

Each record binds the depth its arm actually ran at: its protocol is the panel protocol at
that arm's depth (the arm's `depths` entry, otherwise the top-level `depth`), so a Baseline
recorded at depth 4 has a different registry key and cannot stand in for depth 3. A
reference from a per-arm panel is interchangeable with one from a uniform panel at the same
depth. Records made before per-arm depths authenticate unchanged.

Candidate-only comparison specs contain one `candidate` arm and `references` mapping
`target`, `baseline` and optionally `prior-leader` to authenticated registry records.
A saved reference is checked against the depth its name is declared at in `depths`
(default the top-level `depth`); a mismatch is refused.
Preflight refuses missing, damaged, incomplete or incompatible evidence before training.
Protocol binds runtime/server/libraries, target precision, context/depth, role, exact
messages/tools/caps and generation policy. Cached target signatures validate fresh greedy
outputs/stopping. Hardware/driver identity is checked at execution. Reports distinguish
fresh rows from historical summaries and report acceptance differences with source-group
uncertainty. Historical speed ratios are descriptive, without paired timing intervals.
Changed runtime/hardware needs new controls; consequential close speed decisions need
fresh matched trials. Filesystem/tool tasks retain whole-episode failure denominators.

### Current protocol

Request policy v3 (`native-greedy-request-v3`, `native-live-v3`) serves with runtime-v8
(`artifacts/native/runtime-v8`) on the native Linux host, exact verification, depth 3,
`--cache-ram 0`, F16 KV and 32,768 context. Runtime-v8 is runtime-v7 plus
`kernel_recovered_imma_v1`, the recovered tensor-core one-bit kernel
([kernel benchmark](#recovered-split-reduction-tensor-core-variant)); runtime-v7 is
runtime-v6 plus `stop_token_cache_v1`, which keeps a stopping token out of the slot cache
when it was an accepted draft ([stop-token cache](../results/stop-token-cache-exactness/result.md)).
Target, Baseline and leader references exist for both targets on the four routine cohorts
([native Linux references](../results/native-linux-references/result.md)); the leader is
export `5749a05f…`. Thinking, sampled and agentic references have not been re-recorded on
this host. The shared root is `artifacts/benchmarks/runtime-v8-linux-01/`:

| Path | Contents |
| --- | --- |
| `establish/` | Three-arm (target, Baseline, leader) specifications, one per precision and cohort, using the lean workloads in `references/runtime-v6-depth3/lean-*` |
| `runs/` | The source panels the references authenticate; never delete while referenced |
| `registrations/`, `references.json` | `record-reference` inputs and the registry record of each panel and arm |
| `specs/` | Candidate-only specifications with target and Baseline references and the leader as a placeholder candidate (`*-candidate.json`; `q4km-screen-candidate.json` is the training screen), `*-leader.json` that add the leader as prior leader, and the build and round-profile specifications |
| `executed-sources/` | The frozen code that ran the panels |

Swap the `candidate` drafter in a copy of a `specs/` file to check a new state. Records of
earlier protocols stay valid for their own runtime and host: runtime-v7's WSL records are in
`artifacts/benchmarks/runtime-v7-depth3/` (including the scenario-level specifications), and
runtime-v6's in `artifacts/benchmarks/references/runtime-v6-depth3/`. A comparison refuses
every reference whose runtime, protocol or hardware differs from its own, so WSL controls
cannot serve Linux runs. Runtime-v6, v7 and v8 give identical outputs and draft counts, so
their acceptance is comparable; speed ratios across hosts or runtimes are not.

## Prompts and captures

`curate` authenticates pinned sources and native token counts. `admit-bank` rechecks
selected fingerprints and known/opaque exclusions. Labels follow requests rather than
incidental context words. Tool stages are explicit; translations name a destination.
Optional `development_replays` retains eight first-file-read histories from measured
project trials, including failures, once per source group. Actual tools/messages and
transcript identities remain attached. Optional `harnesses` adds the intended `tools`
prefix to 100 TRAIN code-edit and 100 TRAIN tool groups. Preserve their ancestry/roles;
the alternate `coding` prefix stays outside TRAIN.

Selection keeps category/subtype/length marginals and required examples. It can choose
shorter complete records within cells to fit 399 decimal GB of maximum raw FP16 features,
leaving about 1 GB for metadata under 400 GB. FP32 raw features are twice that. No padding/
truncation manufactures coverage. Exact inputs/caps set the maximum, EOS the actual size.
These are dataset planning ceilings, not global disk limits. [Data](data.md) owns formats,
partial tails, mixtures and ancestry.

`generate-sessions` builds TRAIN coding projects from CommitPackFT repository groups that
share no repository with the curated pool, skips groups whose hash marks them as DEV, keeps
permissive licenses only and drops trailer and contact lines from commit messages. Each task
copies two to eight parseable files, asks for the commit's change and checks that the file
changed and every file compiles. It runs the live harness (`tools` system prompt, the five
file/test tools, 16 turns, 1024-token calls) with the target alone and records transcripts
and responses; it resumes completed tasks, and an optional `skip` starts after earlier tasks
of the same seeded order. Task success is not required: the target's own
greedy turns are the supervision. `select-expansion` selects a TRAIN expansion bank from the
curated pool, new public sources (`math`, `oasst2`) and session transcripts, converting each
session into one [session record](data.md#session-records). It excludes base-bank, protected
and heldout identities and token prefixes, fills `source|subtype|band` quotas in seeded
order, rechecks tokenization and forecasts maximum FP16 features against a declared ceiling.

Long-project sessions start at 11–20K tokens of context, as the agentic suite does, and
come from CommitPackFT single-file commits placed inside their full GitHub snapshot.
`plan-long-projects` ranks the source's repository families in seeded order and keeps one
permissive, parseable change per family. It excludes the DEV agentic projects and the
Boltons confirmation project by repository family, renamed or prefixed names and upstream
commits from the agentic definition and its lineage audit, plus curated-pool repositories,
existing TRAIN session projects and DEV-marked groups. Snapshots at the pinned commit are
fetched outside product code into a manifest of URL, commit and archive SHA256.
`prepare-long-projects` (CPU tokenizer only) reads each authenticated archive and refuses
it unless every root licence file is MIT, BSD, Apache-2.0 or ISC text, the snapshot holds
the commit's new file exactly, no workspace path names a protected project and no file
matches protected upstream code (13-token shingles, at least twelve shared and 35%
containment, or a whitespace-insensitive exact copy). The workspace keeps up to the declared
number of parseable Python files nearest the changed file, with the changed file at its
pre-commit contents. The user message gives whole relevant files (direct imports, then
importers, then nearest files, the changed file last) followed by the commit message,
trimmed to the declared token band. The check compiles every file and requires the changed
file to differ. `generate-sessions` then runs these tasks when its specification names the
preparation result in the optional `prepared` field; a specification without it builds
the original CommitPackFT projects unchanged.

`select-expansion` takes these optional fields; omitting all of them keeps the original
selection. `thinking_quotas` (with `thinking_output_cap`) selects pool or source prompts
under the same quota keys and renders them with thinking enabled: the row carries
`thinking: true` and the enabled template's token identity, so `capture` writes it as a
thinking-on record. `session_input_limit` (16,384 to 32,768) keeps session turns beyond
16K tokens; such records use length band 4. `excluded_mixtures` excludes every record
identity, group and alias of corpus-v1 mixture components, `excluded_workloads` the
task identities of comparison workloads, and `additional_exclusions` further protected
identity lists such as the confirmation reserve's protection file.

Capture resumes the exact specification in its existing output: authenticate completed
`progress.json` records and skip their IDs. Keep incomplete attempt directories and
failure receipts. Before restarting, inspect `attempt-N/owner.json` against process
birth/boot and actual GPU clients; a missing controller or tool session does not prove
the native process ended. Preserve healthy owned work. A changed bank, selection, target,
precision or policy needs a new specification/output rather than editing the old one.
Use the recorded source when recovering an interrupted execution; checkout edits do
not replace its provenance.

Unattended execution needs a persistent host session or detached process with a recorded
controller identity and owned log. Verify the native owner and advancing validated
records before handoff. Read-only inspection does not require launching another teacher.
Completion requires the terminal result, exact requested IDs and product validation;
an existing data directory or a live process alone is insufficient. A user cancellation
ends the allocation; old specifications do not authorize restarting it.

[`capture`](../benchmarks/capture.py) binds teacher/runtime/tokenizer, bank/IDs, dtype,
context, byte ceiling, exclusions and supervision policy. Every new record passes the
bounded product loader before publication: hashes, geometry, finite values, token/label
arrays and EOG. Validation receipts and partial attempts stay with ignored raw data under `artifacts/`.
[FP16 adoption](../results/fp16-storage-adoption/result.md) qualifies HE4/Q4KM storage;
training converts it to FP32 on the device. It does not establish lifetime/other-family equivalence.

### Replay and batched writing

The capture helper's optional fifth argument `replay` raises its output buffer to 512
positions and accepts requests with `replay: {"chunk": N}` and spans; without it the helper
behaves exactly as before. `replay-tokens` (schema `token-replay-v1`) binds `source`, the
replay-capable `binary`, `libraries`, `target`, `corpora` and/or `streams`, `ids`,
`feature_dtype`, `context`, `chunk` (1–512) and `capture_limit_bytes`, holds the GPU lease,
resumes completed records like `capture` and refuses a target that differs from the records'.
`replay-agreement` (schema `replay-agreement-v1`, CPU) binds `reference` and `replay`
corpora and optional `ids`, `bands` and `rows`. `write-batched` (schema
`batched-writing-v1`) binds the server `binary`, `target`, reference `corpora`, `ids`,
`concurrency` (1–64 slots per level), `slot_context` and `seed`; each level starts its own
server with `-np` slots and `slots × slot_context` context, warms it with one short request,
then sends every prompt through that many concurrent requests. Server launches outside this
command keep one slot.

`plan COMPARISON.json` reports native units/warmups; `verify-inputs` hashes models/runtime.
The older schema 1 [metadata example](../tests/fixtures/benchmark-plan.json) remains supported.
Training CPU preflight is in [execution](execution.md). Neither grants authority.

## Native comparison

[`native-comparison-v1`](../benchmarks/comparison.py) binds build/server/libs, target,
arms, workload, repeats/warmups, context/depth, role and tier. References use `path`,
`sha256`, `bytes`, relative to the specification or absolute. Fresh control runs start with `target`; cached runs execute speculative arms only.
Keep external baselines and candidates distinct. Server/libs must match their build.

Greedy arms must match target token IDs, text and stopping; actual returned generation
settings are checked. Native servers set `GGML_CUDA_DECODE_CONSISTENT=32` so verification
batches of up to 32 tokens use one-token arithmetic; without it greedy outputs differ after
near-ties. Its cost and depth limit are [measured](../results/cuda-decode-consistency-cost/result.md). Accepted/proposed counters reconcile with independent native logs;
fallback-only arms are refused. Correction/bonus tokens are not accepted drafts. Zero
proposals make acceptance unavailable.

Request time covers pretokenized HTTP completion including prefill/decode, excluding
rendering/tokenization/load/warmup. Live time includes harness/tools. Models are loaded
and warmed, thinking disabled unless explicitly enabled in the frozen task, KV F16 and draft policy fixed. Independent requests/tasks
start with empty caches; live sessions reuse prefixes after their first call. `gpu.txt`,
commands, loaded libraries and ownership receipts identify execution. The lease preserves
other tasks' jobs.

Reports use total tokens/total time, acceptance totals, EOS/cap counts and category/length/
harness breakdowns. Panels draft at depth 3 for both targets (`--spec-draft-n-max`),
so a round proposes at most three tokens; 53% acceptance means about 1.6 accepted per round.
Depth 3 is Baseline's best exact depth and within about 1% of each measured drafter's own
best ([depth fairness](../results/draft-depth-fairness/result.md)); the earlier depth 2
overstated HE4/Baseline speed ratios by about 8%. Fixed depth 3 remains the track for
acceptance diagnostics and comparisons.

For deployment speed, an optional `depths` mapping serves named drafting arms at their own
depth, for example `"depths": {"baseline": 4, "candidate": 3}`; other drafting arms keep
the top-level `depth`. It may name measured drafting arms and saved non-target references,
never a target-only arm. Omitting it keeps the previous behaviour and protocol identity.
When present, the panel protocol gains `arm_depths`, each arm's server runs with its own
`--spec-draft-n-max`, and results record `arm_depths` plus a `depth` beside each summary's
counters and `derived_rounds` (null for target-only). `live` and agentic specifications
accept the same field. Summaries therefore add `derived_rounds` (emitted − accepted − requests: each round emits
its accepted drafts plus one target token, and the first token comes from prompt processing)
with proposed, accepted and emitted tokens per round. On the 1,200-request Q4KM panel this
matched the native round counters within 0.1%. Requests ending at EOS match exactly; a
request stopped by its token cap can count one round more than the runtime's draft-round
counter, consistent with a final capped token emitted without a draft. `speed_vs_target` and `speed_vs_baseline` divide each arm's tokens/s by
that control's; against historical controls they are descriptive, not paired timing.
`server_prompt_seconds` and `server_prediction_seconds` sum the native server's own
prompt-processing and token-prediction partitions (null when a historical row predates
them). Drafters add prompt work, so keep the split visible: in the live panel it cancelled
the decoding gain ([standing](../results/he4broad-final-standing/result.md)). That extra time is
runtime overhead shared by every DFlash-style drafter: host-side feature copies and larger
context checkpoints, not drafter compute ([prefill overhead](../results/speculative-prefill-overhead/result.md)).
Runtime-v6 removes most of it with outputs identical to runtime-v5
([fixes](../results/drafter-prompt-overhead-fixes/result.md)); panels use its successor runtime-v8.
Servers run with `--cache-ram 0`: the RAM prompt cache saved the previous request's state
before each new one (about 90 ms per request, outside `server_prompt_seconds`) and nothing
reused it. Live sessions reuse their prefix from the slot's own cache, which stays on.
Paired group bootstraps preserve category composition and keep repeats/
variants together. Sparse strata make intervals unavailable; multiple candidates also
receive adjusted intervals. Wide intervals are inconclusive. Compare matched arms within
targets, without pooling away regressions or mixing absolute scores across changed eras.
Sampled mode declares temperature/top-p/top-k/min-p, fixed seeds and target/runtime-bound
qualification. The new cohorts use temperature 0.7, top-p 0.9, top-k 0, min-p 0 and no
repetition/frequency/presence penalties, matching the retained sampled setup. The target
sampler defines the output distribution; exact sample-and-match/rejection verification
preserves it mathematically. `verify-sampler` tests both native algorithms, zero support
and EOG against controlled probabilities. Passing finite tests qualifies the implementation
within that scope; it does not prove neural output quality or all distributions.
The maintained serving policy uses greedy draft proposals and target sample-and-match:
it draws a target token at each accepted position and keeps a draft token only when
it matches. Fixed-seed paths can therefore coincide across arms under consistent32.
Rejection sampling is also verifier-tested, but is not this serving policy.
Pair the same request and seed across arms, keeping all seeds/variants within their source
group in the bootstrap. Compare aggregate emitted tokens / total request seconds, with
paired source-group 95% intervals. Outputs can differ despite identical seeds: do not
assert greedy equality, equal response quality or matched completion latency. Sampled
throughput is one required scenario, never a substitute for greedy or application evidence.

### Thinking and sampled cohorts

`thinking-slice`/`thinking-suite` contain 12/40 requests with `thinking: true` and
2,048-token answer caps. Selection is fixed before measurement: take coding and reasoning
requests from broader120, sort each category by SHA256 of `scenario-levels-v1:` plus its
source group, then take 20 per category; the slice takes six per category from that order.
`sampled-slice`/`sampled-suite` use the same IDs and caps, with thinking disabled and two/
three seeds. The slice is nested, not independent confirmation. Both target precisions
use the same selection. Admission checks independent source groups, balanced coding/
reasoning counts, thinking policy, caps and the fixed sampled settings above. Omitted
thinking means the existing disabled default. An enabled template changes token identity, so its old token hash
cannot be reused; native rendering and raw request/response receipts bind the new prompt.
The runtime receives `chat_template_kwargs.enable_thinking: true` for rendering; direct
completion consumes that tokenized thinking prompt. Greedy verification still compares
all generated tokens, content and stopping exactly, including reasoning and cap stops.

`record-reference` also admits qualified sampled panels. It retains every seed-specific
raw signature, without requiring cross-arm equality. Projection may select tasks and a
subset of source seeds; the source panel must still be complete. Existing greedy reference
identities and the campaign screen are unchanged. Sampled references use
`native-sampled-request-v3`; greedy and sampled controls cannot mix.

### Measured cost of thinking and sampled coverage

The final S4 data-curve export `405e326b…` was measured with runtime-v7, depth 3,
cache-ram 0 and exact verification on the RTX 5080. These are summed request seconds
per arm; full panel wall includes load, warmup and teardown. Level 2 costs project the
fixed request/seed slice from the completed Level 3 run, so their independent startup
cost is not measured. The 12/40-request thinking sets use one seed; sampled uses 12×2
and 40×3. Both drafters matched target-only exactly on all greedy outputs. Both native
sampler qualifications passed; all observed sampled paths also matched under the fixed
seeds, although sampled validity does not require that equality.

| Precision / cohort | Target request s | Baseline request s | HE4 request s | Full panel wall s |
| --- | ---: | ---: | ---: | ---: |
| q4km-thinking-slice | 100.9 | 58.9 | 66.4 | projected slice |
| q4km-thinking-suite | 342.2 | 196.8 | 225.0 | 797.6 |
| q4km-sampled-slice | 103.4 | 78.2 | 81.8 | projected slice |
| q4km-sampled-suite | 635.0 | 478.5 | 501.5 | 1658.5 |
| nvfp4-thinking-slice | 75.5 | 45.5 | 50.9 | projected slice |
| nvfp4-thinking-suite | 272.6 | 158.4 | 179.8 | 630.2 |
| nvfp4-sampled-slice | 105.9 | 80.7 | 86.2 | projected slice |
| nvfp4-sampled-suite | 619.7 | 476.4 | 499.5 | 1636.2 |

The separate 20 agentic and 20 thinking diagnostic windows add 10.6 seconds to the
training-time measurement: 2.9 seconds agentic and 7.7 thinking. The combined 160-window
pass took 20.6 seconds and preserved all original 120-window metrics exactly. The Q4KM two-update resident pilot measured a complete checkpoint/export/screen/restore/next-
sample interruption of 73.4 seconds, below the 120-second budget. Its supplementary groups
added 4.5–4.6 seconds per resident pass; startup and standalone timings remain separate.

### Scenario reports and decisions

`scenario-report SPEC OUTPUT` accepts `schema: scenario-report-v1`, `level: 2|3`,
`candidate`, authenticated evidence in `panels`, and `confirmation: null` (or
an authenticated successful fresh Level 3 report). Panels share the target/runtime/context/
role and candidate content; their arms may run at different declared depths, and each
panel's `arm_depths` is reported. Each panel retains its own scenarios; primary panels
cannot duplicate a scenario. The names `screen`, `prefix`, and `live` are supplementary
and do not replace the dedicated agentic suite. Level 3 requires the full thinking,
sampled and agentic suite tiers; a pilot or slice cannot supply its primary coverage.
Fresh confirmation needs at least 20 requests per request scenario and 15 agentic tasks.
Its thinking and sampled requests keep the same mode, cap and sampling policy as DEV;
sampled confirmation requires three fresh seeds and authenticated sampler qualification.
Use separate reports for Q4KM and NVFP4.

Each panel accepts a result artifact reference directly, or an object containing
authenticated `result` and desired `comparison` references, plus optional `references`
mapping every measured arm to its saved registry record. The latter projects a fixed
slice from a complete source suite. Source tasks, arms, protocol and seed selection are
authenticated before projection. Registry proofs must bind that same simultaneous
source execution and the exact selected rows; unrelated historical controls cannot gain
a fresh interval this way. Projected panels retain source allocation time separately and
leave standalone slice allocation time unavailable. The joint reports use this contract
for both slice and suite references, including the agentic evidence.

Reports include speed versus target-only and Baseline, paired 95% intervals for fresh
matched arms, acceptance, drafter bytes, and recorded resident GPU MiB. Application panels
also show completed/failed strata and completion counts; a completion regression cannot
pass the verdict. Agentic speed is assigned tasks divided by total primary task time,
including tools and failed tasks; it is not generated-token throughput. Project groups
stay together in the bootstrap. Loop-detection cost stays in raw cost reporting while
the repeated detection call remains excluded from primary task time. Request scenarios
use emitted tokens per request second. `arm_elapsed_seconds`, when recorded, includes model load, warmups and
teardown; summed row seconds measure only the panel requests/tasks. Resident memory
is a device-wide after-load observation, including target, KV, drafter and desktop; it
is not peak allocation or incremental drafter memory. Older runs without this receipt
report memory unavailable. Historical speed ratios never acquire paired timing intervals
through re-reporting. Source-group intervals cover prompt variation, not between-run drift.
The [joint final-candidate standings](../results/scenario-benchmark-levels/joint-standings.md)
include every primary scenario for both levels and formats. Thinking/sampled regressions
prevent a faster claim on both formats; Q4KM agentic also fails the interval rule, while
NVFP4 agentic alone narrowly passes it. Neither outcome supplies missing fresh timing
intervals for the ordinary request scenarios or replaces fresh confirmation.

### Owner-policy report

`policy-report SPEC OUTPUT` reports the owner's tolerance policy (working rule 2026-11-16)
beside the `scenario-report` verdict, which is unchanged. It is a separate command because
its five release scenarios split greedy requests by input length instead of by the seven
primary scenarios, and because it pools repeated runs. The specification has
`schema: owner-policy-report-v1`, the `candidate` arm name (Baseline is the arm named
`baseline`) and `panels`, mapping any of `greedy`, `thinking-on`, `sampled`,
`agentic-coding` and `screen` to a list of one or more runs. Each run is evidence in the
scenario-report form: a result reference, or `result`, `comparison` and `references` for a
slice projected from a complete suite. Tiers are `comparison` (broader120) for greedy,
`thinking-suite`/`thinking-slice`, `sampled-suite`/`sampled-slice`,
`agentic-suite`/`agentic-slice` and `screen`. The candidate must be measured in every run.
All panels share runtime, target, context, role, candidate and Baseline content; depths
may differ between panels and arms.

| Scenario | Rows | Speed |
| --- | --- | --- |
| `greedy-short` | broader120 requests with frozen `ancestry.input_tokens` below 2,048, every category including structured/tool | Emitted tokens / summed request seconds |
| `greedy-long` | The remaining broader120 requests | Same |
| `thinking-on` | Thinking suite (40) or slice (12) | Same |
| `sampled` | Sampled suite (40 × 3 seeds) or slice (12 × 2); seeds stay in their source group | Same |
| `agentic-coding` | Agentic suite (60) or slice (12); project groups | Assigned tasks / total primary task time including tools and failures |

Each scenario reports per-arm summaries with acceptance and depth, and the
candidate/Baseline comparison from the maintained paired source-group bootstrap
(`statistics.paired`, category-composition strata) over that scenario's own rows, with
its 95% lower bound, output correctness and agentic completion. Greedy input bands
(<512, 512–2047, 2048–8191, 8192–16384) and the 60-request screen are supplementary and
never decide.

Verdicts: `passed` when at most two scenarios have a lower bound below 1.00, each at 0.97
or better, every other lower bound is at least 1.00, every candidate output is correct
(greedy exact against target-only, sampled valid) and agentic completion is not below
Baseline's. `failed` when all five intervals exist and the rule is violated, or on any
correctness or completion loss. `not-demonstrated` when a scenario is missing or its
interval is unavailable: a sparse stratum, fewer than two source groups in a stratum, a
Baseline available only as a saved reference (no paired timing), or a slice where Level 3
needs a suite. `level3` is the release result; `development` accepts slices and is
labelled development-only for short-run finalists.

Several runs for one panel are pooled when they repeat the same panel protocol (workload,
sampling, runtime and depths) with the same arm models; arm order may differ, as in a
reversed-order repeat. Run *k*'s rows become repeats `k × repeats + rep` inside their source
groups, and none is dropped. Pooling narrows prompt-sampling noise only as far as matched
repeats agree; source-group intervals still do not measure drift between sessions.

Current limit: broader120's 24 long requests are 19 documents, 3 code, 1 conversation and
1 structured, so the conversation and structured strata have one source group each and
`greedy-long` has no interval under the maintained grouping. The policy therefore returns
`not-demonstrated` on this panel until its long coverage or the grouping rule changes; that
is a protocol decision, and pooled repeats do not add source groups.

Every scenario also carries `supplementary_merged_strata`: the same bootstrap with every
category stratum of fewer than two source groups merged into one pooled stratum
(`paired(..., merge_sparse=True)`), listing the merged strata. It never enters `level3` or
`development`. Whether it may count is the owner's decision; setting the optional
`"count_merged_strata": true` adds `merged_strata_verdicts`, the policy recomputed from these
figures, beside the unchanged maintained verdicts. It defaults to false.

To build a five-scenario report on runtime-v8, use fresh three-arm runs (target, Baseline,
candidate) so Baseline timing is paired. Candidate-only `*-candidate.json` specifications
compare against a saved Baseline and leave the interval unavailable. For greedy, copy
`artifacts/benchmarks/runtime-v8-linux-01/establish/q4km-requests.json`, replace the
`candidate` drafter, add `depths` for the deployment depths and run `compare`. Thinking,
sampled and agentic panels follow the runtime-v7 scenario-level layout: three-arm suite
specifications executed with `compare` (or `live` for agentic) into `runs/`, with slice
specifications and registrations projected from them. A Level 3 entry is the suite's
`result.json`; a development entry is `{"result": suite result, "comparison": slice
specification, "references": the slice's registry record for every measured arm}`.

```json
{
  "schema": "owner-policy-report-v1",
  "candidate": "candidate",
  "panels": {
    "greedy": [{"path": "runs/q4km-requests/result.json", "sha256": "...", "bytes": 1}],
    "thinking-on": [{"path": "runs/q4km-thinking-suite/result.json", "sha256": "...", "bytes": 1}],
    "sampled": [{"path": "runs/q4km-sampled-suite/result.json", "sha256": "...", "bytes": 1}],
    "agentic-coding": [{"path": "runs/q4km-agentic-suite/result.json", "sha256": "...", "bytes": 1}]
  }
}
```

Add a second entry to a list to pool a matched repeat. Use separate reports for Q4KM and
NVFP4.

### Frozen request confirmation reserve

`/home/philip/BitDraft/artifacts/benchmarks/scenario-levels-01/confirmation/` holds
120 new, unexecuted role-`confirmation` requests: 20 each coding, chat/writing,
reasoning, long documents, thinking-on and sampled. The agentic confirmation suite
is a separate workload. Sources are MBPP sanitized, BIG-Bench Hard, GovReport through
LongBench, and newly authored chat/writing. The frozen source revisions, licenses,
selection/access receipts and exact overlap checks live beside the bank. Source families
are disjoint from the current TRAIN and DEV banks; this is not a target-pretraining or
semantic-deduplication claim. `specs/` freezes both target formats and fresh sampled
seeds 9101–9103 without running inference.

Future training and DEV admission must include `confirmation/protection-v2.json` in
its exclusions, alongside the existing opaque protected identities. Do not change
historical specifications. `protection-receipt.json` authenticates this union and the
superseded first union, which incorrectly included TRAIN identities and was corrected
before use. `access.json` and `ancestry-audit.json` record curation-only access and no
training, calibration, selection or inference consumption. A confirmation run is a
separate explicit operation after the DEV verdict is eligible; never reuse this reserve
for routine decisions.

## Depth calibration

`calibrate-depth SPEC [OUTPUT]` picks the draft depth to serve a trained drafter with on a
given target, runtime and GPU, in two to four minutes. It is separate from training and
from the fixed benchmark depth. Schema `depth-calibration-v1` names an authenticated
greedy DEV `comparison` (runtime build, server, libraries, target, workload, context),
the `drafter`, `tasks` and `cost_tasks` counts, `survival_cap` and `cost_cap` output
caps, `max_depth` (2–15) and `exact` (exact verification, `GGML_CUDA_DECODE_CONSISTENT=32`;
false sets 0). Example: [panel calibration](../results/draft-depth-calibration/spec-example.json).
Servers take the benchmark arguments, including `--cache-ram 0`; calibrations recorded
before that change also counted the RAM prompt-cache save in their request overhead.

It runs under the GPU lease:

1. A survival run drafts at `max_depth` on `tasks` requests spread over categories and
   input lengths. The server's per-position counters give the share of rounds whose first
   *i* drafts were all accepted.
2. Short cost sessions at every depth from 0 (target only) to `max_depth` run `cost_tasks`
   requests chosen across input lengths. Each gives decode milliseconds per round, split
   into the drafter's own forward (server `dur` counter) and the rest (target verification
   and round overhead). The target cannot verify a batch without a drafter in this
   runtime, so the target part is measured through the drafting path.
3. The model predicts tokens per round at each depth by splitting each deep round's
   accepted run into rounds of the shallower depth. Summing per-position shares instead
   undercounts accepted drafts per round at shallow depths by up to 14%. It combines this with the round cost
   into decode and complete-request tokens/s. Complete time keeps the survival run's
   per-request prompt and request overhead, scaled by the measured ratio for target only.

`calibration.json` in the output records identities (build, server, libraries, target,
drafter, workload, GPU, code), settings, the cost curve, survival shares, predictions,
the best depth, depths within 2% of it, the gain over target only and per-category best
depths. Other per-request rows and server logs stay in the session folders. Cite the
record by its path and hash. Measured accuracy against panels is in
[the result](../results/draft-depth-calibration/result.md). Predictions are for the
calibration's own capped requests; outputs do not check exactness, which remains the
benchmarks' job. Category recommendations rest on few rounds.

## Round profile

`profile-rounds SPEC [OUTPUT]` splits one drafting round into target verification, drafter
work and the rest, and splits drafter work by component, for each drafter, depth and prompt
length band. Re-run it on every new candidate. Schema `round-profile-v1` names a clean greedy
DEV `comparison` (runtime, server, libraries, target, workload, context), a `diagnostic_build`
(the `native-build` result of the same runtime with `round_profile_nvtx_v1` appended to its
`source_edits`; any other difference is refused), the Nsight Systems `profiler` binary, named
`drafters`, `depths`, `lengths` (bands of `minimum`–`maximum` input tokens with a task count)
and `output_cap` and `exact` as in calibration. The current inputs are in
`artifacts/benchmarks/runtime-v8-linux-01/specs/profile-v8-leader.json` and the diagnostic
runtime is `artifacts/native/runtime-v8-round-profile`. Nsight Systems 2025.5.2 on the
native host is unpacked from the Ubuntu package into `~/.local/opt/nsight-root`.

For a future runtime, copy its clean `native-build` specification, append
`round_profile_nvtx_v1` to `source_edits`, and run `native-build` into a new diagnostic
build directory. Preserve all other build settings and the clean runtime. Copy the profile
specification, bind its `comparison` to that runtime's authenticated greedy DEV comparison,
and bind `diagnostic_build` to the new build's `result.json` (refresh path, SHA256 and bytes).
Keep drafters, bands, depths and output cap fixed when comparing runtime versions.
The build requires every source edit to match exactly once; profile admission verifies
that the diagnostic differs from the clean build only by the profiling edit. Rebuild and
measure before adopting the new profile specification.

The v7 qualification is retained in `artifacts/benchmarks/round-profile-baseline-s4-v7-01/`
(`round-profile.json` and `v6-comparison.json`). It used 192.18 seconds under the GPU lease.
All three clocks produced identical outputs in all 12 drafter/depth/band cells, and all
18 session output records matched v6 run 02. Node attribution was 100% within floating-point
roundoff. Every node component changed by less than 0.004 ms; clean round times changed
by −4.74% to +0.18%, within the prior 5% repeat variation. This single run establishes
compatibility, not a runtime speedup. Preserve the v6 records as historical measurements.

Each drafter and depth runs three server sessions under the GPU lease, on the same requests:

| Clock | Runtime | What it measures |
| --- | --- | --- |
| `clean` | Unmodified | Server counters: decode ms per derived round and the drafter's generate counter (`draft_ms`) |
| `phases` | Diagnostic, Nsight, CUDA graphs as served | Named host ranges per round and the GPU work launched inside each |
| `nodes` | Diagnostic, Nsight, CUDA graphs off, one range per graph node | GPU kernel time per graph node, grouped into components |

The diagnostic runtime marks drafting (`srv.draft`), the target's checkpoint copies, the
target verification batch, the feature handoff into the drafter's context cache
(`srv.process`), sampling and acceptance (`srv.verify`), and inside each graph build, input
upload, launch, synchronisation and the drafter's token selection. With
`BITDRAFT_NVTX_NODES` set, every graph node, and every node folded into a fused kernel,
carries its operation, output shape and source tensors. A round runs from one `srv.draft` to
the next with no prompt batch between them.

Components come from those names, not from a fixed model layout: weight tensor names
(`blk.N.attn_q`, `ffn_down`, `fc`, `output`, …) first, then operations (flash attention,
rope, cache writes, top-k), and the drafter phase they run in. The tables `WEIGHT_ROLES`,
`OP_ROLES` and `GROUPS` in [round_profile.py](../benchmarks/round_profile.py) own that
mapping; extend them for a new architecture. Each component reports GPU ms per round, weight
bytes read per round and multiply-adds per slot (one row of its input), from node shapes.

`round-profile.json` holds every clock under its own name, averaged per traced round, with a
`reconciliation`: the clean drafter counter against the traced draft wall, drafter kernel
time on the phases and nodes clocks, and the share of nodes-clock drafter time attributed to
graph nodes. `round-profile.txt` is the short table. Traces stay in the session folders until authorized result closeout;
`round-profile-report SPEC OUTPUT` rebuilds both files from them without the GPU.

Limits: Nsight's API tracing lengthens the phases round (measure the gap in each run, never
pool it with clean time). The nodes clock exists for kernel attribution; its walls are slower
because graphs are off. The server's `draft_ms` covers only drafting; the feature handoff
(`srv.process`) is drafter work outside that counter. Clean runs cannot split target
verification from the handoff and sampling.

The [Baseline/S4 WSL2 result](../results/round-profile-baseline-s4/result.md) records
the runtime-v6 component split and trace retirement. Its retained summaries cannot
regenerate the raw trace analysis.

## Live application and instructions

`prepare-project-workflows` freezes source/recipes and verifies broken, repaired and
followup controls in real project modules. `prepare-panels` creates live inputs and 30
fixed-history groups; `prepare-checks` selects 6 of those groups under `minimal`, `coding`, `tools`. Record instruction origins;
identify authored alternatives. Treatments replace system messages, retaining task facts.

`live` executes list/read/replace/write/test tools in isolated copies. Bubblewrap protects
checks and excludes network, credentials and host source. Reads support ranges; edits
require unique matches; truncation is explicit. Reset each task; count both initial and
followup phases. Limits: 16 turns/900 seconds per task, 1024 tokens per call. Retain warmups
outside primary totals and reconcile their counters. All failures/timeouts stay denominated.
Report completion, whole-task/model/tool/overhead time and greedy trajectory/filesystem
agreement, including shared failures. Faster incomplete work is not a win. Deployment
claims need trials in the intended application/configuration.

Live greedy agreement is checked, not guaranteed. A call that ends on an accepted draft
used to leave that draft in the slot cache, so the next call reused one more cached token
than target-only, batched its new prompt differently and could flip a near tie; under
runtime-v6 S4 diverged this way on 1 of 12 Q4KM and 2 of 12 NVFP4 tasks. Runtime-v7 drops
the stopping token and anything after it from the cache, so drafter arms leave the cache
as target-only does. A diverged arm still makes its panel unusable for `record-reference`.

## Training and selection

[`training-campaign-v2`](../benchmarks/campaign.py) binds intent, candidate-only screen,
initial drafter, optimizer-time cadence, failure policy and `maximum_pause_seconds` ≤120.
`campaign-preflight` validates roles, model/data/source and saved controls on CPU. V1 is
legacy-pilot-only. V2 pilots accelerate cadence for tests while retaining all 60 prompts.

Boundary/final checks keep the same trainer and Adam/RNG state alive. Completed gradients
are released; trainable parameters and optimizer moments use a CPU mapping of that
worker's just-published checkpoint while frozen model buffers remain on GPU. File-backed
pages can be reclaimed instead of retaining another anonymous-memory model copy.
The candidate runs under the same owned GPU lease,
then trainable state returns before the next update. This avoids reconstructing the trainer.
GPU memory and host paging still require measured qualification for a new model/hardware.
A new acceptance leader
publishes a reusable reference; later checks compare with that record without rerunning
its model. The candidate must beat that leader by more than the paired source-group
interval in `historical_comparisons["candidate/prior-leader"]`; otherwise the earlier
state stays leader ([selection](domain.md#selection)). Invalid/partial evaluations never select.
Adam/RNG/cursor restore exactly; changed recipes/data require explicit continuation.
`training-boundary` and next `training-ready` journal events measure save/export, evaluation,
selection bookkeeping and restoration through next sample readiness.
The next sample must be materialized before readiness is
recorded; its cold data-loading cost counts in the pause. Capture caches and background
loads are released before native evaluation. A proposed leader stays pending until the full pause passes.
Boot/controller changes make
that interval unavailable. The campaign records interruption budgets and refuses continued
budget qualification after an overrun; final evaluation has no subsequent resume interval.
Initial startup is separate. Preserve all costs and incomplete attempts.
An optional authenticated `initial_reference` reuses the exact initial drafter's compatible
screen; it does not require another initial inference pass. Recovery retries a whole
incomplete candidate screen in a new output directory, retaining the failed attempt.
Normal repeated checks use one training worker. Journal capacity may rotate it into a
new supervised worker while preserving global budgets and durable recovery; that
rotation is not a check pause and does not count against the pause budget.
Recovery of a final check at the update guard reuses its completed evaluation and does
not launch another identical screen.
`initial_drafter` must match a completed producer bound to the durable parent, or a
zero-update export derived from that parent. An arbitrary same-name GGUF cannot seed a
campaign. V2's compatibility field `provisional_fast_leader` records an acceptance score;
it is not a throughput verdict. Pilot V1 campaigns measure every arm live and select on
tokens/s, requiring the lower bound of the paired candidate/prior-leader speed-ratio
interval to exceed one. Completed campaigns return idempotently without new runs.

For production, bind an authorized training intent, matching initial export, saved
60-request comparison and optional initial reference. Omit `pilot`, set cadence to 1800
optimizer seconds, `maximum_pause_seconds` to 120 and `evaluation_failure` to `stop`.
An intent's optional `model_update_seconds` sets a separate cumulative time allowance:
7200 requests two hours of optimizer work. `model-update-budget` is a complete campaign
stop with a final screen; the update-count guard still applies if reached first.
Run `campaign-preflight SPEC` before `campaign SPEC NEW_OUTPUT`; inspect with
`campaign-report OUTPUT`. Qualification recipes deliberately stop after a few updates;
copying their budgets does not create a production allocation. A new target precision
also needs matching teacher data and its own qualified controls.

Latest recovery, provisional fast leader and full-DEV incumbent are distinct.
[`suite`](../benchmarks/suite.py) prevalidates matching runtime/settings/role/target/arms.
`full-suite-v2` uses the lean one-pass panels and saved controls. `completed_panels`
reuses complete matching specifications, never partial blocks.
`candidate_bindings` references state/drafter/execution journals; completed producer or
boundary-publication receipts must bind the state to the evaluated export. Highest full-request score selects the experimental incumbent,
with earlier-update ties. Workflow/harness evidence stays separate; no deployment promotion.
Bindings may include a saved `prior-leader` as well as newly executed candidates. Each
must match that panel's authenticated model identity and retain its producer journal.
A ranking containing historical candidates is explicitly descriptive; close speed decisions
require fresh matched trials. Preserving a prior reference does not rerun its model.
The V2 suite's `targets` maps each precision to authenticated `requests`, `harness` and
`live` specifications, with optional `candidate_bindings` and `completed_panels`.
Use one greedy pass and the saved controls for every panel. Inspect individual outcomes
before interpreting an aggregate or recording an incumbent.

Diagnostics preserve RNG and report category loss/top1 on the original 120 fixed role-safe
DEV windows. An optional `groups` mapping in `training-diagnostics-v1` binds independent
`training-diagnostics-group-v1` artifacts with the same corpus/exclusions/windows fields
and exactly 20 fixed windows each, named `agentic-coding` and `thinking-on`. Each group
passes the same DEV/TRAIN/protected-identity and capture checks. Original top-level metrics
and reach-weight adaptation continue to use only the original 120 windows; group metrics
and elapsed seconds are separate. Agentic captures use measured DEV histories in the
TRAIN expansion's multi-span session format. Thinking captures use separate DEV requests
outside the 40 thinking comparison IDs, with target-generated thinking enabled. Both
report per-position top-1/prefix survival ([execution](execution.md#sampling-and-clocks)).
New training intents can bind the qualified 160-window configuration at
`/home/philip/BitDraft/artifacts/benchmarks/scenario-levels-01/diagnostics/training-diagnostics.json`.
Existing intents keep their bound diagnostics for exact resumption; the added groups do
not change the original 120 windows or their selection weights.
They are not acceptance. Measure the complete training interruption against the short
wall budget; do not stretch the training interval to accommodate an oversized check.
Keep model/data/evaluation/checkpoint clocks separate.
`campaign-report OUTPUT` needs no GPU: liveness, pointers, journals, learning/native scores,
costs/exposure. Attempts include retries; unique positions use actual label widths;
lifetime history stays unknown. Failed native attempts and controller failures remain
visible after recovery; a retained `failure.json` alone does not describe current health.
Inspect status, actual ownership and terminal journals before restarting. After code
changes, read `OUTPUT/binding.json` and run from its code snapshot under
`OUTPUT.parent/executed-sources/IDENTITY`, using the configured numerical interpreter:

```sh
BINARY_NUMERICAL_PYTHON=/absolute/numerical/bin/python /absolute/numerical/bin/python -B -m benchmarks --frozen campaign /absolute/SPEC /absolute/OUTPUT
```

Preserve incomplete evaluation blocks; their failure policy governs continuation. Per-run
retention bounds states/exports; global retirement preserves recovery/selected states
with receipts after two later segments.

## Qualification and closeout

`native-build` uses pinned source/headers/compiler, without old objects; helpers rebuild
rendering/tokenization/capture/verifier tools. Source includes the measured sampled-EOG
repair. An optional `source_edits` list names edit sets in `benchmarks/runtime_edits.py`;
each edit must match the pinned source exactly once, and `runtime_repairs` records them. Trace builds are diagnostic, excluded from timing. `verify-greedy` independently
checks target argmax, prefixes, correction/bonus/EOG and counters. `verify-sampler` tests
both actual native algorithms on probabilities, zero support and EOG. CPU distributions
and neural GPU requests are separate evidence. `reduce-trace` only checks supplied structure.

`profile-native` owns/verifies/stops/reaps Nsight and checks expected/forbidden kernels and
layer counts. Integer/packing activation needs positive/negative controls; profiler time
is excluded. Follow [results/retention](results.md) and the pinned
[close-results procedure](../skills/close-results/SKILL.md). Extract and verify compact
measurements, figure inputs, failure denominators and executed-source identities before
retiring obsolete raw outputs. Update FINDINGS; record artifact disposition and lost
replay/resume. Reusable references keep the request/source records their authenticated
readers require. Current data/model inputs and recovery/selected states retain their
dependency closure. An ignored path or finished run does not establish disposal readiness.

[Measured qualification](../results/reusable-acceptance-checks/result.md) covers complete
training pauses, exact recovery, both target precisions and reused-reference comparisons.


## Kernel benchmark

`capture-kernel-activations SPEC OUTPUT` captures the first full six-layer drafter
round from one declared DEV request at each requested slot count. Its specification
has schema `kernel-capture-v1`, authenticated `build` (kernel build), `target`,
`drafter` and `workload` artifacts, a DEV `task` ID, `slots`, `context` and `cap`
(32–256 generated tokens). The helper intercepts the runtime quantizer, saves its
input before quantization, and calls the original runtime function. CUDA graphs are
disabled only for capture; captured request times are not benchmark results. Capture
validates the six gate/down pairs, activation bit widths and tensor sizes. This
adapter currently targets the six-layer S4 geometry. The resulting activation
manifest is an input to the separate timing command.

`build-kernel-benchmark SPEC OUTPUT` builds a standalone CUDA harness against an
existing native runtime. It copies the kernel definitions directly from that
runtime's `w1a1.cu`, records the source and extracted-byte hashes, compiler command,
authored harness identities and linked runtime library identities. It does not edit
or rebuild the runtime. The build specification has schema `kernel-build-v1`,
`runtime` (authenticated native build result), `compiler` (authenticated nvcc), and
`architecture` (for example `120a`). Use the same compiler, architecture and
compatibility headers as the runtime being measured. The one-bit compile preserves
the runtime’s `-use_fast_math --ftz=false` combination and explicit F32 rounding
in the quantizer and affine epilogue. It uses the same source bodies, separately
compiled into the harness; it does not claim the harness is the server binary.

`benchmark-kernels SPEC OUTPUT` compares named variants on the supplied drafter,
Baseline and authenticated DEV activation inputs. It prepares all input files before
taking the project GPU lease, then releases the lease when the measurements finish.
A busy lease is a refusal; retry only after the current owner's release. Never run
compilation or checks during timing. Follow the task's allocation and quiet-period
requirements before building or executing.

The `kernel-benchmark-v1` specification contains authenticated `build` (kernel build
result), `drafter`, `baseline`, and `activations` records; distinct `layers`, `groups` (`gate` for joint gate/up, or `down`), `slots`
(from 2, 4, 8, 16), and `variants`; positive `warmup` and `repeats`; `trials` (3–101);
and `bandwidth_gb_s`. The bandwidth is an explicit assumption for a lower bound,
not a measured DRAM bandwidth. For example use 10000 initial warmups, 2000 repetitions and
nine trials. Each trial also warms up with up to 200 launches; the first trial is
discarded in addition to the initial warmup. Each retained
trial records its individual CUDA-event time, and the report gives median, minimum,
maximum and relative range. One multiply-add is counted as one operation.

The activation manifest has schema `kernel-activations-v1`, role `dev`, and a
`tensors` mapping from names such as `blk.0.ffn_gate.4` or `blk.0.ffn_down.16`
to authenticated, token-major F32 files. Each file contains exactly slots × input
width elements. The capture specification must identify the same drafter bytes as
the measurement. Retain how these inputs were captured and their source identities.
Gate and up share the same input. The runtime's quantizer applies the export's real
input gains, clipping mode and clip value. Quantization is outside one-bit kernel
timing; scales, row midpoints and the complete F32 affine epilogue are inside it.

The available variants are `reference` (joint gate/up or dispatched down DP4A),
`cross` (runtime cross-slot DP4A), `mma` (runtime integer tensor cores),
`mma-broadcast` (two sign words loaded per four-lane group and broadcast before
expansion), `mma-warp` (one 16-row warp per block to increase independent blocks),
and `baseline` (Baseline's ordinary runtime graph). MMA tiles sixteen slots
as two eight-slot launches. Gate/up alternatives use two launches, charged together.
Every one-bit trial compares all final output bytes with the dispatched reference,
including signed zeros. A mismatch remains a failed result. The Baseline uses different
weights and is checked against its own ordinary runtime output; cross-model byte
equality is not claimed. Gate and up Baseline results are reported separately and summed
for comparison with the joint one-bit operation. Baseline timing includes its ordinary
activation quantization, unlike the prepacked one-bit kernel timing.

CUDA graphs replay the same launch path for each variant. The Baseline graph is
created by the linked runtime backend with the maintained decode-consistency setting;
its replay bypasses repeated CPU graph construction. This is an isolated hot-weight
kernel measurement, not a full-forward speed claim. Effective weight GB/s divides
stored weight bytes by time and does not establish actual off-chip traffic. Model
quality, attention, vocabulary scoring and server overhead remain outside this test.

Add a variant in `native/kernel-variants.cuh`, dispatch it in
`native/kernel-benchmark.cu`, and register its name in
`benchmarks/kernel_benchmark.py`. New matrix families can reuse the authenticated
input manifest, sample parser, timing and identity records.
`prepare_feed_forward` is the current preparation adapter; a new family supplies
prepared commands and metric metadata to the same timing loop, plus its native
launch and ordinary-runtime reference. Do not weaken byte equality
for an alternative arithmetic path.

Artifact disposition: retain build receipts, compiler/source identities, specifications,
activation provenance, raw timing samples and logs under `artifacts/benchmarks/`.
Kernel helper builds default to `artifacts/native/`.
Extracted tensor copies and binaries are reproducible scratch; original models,
exports and input captures remain read-only retained inputs.

The Q4_K_M Baseline is mixed precision: its layer 2 and layer 5 down matrices are
Q6_K. The adapter reads the stored Q4_K or Q6_K type and records `weight_format` and
actual payload bytes for each measurement. It does not requantize the Baseline to
make every matrix four-bit. Aggregate down comparisons therefore use the original
mixed-precision Baseline; individual records distinguish the two formats.

The separately named experimental edit `kernel_gate_mma_warp_v1` can be appended
to runtime-v7's edit list for `native-build`. It uses the qualified one-warp MMA
body for 9216 × 2560 joint gate/up operations at 4, 8 and 16 slots with row
midpoints present. Other shapes and slot counts retain the original dispatch.
This edit is opt-in; it is not part of the default benchmark runtime. Its source
parity test ties the embedded kernel to the standalone variant. Qualify generated
token signatures and proposed/accepted counts against clean runtime-v7 on DEV
requests before interpreting server timing. Existing calibration measurement and
round-profile counter readers provide clean per-round drafter timing without a
profiling runtime or tracing overhead.

Example invocation sequence, with the authenticated specifications described above:

```sh
python -m benchmarks build-kernel-benchmark kernel-build-spec.json artifacts/native/kernel-benchmark-v7
python -m benchmarks capture-kernel-activations kernel-capture-spec.json artifacts/benchmarks/kernel-inputs
python -m benchmarks benchmark-kernels kernel-timing-spec.json artifacts/benchmarks/kernel-times
```

The reported weight-read floor assumes one cold read of the weight payload and row
metadata at the supplied bandwidth. It is not a physical lower bound for these
cache-resident repeats: a Baseline matrix may finish faster than its cold-read floor.
Report this distinction alongside effective GB/s, and retain outlier samples when
stating the timing spread.

The RTX 5080 qualification inputs, all per-layer samples, byte checks, repeat checks
and served experiment are retained under `artifacts/benchmarks/w1-kernels/`.
`all-spec.json` reproduces the full kernel grid into a new output, and `report.md`
explains its scope and artifact disposition. A separate activation trace verifies
that the experimental runtime executes twelve gate/up MMA launches and six down
DP4A launches per full six-layer round; trace times are excluded from clean timing.

### RSR-core binary variant

An optional `rsr` field in the kernel build specification supplies authenticated
`source` (`kernels/bit_1/cuda/rsr_v5_9.cu`), `licence`, and the reviewed `commit`
`99bcccded70416839f1ef1fc6d58726c507d1fb9`. Source and MIT notice hashes are pinned.
The builder copies the device definitions literally, preserves `RSR-LICENSE`, and
uses the public rectangular wrapper's 256-thread (k ≤ 4) or 512-thread policy.
It does not import upstream Python, run its installer, or use its Torch JIT builds.

`prepare-rsr SPEC OUTPUT` is an offline CPU command. Its `rsr-index-v1` specification
contains authenticated `build` and `drafter`, distinct `layers`, and `blocks` chosen
from 4, 8, 12. A separate native executable with no CUDA dependency constructs the
public uint16 permutation / int4 group / int32 block-metadata layout. Columns within
each pattern group remain sorted. Output indexes authenticate their source drafter,
geometry, padded rows, resident index bytes and every generated file.

Add the authenticated preparation result as `rsr_indexes` to the kernel timing
specification and select `rsr-4`, `rsr-8` or `rsr-12`. Groups now also accept
`gate-only` and `up` for individual calls; `gate` retains the real fused gate/up
reference. Slot count 1 uses the first unchanged slot from the captured two-slot
DEV input, recorded as `activation_prefix_slots`. Other counts use their own captures.

The binary kernel computes Bq for B=(W+1)/2. The adapter converts the exact integer
codes to F32, computes their integer sum once outside timing, and applies
Wq=2Bq−sum(q). The signed conversion and original affine epilogue are timed.
The public kernel has one input vector per launch, so it is repeated per slot.
Integer equality is checked separately against the original dispatched DP4A body
with unit affine parameters; final F32 byte equality is still required. These
integer-code sums stay below F32's exact integer limit for the admitted S4 shapes.
A false integer check is retained even if the affine output happens to match.

Results report `rsr_index_bytes`, `rsr_bits_per_weight`, `packed_weight_bytes`,
`rsr_block` and `rsr_groups`. Effective weight GB/s and the assumed read floor use
the resident index plus row scales/midpoints, rather than pretending the kernel
reads only packed signs. Preprocessing and index upload are outside timing. Blocks
4/8/12 expose the trade-off between permutation size and the growing number of
pattern groups; no claim that one block size is universally optimal is made.

The S4 GPU qualification covers all six layers, individual gate/up/down and joint
gate/up, at 1/2/4/8 slots with all three RSR blocks. Retained specifications, indexes,
raw samples, exact-check flags and a reversed-order four-slot repeat are under
`artifacts/benchmarks/rsr-kernels`; its `report.md` explains timing boundaries,
index sizes and noise. All 312 RSR cases passed integer and final-byte checks.
This qualifies the measured adapter, not a served-runtime replacement.

### Recovered split-reduction tensor core variant

`imma-expanded` adapts the November 4 `imma_expanded<4,...>` implementation from
the archived `residual30-native03/ggml/src/ggml-cuda/w1a8_algorithms.cuh` snapshot
(SHA-256 `4472b75a9b82c95040681a30a014823316dfc9f5e9625201ec06611971ef2a6e`).
The archive receipt and original header are retained under
`artifacts/benchmarks/imma-recovery`. The original seven-slot bounds become the
requested count (up to eight per launch); the output uses the current runtime's
exact affine epilogue, including midpoints. Four warps split the reduction over
the same 16 rows, with vector loads of four sign words and a shared integer sum.
Columns must be divisible by 128. Longer rows use the original unsigned binary
weight form and subtract exact activation-code sums, prepared outside the matmul
timer as in the archived caller. `imma-signed` keeps signed operands for every
shape; `imma-split1` tests one warp with the same vector loading and unrolling.
The latter two variants distinguish reduction parallelism from loop/load changes.

S4 qualification covers every layer and gate/up/down group at 2/4/8 slots, with
400 exact-output cases including the pilot. The separate `runtime-v7-kernel-imma`
build applies `kernel_recovered_imma_v1`; its signed kernel body is checked against
the benchmark source by a CPU test. Clean-clock served checks at depths 1/3/7
retain identical tokens and proposal/acceptance counts across three DEV requests
and three trials per runtime. Evidence and the recovered-source adaptation diff
are retained under `artifacts/benchmarks/imma-recovery` in the main checkout. The owner
adopted the kernel on 2026-11-14; runtime-v8 carries it ([current protocol](#current-protocol)).

The relocation receipt at `artifacts/receipts/kernel-benchmark-relocation-01` records
file identities, preserved original summaries, and 101 successful kernel replay
cases from the main checkout after verified worktree-copy removal. A relocated
experimental-server replay also reproduces tokens and proposal/acceptance counts.
`replay-plan.json` names the retained branch source snapshot and exact commands,
so replay does not require a surviving worktree or changes to main's source.

## Agentic coding panels

The [agentic definition](../benchmarks/definitions/agentic-coding-v1.json) freezes
six DEV projects: cJSON and Parson (C), inih and rapidcsv (C++), and humanize and
Tomli (Python). Each contributes ten tasks, for sixty total; two fixed tasks per
project form the twelve-task routine slice. The existing four cohorts and their
references retain their definitions. These new panels supplement them.

| Panel | Role and units | Policy |
| --- | --- | --- |
| `agentic-slice` | Level 2, 12 DEV tasks | One greedy pass, two tasks per project |
| `agentic-suite` | Level 3, 60 DEV tasks | One greedy pass; project-group bootstrap |
| `agentic-confirmation` | 15 reserved tasks on Boltons | Frozen, unexecuted; separate one-time confirmation allocation required |

The suite contains bug fixes, small API additions, refactors and test-writing.
Every DEV task has initial and follow-up acceptance checks; preparation requires
broken controls to fail and both known repairs to pass. Regression-test tasks
also run the submitted test against a protected faulty implementation, so an
empty or always-passing test cannot satisfy the check. These are authored,
controlled changes in real libraries, not naturally occurring issue-resolution
traffic. Source excerpts provide relevant longer context; the smallest project
has less context. No padding manufactures length. Task success and failures stay
in the denominator, independently of exact drafter/target trajectory agreement.

New agentic tasks stop at the first assistant content exactly equal to its
immediately preceding assistant content, including tool-call text. This counts
as a failed task even if the final workspace passes. The detection call is retained
in raw responses and native reconciliation, but its tokens, draft proposals,
acceptances and duration are excluded from primary measurements. `loop` records
the excluded call; `raw_wall_seconds` preserves its actual cost. The `by_completion`
summary reports completed and failed tasks separately alongside pooled results.
Existing `live` tasks retain their original stopping and reference protocol.

Difficulty calibration uses only an explicit `calibration: "target-only"`, DEV,
`pilot` specification with the sole arm `target`. Record each revision and its
target-only completion rate; aim for at least about 60% completion while retaining
real coding work. Freeze membership, prompts and checks before comparing drafters.
If that rate cannot be reached without trivial tasks, report the failed calibration
instead of establishing routine references. Calibration panels are not references.
Agentic fixtures normalize random GCC temporary object names in returned diagnostic
text with `stable-compiler-paths-v1`; raw test logs retain the original text.
The `no-python-bytecode-v1` fixture environment suppresses bytecode in child Python
processes. New agentic workspace signatures hash every regular file's raw bytes,
including any binary outputs; legacy live signatures retain their text encoding.
The sandbox process limit counts every process and thread of the user, including the
desktop and concurrent jobs; it is 4096, because 1024 was reachable on the native Linux host.

Task and live-result records carry plain `scenario: "agentic-coding"`, `project`
(the `owner/repository` identity), `style` (`minimal`, `coding`, `tools`) and
`task_kind` fields. `group` is `project:owner/repository`, so related tasks stay
together in paired bootstraps. The suite has twenty tasks per instruction style.
Style is varied across tasks rather than counting the same task three times.
This provides coverage, not a paired estimate of the prefix's causal effect.
There are six independent project groups, not sixty independent repositories;
intervals do not estimate run-to-run timing noise. Close speed calls require
fresh matched repeats. The separate prefix cohort remains available unchanged.

`prepare-project-workflows` schema 2 accepts `role`, authenticated `projects`
(maps of project names to relative source paths and file references), `recipes`
and `provenance`. Recipes use per-file `mutations`, `initial_changes` and
`followup_changes`, protected Python check scripts, prompts and metadata.
Checks may invoke the sandbox's C/C++ compilers. Compiler outputs belong under
`/tmp`, never the final project file tree. For `confirmation`, preparation freezes
recipes and records access but executes no controls; lack of qualification is
explicit in its result. `prepare-panels` schema 2 takes authenticated
`live_definitions` and preserves role and per-task style without creating a
separate request harness. `prepare-checks` schema 2 takes an authenticated full
`live` comparison and exactly twelve `slice_ids`; it preserves task definitions
and writes the suite and slice establishment specifications. All execute through
`live` and use `native-live-v3` reference semantics. Projection of slice references
from a complete valid suite is supported by `record-reference`.

Inputs, controls, source snapshots, provenance and runs live in the main checkout's
`/home/philip/BitDraft/artifacts/benchmarks/agentic-coding-01/`. The source-role
check resolves all 640 + 615 TRAIN session tasks and 750 bank CommitPack rows
back to full source alias sets. Current-name lookup spans 46,412 raw aliases;
reachable-history matching additionally identifies the historical Humanize name
and three Boltons forks in unused raw rows. Their transitively expanded repository
families and upstream commits have zero selected TRAIN matches. The raw source
as a whole is not disjoint; the admitted TRAIN selections are.
All 750 selected
Magicoder seeds are authenticated against their retained seed hashes and prompt
bodies; seeds, problems and solutions plus all 6,000 TRAIN bank requests and
the 750 CommitPack rows' old/new source files are scanned against upstream project
code. The 6,000-row bank contains 750 CommitPack rows, not 6,000 CommitPack rows.
The scan uses
13-token shingles, at least twelve shared shingles and 35% containment in the
smaller set, plus exact matches of at least 25 tokens. No selected project hits.
First-commit dates are recorded; all projects predate the conservative source
cutoff, so dates establish no additional exclusion. Magicoder has no repository
names: absence of these matches is not proof of repository or semantic disjointness.
The target's public pretraining exposure is unknown. No protected evaluation data
is used by this curation.

`context_tokens` records each live call's rendered context size.
`arm_elapsed_seconds` includes loading and warmups, excluding shared panel preflight;
summary `elapsed_seconds`
sums assigned-task wall times including tools. Slice cost projected from suite
rows excludes an independent slice load/warmup. Qualified revision five completes
37/60 tasks on each format and all three arms; the slice completes 5/12 Q4KM and
7/12 NVFP4. All 240 drafter tasks exactly match target-only outputs and workspace
fingerprints. See the [result](../results/agentic-coding-suite/result.md) for
intervals, calibration history and completed/failed strata.

| Format / arm | Full measured arm, minutes | Slice projected raw task cost, minutes |
| --- | ---: | ---: |
| Q4KM / target-only | 12.30 | 2.64 |
| Q4KM / Baseline | 9.28 | 2.02 |
| Q4KM / S4 final | 9.34 | 2.00 |
| NVFP4 / target-only | 13.17 | 3.03 |
| NVFP4 / Baseline | 9.29 | 2.10 |
| NVFP4 / S4 final | 9.22 | 2.09 |

Full-arm costs include startup, warmup and bookkeeping, excluding shared panel
preflight. Slice costs include actual loop-detection cost but are projections
from full evidence, without independent startup/warmup; no standalone slice
runtime was measured. Warmup and remaining arm overhead are recorded separately
in the result. The intended roughly three-minute slice and ten-to-fifteen-minute
suite budgets are met by the measured drafter arms within these timing scopes.

Twelve saved records cover target-only, Baseline and S4 final export `405e326b…`
for both panels and formats. Candidate-only specifications are
`/home/philip/BitDraft/artifacts/benchmarks/agentic-coding-01/candidate-specs/{q4km,nvfp4}-v5-agentic-{slice,suite}.json`.
Copy a specification and replace its candidate model reference for a new state;
keep the three saved controls. Execute it with the maintained `live SPEC OUTPUT`
entry, under the GPU lease and using a fresh absolute main-artifact output path.
The [definition](../benchmarks/definitions/agentic-coding-v1.json) authenticates
these specifications. Existing registry records were not overwritten.

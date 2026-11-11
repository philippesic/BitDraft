# Benchmarks

Use **short** during training and **full** for candidate comparisons. Loss/top1 diagnoses
learning; native runs measure acceptance/speed. Request scores and complete application
tasks remain separate evidence.

```sh
uv run --locked --no-sync --offline python -m benchmarks COMMAND INPUT [OUTPUT]
```

| Commands | Purpose |
| --- | --- |
| `plan`, `verify-inputs` | Native comparison units/warmups; optional model/runtime hashing |
| `native-build`, `native-helpers` | Authenticated runtime/helper builds |
| `curate`, `admit-bank`, `select-supplement` | Role-safe prompt preparation/selection |
| `prepare-project-workflows`, `prepare-replays`, `prepare-panels` | Project controls, measured histories, live/harness inputs |
| `capture`, `evaluate` | Teacher features; matched teacher-forced DEV loss |
| `compare`, `live` | Isolated native requests; executing tool episodes |
| `campaign`, `campaign-report` | Training/evaluation; read-only health/learning/exposure |
| `suite` | Full panels and optional state selection |
| `verify-greedy`, `verify-sampler`, `profile-native` | Independent decisions, distributions, kernel activation |
| `reduce-trace` | Structural round accounting, not an independent verifier |

Execution commands freeze source and use [the numerical environment](development.md)
where needed. Preserve inputs, logs, ownership and failures. Comparison outputs must be
new; capture resumes verified records. Resume campaigns/suites with their recorded source
and specification. Checkout edits do not alter active runs. CPU preflight is not runtime
qualification. Run [local checks](development.md); there is no hosted CI.

## Workloads

| DEV panel | Units per arm/target | Execution |
| --- | ---: | --- |
| Short | 120 requests | Zero/periodic/final training checks |
| Full | 600 requests | Two blocks, balanced positions across task groups |
| Harness | 30 groups ×3 prefixes =90 requests | Two balanced blocks |
| Live | 12 tasks, two per family | Three rounds balance arm positions |
| Sampled | 120 requests ×3 seeds | Separate qualified robustness mode |

Initial targets: Qwen3.5 4B Q4KM/NVFP4. Arms: target-only, released DFlash Q4KM, candidate.
Three arms/two targets cost7200 full requests,1080 harness requests,216 live episodes,
plus load/warmup. Sampled adds2160 requests. Smaller pilots are explicit; more candidates
add complete cells. [The bank definition](../benchmarks/definitions/prompt-bank-v1.json)
owns exact coverage/sources/identities. Short is nested DEV; category/length/group
marginals are checked. Keep related variants in one role/group.

Current cohorts are DEV and repeated use is selection exposure. New600-request/24-live/
90-harness confirmation needs independent curation, frozen finalists/protocol and access
receipts; a role label does not supply unseen data. Never consume heldout24 aliases.
Public dialogue is partly synthetic; prior exposure is unknown. Planning dominates
non-arithmetic reasoning; formal logic/independent projects remain thin. Length is not
difficulty, the mixture is not measured traffic, and one-project trials do not establish
universal transfer. Workflow regressions cannot be rescued by request scores.

## Prompts and captures

`curate` authenticates pinned sources and native token counts. `admit-bank` rechecks
selected fingerprints and known/opaque exclusions. Labels follow requests rather than
incidental context words. Tool stages are explicit; translations name a destination.
Optional `development_replays` retains eight first-file-read histories from measured
project trials, including failures, once per source group. Actual tools/messages and
transcript identities remain attached. Optional `harnesses` adds the intended `tools`
prefix to100 TRAIN code-edit and100 TRAIN tool groups. Preserve their ancestry/roles;
the alternate `coding` prefix stays outside TRAIN.

Selection keeps category/subtype/length marginals and required examples. It can choose
shorter complete records within cells to fit399 decimal GB of maximum raw FP16 features,
leaving about1GB for metadata under400GB. FP32 raw features are twice that. No padding/
truncation manufactures coverage. Exact inputs/caps set the maximum, EOS the actual size.
These are dataset planning ceilings, not global disk limits. [Data](data.md) owns formats,
partial tails, mixtures and ancestry.

[`capture`](../benchmarks/capture.py) binds teacher/runtime/tokenizer, bank/IDs, dtype,
context, byte ceiling, exclusions and supervision policy. Every new record passes the
bounded product loader before publication: hashes, geometry, finite values, token/label
arrays and EOG. Validation receipts and partial attempts stay with external raw data.
[FP16 adoption](../results/fp16-storage-adoption/result.md) qualifies HE4/Q4KM storage;
training materializes FP32. It does not establish lifetime/other-family equivalence.

`plan COMPARISON.json` reports native units/warmups; `verify-inputs` hashes models/runtime.
The older schema1 [metadata example](../tests/fixtures/benchmark-plan.json) remains supported.
Training CPU preflight is in [execution](execution.md). Neither grants authority.

## Native comparison

[`native-comparison-v1`](../benchmarks/comparison.py) binds build/server/libs, target,
arms, workload, repeats/warmups, context/depth, role and tier. References use `path`,
`sha256`, `bytes`, relative to the specification or absolute. Arm1 is `target`; keep the
external baseline and candidates distinct. Server/libs must match their build.

Greedy arms must match target token IDs, text and stopping; actual returned generation
settings are checked. Accepted/proposed counters reconcile with independent native logs;
fallback-only arms are refused. Correction/bonus tokens are not accepted drafts. Zero
proposals make acceptance unavailable.

Request time covers pretokenized HTTP completion including prefill/decode, excluding
rendering/tokenization/load/warmup. Live time includes harness/tools. Models are loaded
and warmed, thinking disabled, KV F16 and draft policy fixed. Independent requests/tasks
start with empty caches; live sessions reuse prefixes after their first call. `gpu.txt`,
commands, loaded libraries and ownership receipts identify execution. The lease preserves
other tasks' jobs.

Reports use total tokens/total time, acceptance totals, EOS/cap counts and category/length/
harness breakdowns. Paired group bootstraps preserve category composition and keep repeats/
variants together. Sparse strata make intervals unavailable; multiple candidates also
receive adjusted intervals. Wide intervals are inconclusive. Compare matched arms within
targets, without pooling away regressions or mixing absolute scores across changed eras.
Sampled mode declares temperature/top-p/top-k/min-p, three seeds and target/runtime-bound
qualification. Unequal paths get descriptive robustness, not greedy equality or checkpoint
ranking by throughput.

## Live application and instructions

`prepare-project-workflows` freezes source/recipes and verifies broken, repaired and
followup controls in real project modules. `prepare-panels` creates live inputs and30
fixed-history groups under `minimal`, `coding`, `tools`. Record instruction origins;
identify authored alternatives. Treatments replace system messages, retaining task facts.

`live` executes list/read/replace/write/test tools in isolated copies. Bubblewrap protects
checks and excludes network, credentials and host source. Reads support ranges; edits
require unique matches; truncation is explicit. Reset each task; count both initial and
followup phases. Limits:16 turns/900seconds per task,1024 tokens per call. Retain warmups
outside primary totals and reconcile their counters. All failures/timeouts stay denominated.
Report completion, whole-task/model/tool/overhead time and greedy trajectory/filesystem
agreement, including shared failures. Faster incomplete work is not a win. Deployment
claims need trials in the intended application/configuration.

## Training and selection

[`training-campaign-v1`](../benchmarks/campaign.py) binds intent, short comparison, initial
drafter, optimizer-time cadence and failure policy. Validate TRAIN/DEV before zero eval;
unload training for boundary/final native checks. Restore Adam/RNG/cursor. Resume requires
the same recipe/dataset; changes require explicit continuation. Potential leaders face
the prior leader in the same block; ties retain earlier states. Invalid/partial evals
never select. Explicit pilot mode does not qualify full coverage.

Latest recovery, provisional fast leader and full-DEV incumbent are distinct.
[`suite`](../benchmarks/suite.py) prevalidates matching runtime/settings/role/target/arms.
`completed_panels` can reuse complete matching specifications, never partial blocks.
`candidate_bindings` references state/drafter/execution journals; workers must bind state
to the evaluated export. Highest full-request score selects the experimental incumbent,
with earlier-update ties. Workflow/harness evidence stays separate; no deployment promotion.

Diagnostics preserve RNG and report category loss/top1 on120 fixed role-safe DEV windows.
They are not acceptance. Lengthen optimizer intervals toward10% native overhead rather
than shrinking panels. Keep model/data/evaluation/checkpoint clocks separate.
`campaign-report OUTPUT` needs no GPU: liveness, pointers, journals, learning/native scores,
costs/exposure. Attempts include retries; unique positions use actual label widths;
lifetime history stays unknown. Inspect ownership/failures before restarting. After code
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
repair. Trace builds are diagnostic, excluded from timing. `verify-greedy` independently
checks target argmax, prefixes, correction/bonus/EOG and counters. `verify-sampler` tests
both actual native algorithms on probabilities, zero support and EOG. CPU distributions
and neural GPU requests are separate evidence. `reduce-trace` only checks supplied structure.

`profile-native` owns/verifies/stops/reaps Nsight and checks expected/forbidden kernels and
layer counts. Integer/packing activation needs positive/negative controls; profiler time
is excluded. Follow [results/retention](results.md) and
[close-results](../skills/close-results/SKILL.md): compact measurements with conclusions,
raw execution externally, unfavorable outcomes intact, full local checks before commits.
Ignored files are not automatically disposable.

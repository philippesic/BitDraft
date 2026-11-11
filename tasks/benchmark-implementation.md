# Training, evaluation and capture implementation

## Goal

Make candidate development answer whether a drafter improves complete interactive
requests over released DFlash Q4KM, preserving the selected target's behavior.
Implement one supervised training/intermittent-evaluation path and one comparison
runner with development and confirmation datasets. Reuse the prompt bank across
compatible targets without confusing prompts, teacher captures or model identities.
This is a proposed implementation plan, not an implemented capability or run receipt.

## Authorized

Philip authorized implementation and validation of this plan, including bounded GPU
experiments on the 2080 Ti. Preserve the active RTX 5080 training job. No production
prompt/capture batch is to be generated now. Bounded reuse/transfer of existing captures
and test setup are within this assignment. Reassess the 6K design before future production
acquisition. No hosted CI, new remote or historical artifact deletion.

## Done when

The implementation is complete only when the phased acceptance criteria below pass,
including a real capture-to-resume-to-native-comparison path. CPU tests alone do not
close this task. Current documentation, command registry, agent navigation and
result closeout must describe the implemented behavior; missing capabilities stay
explicit. Delete this task when implementation and scoped closeout are complete.
The current planning deliverable is complete after research, Sol review, corrections
and local validation. Its completion does not authorize the implementation phases.

## State

Owner: root. Implementation in progress. Implemented read-only benchmark plan/input
verification and structural acceptance trace reduction, explicit F16 feature loading into
F32 tensors, and a bounded precision-screen command. Current docs and command registry
are linked. CPU tests exercise mismatch/leakage/accounting; numerical tests exercise F16
round-trip loading and corruption refusal. On the idle 2080 Ti, screened 24 real arrays
from three short existing captures and the initializer fusion projection: no overflow;
training/native suitability remains unresolved. See [result](../results/fp16-feature-screen/result.md).
Torch 2.11.0+cu128 on that host supports sm_75; default Python has no Torch. The GPU is
idle again. A later full-model matched storage attempt was refused first for a Q6_K target
embedding used as a Q4_K head, then (with the retained Q4_K head selected) for the loader
host-memory budget on the 16 GB WSL host. No updates ran and no guard was relaxed.
Compatible modern target/initializer/head artifacts are now present in the isolated
remote qualification directory. Target identity must remain separate from head identity. Qualification raw source/inputs live outside the repo on the 2080 Ti.
No production prompts/captures were generated. Full native runner, intermittent evaluation,
live tasks, harness sensitivity and end-to-end GPU training remain unimplemented.
 Existing code provides
CPU training, authenticated inputs, transactional schema-3 state, resume/branch and
GGUF export/readback. GPU training, fresh capture, a source-complete native runtime,
serving qualification and benchmark selection remain unqualified or absent. See
[support](../docs/support.md), [architecture](../docs/architecture.md),
[data](../docs/data.md) and [domain](../docs/domain.md).

### Research basis and what it does not establish

- [Spec-Bench](https://sites.google.com/view/spec-bench) tests conversation,
  translation, summarization, QA, math and retrieval-grounded work. Adopt breadth,
  not its exact distributions or an implied claim of Spec-Bench compliance.
- [HELM](https://arxiv.org/abs/2211.09110) motivates explicit scenarios and multiple
  metrics. Our coverage mixture is a declared research workload, not measured user traffic.
- [HELMET](https://arxiv.org/abs/2410.02694) finds synthetic recall insufficient as
  a proxy for downstream long-context tasks. Use genuine code/documents/conversations;
  do not pad prompts or turn a needle test into a real-use claim.
- [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) motivates paired,
  clustered uncertainty and pilot-based sample sizing. Neither 120 nor 600 guarantees
  adequate precision; correlated turns and repeats are not independent tasks.
- [EAGLE-3](https://arxiv.org/abs/2503.01840) motivates checking training/inference
  distribution alignment. It does not validate a new objective or architecture here.
- [MLPerf inference](https://github.com/mlcommons/inference/tree/master/docs)
  distinguishes serving scenarios and accuracy from performance qualification.
  Borrow this separation; this project is not an MLPerf submission.

Historical evidence: [data/objective controls](../results/data-and-objective-controls/result.md),
[native replay](../results/native-parity-diagnostics/result.md),
[runtime correctness](../results/verifier-and-runtime-controls/result.md),
[input locality](../results/input-locality-and-clock/result.md), and
[checkpoint selection failure](../results/he4broad-fresh-source-training/result.md).
The old scripts/dflash_baseline/precision_benchmark.py supplies useful paired requests,
activation counters and raw-output auditing; its fixed paths, narrow cohorts and
source-dependent runtime are not a reusable new runner. Old block capture plans bind
source, prompt, tokenizer and feature identities and permit compatible payload sharing.
Their proposed rotating replay cache is not evidence that exact regeneration works.
Recover historical source via [evidence access](../docs/evidence.md); never depend on
an old checkout's environment or objects to qualify the new implementation.

### Scope and common names

Initial targets: Qwen 3.5 4B Q4KM and NVFP4, independently qualified. Baseline means
released compatible DFlash Q4KM, fixed per target; target-only means no drafter.
Candidate means a particular architecture/recipe/checkpoint. Incumbent means the
best supported experimental candidate, never a replacement for the external baseline.
W1A8 describes specified one-bit-weight/eight-bit-activation components, not a unique
model. Historical DSpark W1A8 must retain its family and original target identity.
Update docs/domain.md with these definitions during phase 1; preserve old result identities.

Start with HE4's DFlash-derived architecture, batch one, warm loaded models, no
cross-request prefix reuse in isolated-request tests, fixed draft policy and target KV
settings within comparisons. Live workflows use the intended application cache policy;
isolated no-cache results cannot substitute for deployed cache behavior.
Thinking disabled initially, recorded explicitly; no reasoning-mode or concurrent-serving
claim. Additional targets, architectures, dynamic policies and KV changes are separate
experiments, not automatic axes of every run. Runtime-unsupported NVFP4 is blocked,
not silently substituted, emulated or called a negative performance result.

### Workload and data specification

Keep the original 10K bank immutable; selected derivatives retain ancestry and roles.
Use six categories, with these starting counts:

| Category | Training | Fast DEV | Full comparison |
| --- | ---: | ---: | ---: |
| Code generation, editing, debugging, explanation | 1500 | 30 | 150 |
| Arithmetic, logic, constraints, planning | 1000 | 20 | 100 |
| Document QA, multi-passage work, summarization | 1250 | 25 | 125 |
| Conversation follow-ups and corrections | 1000 | 20 | 100 |
| Structured output and tool interactions | 750 | 15 | 75 |
| Writing, rewriting, translation | 500 | 10 | 50 |
| Total | 6000 | 120 | 600 |

Half of code tasks should involve existing code; no more than half of reasoning should
be arithmetic; one fifth of writing should be translation. Primarily English. Treat
these as acquisition targets to audit, not as claims about existing records. Inspect
existing Magicoder/Dolly/GSM8K examples and retain useful coverage; supplement missing
interaction/document tasks. Never synthesize breadth by relabeling examples.

Training input length quotas: 3720 below 512 tokens, 1800 at 512-2047, 420 at 2048-8191,
60 at 8192-16384. Count complete templated input including history with each target's
actual tokenizer. No truncation/padding to manufacture compliance. Quotas are marginals,
not every category crossed with every length. Put long contexts mainly in code,
documents and conversations. Complexity targets: 40% direct, 40% compositional,
20% context-dependent, with human-readable tagging criteria; length is not difficulty.

Evaluation deliberately keeps broader length coverage: fast 60/36/18/6 and full
300/180/90/30 across those same bands. Report long-context outcomes separately, with
limited training coverage stated. One unit is one independently grouped task; initially
one selected continuation per conversation in the request suite, with fixed authentic
history. The separate live portion of the larger comparison executes complete tasks. Related turns, source documents, repository tasks and paraphrases stay
in one split across request, replay, live and harness panels. Request-only results do
not establish complete autonomous task success.

Desired response mix: 30% short, 50% ordinary, 20% extended; caps 256/1024/2048 respectively.
Natural stopping stays enabled. Record realized lengths, EOS and cap hits; caps are
not expected lengths. Truncation is distinct from a valid completed task. Freeze
workload, generation settings and cap handling before candidate measurement.

Before acquisition: reassess 6K adequacy using existing learning/exposure records and
a small target-side cost pilot when separately authorized. Review 20 examples/category
across length/complexity in TRAIN/DEV; automatically validate all records. Have multiple sources for
major categories where feasible and include source-disjoint evaluation coverage. Audit
known historical exposure; unknown exposure cannot be relabeled fresh. No opaque
heldout24 inspection, training, selection or role conversion.

### Storage and capture precision

User planning ceilings are 400 GB FP16 / 800 GB FP32 per target for capture payload,
not global disk limits. Decimal GB differs from GiB. At band midpoint inputs (mean 1024)
and mean 512 generated tokens, eight width-2560 taps cost 377.487 GB (351.563 GiB)
FP16 or 754.975 GB (703.125 GiB) FP32 for 6K tasks. Both targets double that; weights,
checkpoints, staging, backups and full logits are additional. Actual selections and
output lengths determine actual bytes; do not treat midpoint estimates as admission.
The 30/50/20 response-cap mixture permits mean 998.4 generated tokens. Even at mean
1024 input tokens that reaches 497.025 GB FP16 / 994.050 GB FP32, over the ceilings.
Using input-band upper bounds would raise that further, to approximately 655 GB /
1310 GB. Thus the current quotas/caps and budget are not yet jointly qualified. Before acquisition,
resolve actual lengths and either propose lower capture caps (distinct from eval caps),
a smaller selected capture subset, or staged residency with explicit total-storage costs.
Changing the user-selected 6K scope or ceilings requires agreement; never silently cut it.

Keep compact prompts, tokens, split lineage and manifests. Store features once per
compatible target capture, outside Git; overlapping windows reference them rather than
copying them. Use hard-label payload for the existing objective; full distributions are
required only for an explicitly selected soft-target objective. Capture complete context
needed by the recipe; any context/window reduction is a scientific change, not storage cleanup.
A byte forecast uses actual inputs, per-task maximum output and temporary peak allocation.
If it exceeds the ceiling, propose a revised selection or staged cache; do not silently
shorten tasks. Shards may live in archival storage with verified local staging. Recompute
is not a substitute for backup until exact replay and its cost are demonstrated.

FP16 is a capture-storage experiment, not a switch to FP16 training. Compare F32 values
with F16 round-trip values across categories/lengths/layers, including outliers, loss,
labels, gradients and updates. Then matched short training from the same parent/order,
followed by native acceptance/correctness; a longer paired continuation tests accumulated
effects. Predeclare tolerances and useful noninferiority margins after measuring reference
variation, before observing the treatment results. Do not accept merely nonsignificant
differences. 2080 Ti results qualify their stated scope, not 5080 speed or NVFP4 support.

### Artifact placement and lifecycle

Use a host-configured external artifact root, proposed default sibling directory
/home/philip/binary-eagle-artifacts (not created by this plan). One authoritative
content-addressed capture store owns immutable shards; dataset manifests reference
shards rather than copying them per candidate. Runs own journals, outputs and checkpoint
references; results in Git retain only compact evidence and regeneration inputs. The
config separately identifies the local fast cache and any verified backup destination.
A path called archive or a second directory on one disk is not proof of a backup.
Before acquisition, the operator resolves actual filesystem/mount/free-space identity,
payload forecast, local staging peak, checkpoint peak and backup availability. Report
local-only durability if no independent backup exists; do not invent a remote destination.

Lifecycle: prepare temporary shard -> validate semantics and hashes -> atomically publish
immutable shard/manifest -> materialize a verified local cache -> train with pinned shard
handles -> retain authoritative data while recipes/evidence depend on it. Evict only owned
cache copies whose identical authoritative copy is verified and not in active use. A failed
publication stays identifiable; it cannot replace valid data. Valuable sole copies, selected
states and source dependencies cannot be removed by cache pressure. Original captures stay
unchanged in an FP16 trial. Retention changes require the existing closeout process.

The pilot must measure local shard sequential/random access, target capture cost, training
steps, materialization, checkpoint publication and evaluation switching. Prefer immutable
packed shards with an index over remote per-window small-file reads. Qualify locality-aware
sampling against the frozen category/task/position distribution and exact resume. No
arbitrary fixed cache size until measured; the user ceilings govern retained capture payload,
not just residency. Moving bytes off the training host does not satisfy a total-payload cap.

### Compatibility and sampling contracts

A run binds prompt selection, capture specification, candidate recipe and runtime.
Authenticate target weight bytes/precision, tokenizer/template, generated token path,
feature layer IDs and pre/post-block meaning, width/dtype, label positions/stopping,
objective, draft geometry and candidate vocabulary/embedding provenance. Reject a
mismatch before loading an optimizer. Dataset.target must be compared against an explicit
candidate target requirement; a stored hash alone does not provide that binding today.
Validate finite values and token/feature alignment with a bounded native replay as well
as file hashes. A correctly hashed wrong capture remains wrong.

Reuse target captures for precision/features only when required inputs and supervision
match. Different target precision needs a new capture identity. Another architecture
needs a qualified adapter and may require additional taps or soft labels. Unsupported
combinations fail with the missing capability and repair path; no guessed conversions.

Implement deterministic category -> task -> eligible-position sampling so long answers
do not accidentally set category weights. Record realized task/length/token exposure,
unique positions and repeats. Preserve exact RNG/order/cursor on same-recipe resume;
branches explicitly record changes and inherited exposure. Cover short/EOS tails without
fabricating labels or silently dropping tasks; the current fifteen-label window rule
must be reconciled with these cases before broad data claims. Either retain the fifteen-label contract and report excluded
short tasks, or qualify a new partial-tail mask/objective including loss normalization
and native behavior; a loader-only patch is insufficient. Changing sampling or the
loss mask is a recipe change, requiring qualification against existing behavior.

### Two evaluation tiers: exactly what runs

There are two public benchmark choices, short and full, under the same compare entry.
Loss diagnostics remain separate and are not a third benchmark. Short runs the 120
requests in the table above, with the declared length mix, normally every training hour.
Full runs 600 requests plus 12 live DEV tasks and a 90-request harness panel. Sampled
robustness is an explicitly selected full-comparison mode, not a hidden hourly expense.
Both run the exported candidate through the qualified native serving path, not a training
simulation. Resolve the expanded execution matrix and cost before any run.

Short requests include authentic coding/conversation/tool histories from application
sessions, within the existing category/length quotas, not 120 additional replay requests.
Record their source session and cluster them by session. Each arm generates normally;
never force a recorded continuation to manufacture acceptance. Fixed-history replay
measures those calls only, not the consequences of subsequent changed tool calls.
A target-only reference authenticates output/stopping; immutable reference outputs may
be reused only with identical runtime/target/settings and repeated determinism checks.
Current candidate timing is measured live; an old reference receipt is not a fresh speed
comparison. If the fast check reports relative speed, time both arms in matched blocks.

### Live tasks within the larger comparison

Run 12 development tasks, two each: fix a bug and execute tests; add and verify a small
feature; investigate code/log failures; answer document questions with a follow-up;
complete a structured tool workflow; revise work after feedback. Use actual repository
or document snapshots, explicit requirements and independent completion checks. Choose
feasible tasks for the target, not only tasks the candidate later wins. Freeze exclusions
and reasons before candidate testing. A broken target-only harness must be repaired before
interpreting candidate outcomes. Failure on valid tasks remains a reported outcome.

Each arm starts from an identical isolated filesystem/tool state and actually reads,
edits, calls tools and responds to their outputs. Pin harness source, system instructions,
tool schemas, dependencies, model template, context management, cache policy, sampling
and stopping. Use deterministic local tools and frozen inputs initially; no changing
network services or real external side effects. Fixed scripted follow-ups arrive at a
specified turn/event; no human idle time in the measured task clock. Run arms serially,
reset state between trials, and record warm/cold boundaries and model load separately.
Start each independent task with empty target and drafter request/KV/prefix caches but
warm loaded models; permit normal prefix reuse and context compaction within its session.
Never leak cache state across arms, repeats or independent tasks. If the application
requires prewarmed shared prefixes, qualify that as a separately pinned entry condition.
Never execute generated code on the source checkout or allow benchmark tools to mutate
shared datasets, credentials, host configuration or unrelated jobs.

Report completion checks, failures/timeouts, elapsed task time, model-wait time, tool
execution time, call count, output length, retries and acceptance. The initial local
harness is sequential: model-wait and tool intervals do not overlap; unclassified overhead
is separate. If later work introduces overlap, report interval unions/critical-path time
rather than summing overlapping component clocks. All tasks remain in
the denominator. Do not average successful-task latency while hiding failed arms; show
paired completion outcomes and latency on jointly successful tasks with that scope
explicit, alongside full task totals and deadline outcomes. Predeclare limits per task
family using DEV controls, equally applied to all arms. Unexpected greedy divergence is
a correctness investigation, not an opportunity to claim a shorter/faster solution.
Live sampled runs permit differing paths; assess completion and distributions separately.

Use 24 distinct live confirmation tasks (four per family) when freezing a finalist claim.
Twelve/twenty-four tasks are a deployment transfer check, not broad statistical proof;
uncertainty can require a later larger frozen set. Repeat count/order come from the DEV
pilot and are frozen for confirmation. Keep request and task outcomes separate: no
combined score that averages away a live regression. Passing the request suite alone
cannot earn a general deployment claim; a demonstrated workflow slowdown or reliability
regression blocks that claim or restricts it to explicitly supported workloads. Uncertain
live outcomes are inconclusive, not passed. Numerical practical margins must be frozen
before finalist execution rather than chosen to excuse the observed result.

### Harness sensitivity within the larger comparison

Run 30 DEV tasks under three configurations: minimal assistant instructions, the intended
coding harness instructions/tool schema, and a second realistic harness configuration.
These are exactly one fixed-history response per task/configuration: 90 requests per
arm per target per timing block, not 90 total across models. No tools execute and no
autonomous continuation occurs in this panel; live episodes belong only in the live panel.
Select tasks with meaningful completion criteria under all configurations; minimal mode
still supplies the same task facts and any required tool outputs. Clearly distinguish
textual/structured-response requests from live tool-use episodes. A configuration that
cannot express the task is ineligible, not a drafter acceptance failure.

Freeze task IDs before candidate evaluation; record exact system prompts, tool definitions,
chat template and generation settings. Hold target/runtime/draft policy fixed within
comparisons. Input length changes caused by the harness are part of the treatment and
must be reported; this tests whole harness-configuration sensitivity, not an isolated
causal effect of wording. Each configuration has its own target-only reference. Compare
candidate versus baseline acceptance and request speed within each configuration, then
report how that relative advantage changes. Never compare raw acceptance across different
target output paths as though the workloads were identical. Use task-clustered paired
intervals, keeping each task's variants together; task success/correctness remains required.

Include real system instructions and tool schemas in relevant TRAIN examples. Reserve
one harness configuration from TRAIN to test transfer, while acknowledging that repeated
DEV use is still selection. Final confirmation requires new tasks and an independently
frozen configuration/version not used to tune the candidate; same prompts or edited
versions retain ancestry. Public availability does not make an example exposure-free.
If no independently held-out harness is available, scope the claim to the tested harnesses
instead of promising unseen-harness transfer. Maintain 30 x 3 confirmation requests with
separate IDs/roles if this claim is sought. Performance confined to one configuration is
a harness-specific result, not a generally better drafter.

### Acceptance accounting qualification

Define proposed as the actual draft tokens submitted to verification under the frozen
linear proposal policy; accepted counts only those draft tokens committed as accepted.
Target-generated correction/bonus tokens are emitted output, not accepted draft tokens.
Record proposals, verifier decisions, first rejection, accepted prefix, corrections,
bonus tokens, EOS and unused suffix at each round. Define EOS/cap boundary counting in
one versioned contract and check it against executed runtime behavior before comparing;
never count nonexistent padding slots as proposals. A different tree/proposal algorithm
requires an explicit accounting extension, not reuse of linear counters by name.

An independent reducer reconstructs accepted/proposed, emitted/round and stopping from
traces; test all-accepted, first rejection, middle rejection, EOS, output cap, zero proposal,
and fallback-only paths. Zero denominator is unavailable, never zero or perfect acceptance.
Raw token decisions and server counter totals must agree. Sampled rejection sampling is
separately qualified; token-equality checks alone are not distribution tests. Full traces
run in a diagnostic lane to avoid altering clean timings; verify unchanged deterministic
outputs/counters and disclose any measured instrumentation overhead. The clean timed calls themselves must retain low-overhead activation and fallback
counts, not rely on detailed traces from different calls to prove activation. Validate
counter overhead on DEV and freeze instrumentation. Fallback-only output cannot qualify
the candidate as active speculative decoding; partial fallback is visible in the result
and may not be silently excluded from timing.

### Evaluation protocol and selection

Cheap diagnostic: 120 fixed DEV windows, initially every 250 updates. Loss/top-1 are
not native acceptance. Fast native DEV: 120 tasks at zero updates, each 60 minutes of
optimizer work and completion, initially one pass. Calibrate cadence to about 10% evaluation
overhead; lengthen the interval rather than silently shrinking the suite. The cadence is
agent-proposed, not a resource allocation or a rule to terminate useful training.
Candidate-only native diagnostics cost 120 requests per target/check. A paired candidate/
baseline timing block costs 240; freshly executing target-only adds another 120. Qualify
reference reuse separately. Schedule the two targets independently within their allocations;
no promise of both targets every hour. Do not rank absolute timings across days/changed
load: compare potential leader changes in a contemporaneous matched candidate/prior-leader
block (240 requests) before replacing the provisional leader. External speed claims need
contemporaneous external baseline blocks. Report omitted timing comparisons as unmeasured.

The same comparison runner supports full DEV and sealed confirmation roles. They use
separate 600-task cohorts with the same coverage design; thus comparison and confirmation
are uses of one system, not one repeatedly peeked-at set. Existing 1002 historical DEV
records can supply compatible coverage but cannot provide missing task families or fresh
confirmation. Fast DEV may be a declared subset of full DEV. Evaluation tasks are additional to the
6K TRAIN selection; they do not consume the training capture allocation. Sampled DEV
and sampled confirmation also use separate roles/cohorts. Do not tune on confirmation.
Freeze finalist(s), recipe, endpoints and analysis before opening a new confirmation set;
subsequent adaptation requires fresh confirmation or an explicit exploratory label.
Pilot execution, sample-seed studies and developer example inspection use DEV only.
Confirmation curation uses an isolated curator/access boundary with opaque IDs/group
indexes for developer exclusion checks; no prompt/output inspection by candidate selectors.
Record every confirmation access, including failed runs, and its effect on future claims.
If independent curation cannot be arranged, describe the limited blinding honestly; no
claim of unseen confirmation based solely on a folder name or editable flag.

Start full timing at two complete counterbalanced blocks: 600 tasks x 3 arms x 2 blocks
x 2 targets = 7200 requests. Three arms are target-only, external baseline and one
candidate; each extra candidate adds 2400 isolated requests. Full DEV also adds
12 live tasks x 3 arms x 2 targets x the frozen live repeat count, plus 30 tasks x 3
harness configurations x 3 arms x 2 targets x the harness repeat count. At two repeats
the harness panel adds 1080 requests. Full confirmation substitutes 24 live tasks.
Report these extra costs; 7200 is not the total full-benchmark budget. A pilot on DEV chooses warmups, order and
repeat count, then freezes them before confirmation. All repetitions and related examples
remain together when resampling groups. The pilot fixes arm order at task/server-block
level, reload boundaries, warmups and cache state. Two repeats cannot by themselves
balance all three arm positions; use a balanced order across task groups or more blocks.
Never top up until a desired result appears.

Sampled robustness: separate 120 tasks x 3 seeds x 3 arms x 2 targets = 2160 requests,
initially T=0.7/top-p=0.9, subject to supported target policy. Paired seeds do not prove
identical token streams. Require verifier/distribution tests and stop/schema checks;
report sampled lengths and quality separately rather than calling arbitrary output changes
speed gains. Greedy qualification demands matched token IDs/text/stops and finite logits.
Failure or unsupported sampled execution limits the claim to greedy; it is not passed.
A sampled application claim additionally requires live tasks under that sampled policy
and intended cache behavior, with seeds/repetitions fixed on DEV. The 2160 request count
does not include these episodes. Greedy live evidence cannot qualify sampled workflows.

Primary serving score: total emitted tokens / total complete request seconds on a fixed,
valid, greedy token-matched DEV panel. Sampled token throughput is descriptive only; it
cannot rank checkpoints by rewarding verbosity or different incomplete answers. Preserve current domain selection semantics: highest complete
valid score attached to durable state; ties keep earlier. Fast one-pass score is a
provisional within-run selection, not evidence of external superiority. Keep latest
resumable state, provisional fast-DEV leader and full-DEV request-selected incumbent separately.
Fast DEV alone cannot replace the full-DEV request-selected incumbent. Its eligibility
requires all mandatory full-DEV panels operationally complete with valid measurement
and verifier fidelity; task outcome failures remain reported data, not missing cells.
The name denotes request-score selection, not deployment qualification. Every incumbent
report carries separate live-task and harness claim statuses. Demonstrated workflow or
harness regressions block a broad deployment recommendation; inconclusive transfer
prevents a positive workflow-improvement claim but does not discard a useful checkpoint.
No significance gate is imposed on each two/four-task family. Positive claims name the
represented families, actual harness versions, target, cache and sampling settings;
absence of observed failure is not proof across all possible workflows. Apply complete-score
selection within each declared tier, never compare scores from different panels.
Full DEV comparison confirms promising
checkpoints; confirmation cannot retroactively be used to search all saved checkpoints.

Report full-request latency, TTFT when instrumented, decode speed, memory, accepted/proposed,
accepted tokens/round and prefix survival; separate clean timings from diagnostic traces.
Report category/length results, aggregate request throughput, slow/no-benefit cases and
uncertainty. No mean of request token rates. Show setup/capture/train/export/eval wall
cost separately from optimizer work. Verifier/runtime fidelity failure invalidates
request-state promotion even if fast; task-quality outcomes govern separately scoped
workflow claims. A target answer failure shared by all arms is not a verifier mismatch.
Task answer quality (exact answers, schema checks, sandboxed code tests where suitable)
is distinct from verifier fidelity and target intelligence; bounded checks are not universal
quality proof. No training-model changes to make benchmark answers look better.

Estimate candidate/baseline ratio of total-token/total-time scores and its paired
task-group bootstrap interval separately for each target; pooled cross-target results
are secondary and cannot rescue a failing target. Resample all associated repeats and
turns together within the frozen category design.

A practical planning target is resolving an approximately 5% overall improvement; it is
not yet an approved promotion threshold. Pilot paired group variability determines whether
600 tasks can resolve it. Wide intervals mean inconclusive, not equivalence. Report all
predeclared comparisons; use simultaneous/multiplicity-aware intervals for multiple
finalists. Training-seed variation is separate: one trained seed supports only that
checkpoint, not a general recipe win; recipe claims need independent training replications.

### Entry points, ownership and agent navigation

Proposed public commands (not available yet):

| Entry | Responsibility |
| --- | --- |
| python -m benchmarks plan SPEC | Resolve inputs/capabilities, coverage, costs and blockers; no GPU/acquisition |
| python -m benchmarks qualify SPEC OUTPUT | Bounded authorized numerical/runtime qualification |
| python -m benchmarks train SPEC OUTPUT | Compose supervised training with checkpoint-bound evaluations |
| python -m benchmarks compare SPEC OUTPUT | Same native runner for DEV/confirmation and multiple candidates |
| python -m benchmarks capture SPEC OUTPUT | Authorized target capture and immutable shard publication |
| python -m benchmarks report OUTPUT | Read-only operational/scientific status and reproducible aggregates |

Keep python -m w1a1_eagle preflight/run/report as the lower-level training API.
benchmarks may depend on product interfaces; product must not import benchmarks.
The benchmark coordinator sequences training segments through existing supervision and
checkpoint handles; the numerical worker never shells out to an evaluator. Preserve one
owner of optimizer state, journals and GPU allocation. Plan/report cannot launch jobs.
Use existing machine/config and clock abstractions; do not invent another job manager.
Register actual entry points in policy/commands.toml when implemented.

Small typed immutable specifications define target, capture, recipe, workload, evaluation
and declared capabilities. Use explicit qualified adapters with stable identities and
protocols. A single discovery boundary resolves adapters; no scattered string dispatch.
Review existing reserved interface guards against the concrete design before adding code;
if a guard obstructs legitimate design, report a focused policy change rather than bypassing
it or building an unnecessary plugin framework. Authored benchmark specs are tracked;
raw data and host locators are external/private. No placeholder extension directories.

Every run gets immutable resolved inputs, source including dirty patches, executed binaries,
dependency identities, ownership and append-only attempt records. Status distinguishes
planned, blocked, running, failed, invalid, incomplete and complete; scientific conclusion
separately records improved/no-benefit/inconclusive/untested. Errors name the mismatched
field and legitimate repair, never recommend force flags or editing evidence hashes.
Partial evaluations cannot select a checkpoint. Resume does not repeat already published
updates; interrupted timing blocks are retained and rerun as whole blocks under policy.

Checkpoint before releasing training for native evaluation. Verify ownership/release before
loading the server; restore optimizer/RNG/cursor exactly afterwards. Checkpoint publication
failure stops the transition. Evaluator failure preserves training state and failure records;
the declared policy decides whether to continue training without selection or stop. No
candidate-specific favorable retries or competing GPU processes.

AGENTS.md stays short: link docs/domain.md for names, future docs/benchmarks.md for workflow,
and close-results for closeout. README lists only implemented commands. docs/benchmarks.md
will describe present supported workflow; docs/data.md, execution.md, architecture.md and
support.md own their existing facts. Do not duplicate this plan into current docs.

Use the shared benchmark-design skill for scientific design and the tracked close-results
skill for investigation closeout. Future project-specific run guidance belongs in
docs/benchmarks.md first; add a skill only if it supplies reusable procedural judgment,
not a second runner or duplicate thresholds. Skill invocation grants no GPU allocation.

### Evidence and closeout

One results/<question>/result.md per durable investigation, following
[results specification](../docs/results.md), with compact result.json/figure data and
regeneration recipe when useful. Raw outputs remain external and immutable. Include
all arms/failures, target and precision, dataset roles, exposure, timing definitions,
actual activation, source/runtime identity, uncertainty method and reproduction limits.
A learning curve plots optimizer work AND unique data consumed, with evaluations tied
to durable checkpoints; never join unrelated initialization histories as one curve.

Update FINDINGS only after verified conclusions, including negative/inconclusive findings.
A planning document is not a measured finding. Keep retain/revise/discard/unresolved
separate from run exit status. Cleanup follows existing ownership/retention procedure;
never delete checkpoints, original captures or frozen source because a chart exists.

### Implementation sequence and acceptance criteria

1. Contracts and plan/report: settle terminology, typed specs, candidate-target-capture
   binding, role/lineage checks and byte forecast. CPU rejection tests cover wrong target,
   taps, objective, vocabulary, leaked groups, unavailable adapters and understated sizes.
   A new agent can obtain actionable blockers without launching a GPU or editing policy.
2. One native vertical path: reproducible source-complete build; target-only and released
   baseline; one tiny authentic capture; one candidate update/save/resume/export/native
   comparison using the minimal shared comparison core (not a throwaway runner). Qualify
   Q4KM first, then NVFP4 explicitly. Do not call a metadata fixture
   the vertical path. Preserve current strict checks; add numerical/runtime tests.
   Bring up the intended application harness with target-only and released baseline here,
   before long training. Run a few DEV live tasks, then the first usable candidate through
   the same path. Use actual session lengths/tool behavior to challenge the proposed mix;
   do not postpone the first application trial until the full 6K capture is paid for.
3. FP16 storage study and GPU training: first CPU round-trip tests, then authorized 2080 Ti
   matched experiment if supported, then extended native confirmation. Preserve FP32
   reference. Device support must include optimizer, RNG, checkpoint and export behavior.
4. Workload preparation design and 6K reassessment: audit retained bank and source options,
   freeze selection/grouping/quotas, token lengths, continuation policy and real byte/cost
   estimates. Obtain a new acquisition scope before downloading/generating/capturing data.
5. Harden the same native comparison runner from phase 2: DEV pilot establishes execution cost, noise, fixed
   analysis and stopping rules. Test wrong/fallback model, nonfinite output, stopping
   mismatch, cap hits, missing cells, telemetry interference and malformed counters.
   Synthetic fixtures test mechanics; actual requests qualify runtime and performance.
   Qualify live sandbox reset, task checks, timeouts and model/tool clocks; test harness
   variation, cache behavior and request-to-session lineage. Resolve deployment regressions
   before generalizing an isolated-prompt improvement.
6. Training coordinator: checkpoint-bound intermittent eval, independent latest/provisional/full-DEV request-selected states,
   exact restart across train/eval transitions, realized exposure and learning curves.
   Fault-inject crash during save, server startup/eval and restore; no duplicate updates,
   lost durable state, accidental ownership adoption or false selection.
7. Full investigation qualification: separately authorized representative acquisition,
   frozen finalist comparison and confirmation with complete receipts; exercise closeout,
   compact result regeneration, navigation, retention and honest unsupported statuses.

At each phase run current quick/full local gates, with CPU rule proofs for new guards.
GPU integration is an explicit separately reported lane; never silently skipped as passed,
never required in routine pre-commit, and never hosted CI. Update current docs with working
behavior per phase. No blanket code cleanup or unrelated operations tooling.

### Review

Second independent Sol review and root reconciliation completed. Agreed changes:
fixed-history 90-request harness panel, separate live episodes, explicit per-arm short
and full costs, cold independent-task caches with warm within-session reuse, greedy-only
checkpoint score and sampled live qualification for sampled application claims. Clean
measured calls must prove activation themselves; separate traces do not prove that.

Root challenged treating small live panels as statistical proof or requiring significant
nonregression for every tiny task family. Sol agreed: preserve exactly three checkpoint
roles, name the full-DEV request-selected incumbent precisely, and attach live/harness
claim statuses instead of conflating operational completion with deployment proof.
A useful experimental state may be retained while its workflow advantage is inconclusive.
Both reviewers reject averaging away a workflow regression or calling no detected failure
universal transfer. These are agreed design boundaries, not measured performance claims.

Independent gpt-6.1-sol review completed. It found the plan sound overall and identified
three concrete corrections, now incorporated: show cap-bound storage as well as expected
size, build the shared comparison core in phase 2 rather than a temporary runner, and
separate provisional fast-DEV leaders from full-DEV request-selected incumbents. Also clarified
confirmation quarantine, per-target paired score intervals, actual counterbalancing and
short-tail objective qualification. No review finding required experiments or expansion
of the 6K scope. This is design review, not runtime or statistical qualification.

## Next

Finish phase 1 contracts against real capture metadata, not just benchmark specifications.
Then implement the reproducible native vertical path, candidate-target binding at the
worker, GPU training and shared comparison core. Run matched full-model F32/F16 training
and native evaluation before adopting F16 captures. The copied modern target/initializer and retained Q4_K head resolve input availability,
but full model materialization is refused by the 2080 Ti host-memory guard. A qualified
more memory-efficient loader or sufficient host RAM is required for this full-size test.
Do not increase a declared budget above available RAM or substitute old Qwen3 models. Keep all remaining
phases open and do not represent CPU gates or the rounding screen as completion.

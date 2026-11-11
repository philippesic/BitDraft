# Benchmark tools

The [implementation task](../tasks/benchmark-implementation.md) records current validation
and remaining work. Request comparison, live tools, harness preparation, DEV loss and
training coordination have entry points below. Sampled verification and full campaign
qualification remain separate requirements; implemented commands are not evidence of
their scientific validity.

```sh
uv run --locked --no-sync --offline python -m benchmarks plan SPEC.json
uv run --locked --no-sync --offline python -m benchmarks verify-inputs SPEC.json
uv run --locked --no-sync --offline python -m benchmarks reduce-trace TRACE.json
uv run --locked --no-sync --offline python -m benchmarks compare COMPARISON.json NEW_OUTPUT
uv run --locked --no-sync --offline python -m benchmarks prepare-panels PANELS.json NEW_OUTPUT
uv run --locked --no-sync --offline python -m benchmarks live COMPARISON.json NEW_OUTPUT
uv run --locked --no-sync --offline python -m benchmarks evaluate LOSS.json NEW_OUTPUT
uv run --locked --no-sync --offline python -m benchmarks suite SUITE.json OUTPUT
uv run --locked --no-sync --offline python -m benchmarks campaign CAMPAIGN.json OUTPUT
uv run --locked --no-sync --offline python -m benchmarks verify-greedy ORACLE.json NEW_OUTPUT
uv run --locked --no-sync --offline python -m benchmarks verify-sampler SAMPLER.json NEW_OUTPUT
uv run --locked --no-sync --offline python -m benchmarks select-supplement SELECTION.json NEW_OUTPUT
```

Execution commands freeze source and launch from that snapshot. Numerical commands use
the configured numerical Python. Source edits in the working checkout do not change an
active benchmark. Resume an existing suite/campaign with its recorded source and
specification; changed inputs need a new output.

Try `python -m benchmarks plan tests/fixtures/benchmark-plan.json` for a synthetic
metadata-only example. It intentionally reports missing capture artifacts.

`plan` is read-only. Schema 1 binds candidate/capture target, tokenizer, template, taps,
feature semantics, width and objective. It refuses incompatible identities, related tasks
across roles, duplicate task IDs and invalid sizes. It forecasts maximum training feature
payload from exact input counts and output caps. It reports budget/acquisition blockers;
a successfully printed plan is not runtime admission. `verify-inputs` additionally hashes
the declared artifacts. This authenticates bytes, not tensor semantics or model correctness.

A specification has `schema`, `candidate`, `capture`, `tasks`, `features`, `storage_bytes`
and `capture_limit_bytes`. Candidate and capture each contain `target`, `tokenizer` and
`template` SHA256 values, integer `taps`, positive `hidden`, `semantics` and `objective`.
Tasks contain `id`, `group`, `role` (train/dev/confirmation), `category`
(code/reasoning/documents/conversation/structured/writing), `input_tokens`, `output_cap`.
Feature references contain `path`, `sha256`, `bytes`. Paths are relative to the spec or
absolute external artifact paths. `storage_bytes` is 2 or 4. Ceilings use bytes, not GiB.
The metadata validator cannot discover aliases or semantic duplicates absent from inputs;
use the product data ledger and source-group audit before any acquisition.

`reduce-trace` consumes an object with `rounds`, each containing `proposed` token IDs,
`accepted` prefix count, `emitted` token IDs and `target_tokens` (at most one correction
or bonus). It reconciles emitted tokens with the prefix and separates target tokens from
accepted draft tokens. A fallback-only trace reports acceptance unavailable. This is
structural reconciliation, not independent proof that verifier decisions were correct.
The future runtime adapter must validate actual verification and EOS/cap accounting.

## Native build and capture

In the check environment, `python -m benchmarks native-build SPEC NEW_OUTPUT` reconstructs
an authenticated source/header bundle and performs a fresh CUDA build. The spec binds
`bundle`, `cuda_compiler`, `architecture`, `jobs` and `unpacked_bytes` under schema 1.
`native-helpers` takes `{schema: 1, runtime: ARTIFACT}` and rebuilds project helpers against
the pinned runtime. Helpers use native chat rendering and vocabulary-only tokenization;
a separate Python template renderer is not an authoritative input source.

In the numerical environment, `python -m benchmarks capture SPEC OUTPUT` acquires native
hard-label features. Its schema is `native-hard-capture-v1`; exact fields are in
`benchmarks/capture.py`. It binds runtime/model/bank artifacts, selected IDs, dtype,
context, payload limit, opaque exclusions and explicit window policy. Optional `role`
keeps TRAIN/DEV separate; heldout identity metadata can protect both. Reusing the same
spec/output resumes verified completed records and preserves partial attempts.

`curate` creates a provisional pool and panels from authenticated sources. `admit-bank`
rechecks selected native token identities and known protected aliases before publishing
the 6K TRAIN, 600 DEV and nested 120-request panels. It requires pinned pool/source-result
and exclusion artifacts; optional heldout exclusions apply to every role. Source and
content checks are bounded safeguards, not semantic deduplication proofs.

## Native request comparison

`compare` requires an authenticated source manifest, server binary, runtime libraries,
target, workload and at least two arms: `target` first, then named speculative drafters.
The schema is `native-comparison-v1`; see `benchmarks/comparison.py` for exact fields.
Short/full/harness request counts are 120/600/90; smaller pilots are explicit.
Comparisons use greedy generation, thinking disabled, F16 caches, equal context/depth,
serial arms, configured warmups and rotated repeat order. Every arm must match the
target's token IDs, text and stopping fields. Draft counts reconcile against independent
native final log counters. Fallback-only speculative arms are refused.

The output must be new. It preserves specification, workload, source snapshot, process
ownership, loaded libraries, server logs, request/response bodies and partial failures.
Totals use total tokens divided by total wall time and accepted/proposed token totals.
These request measurements alone do not establish live-task transfer or model quality.
The runtime must expose its final counters at ordinary info verbosity; trace logging is
not part of clean timing. CUDA launch checks active processes and obtains a host lease.

`python -m benchmarks curate SOURCES.json NEW_OUTPUT` runs in the numerical environment.
It authenticates downloaded source artifacts and the original TRAIN manifest, renders
complete histories with the target template and thinking disabled, and obtains actual
native token counts. It never reads sealed-test prompt bodies. Selection solves category,
subtype and length quotas from real available records; insufficient coverage is refused.
Related source groups and identical tokenized prompts cannot cross roles. Public source
exposure remains unknown. Curation outputs need semantic and ancestry review before capture;
quota compliance alone does not establish representative or sufficient data.

## Independent verifier checks

`native-build` applies the pinned sampled-EOG boundary repair and records it in the build
receipt. Optional `verification_trace: true` produces a diagnostic runtime that records
actual proposals, independently scanned target-logit argmax IDs, accepted vectors, replay
and commit decisions. Comparison specs must explicitly mark that runtime `diagnostic`;
instrumented results cannot select checkpoints or supply primary timings.

`verify-greedy` takes schema 1 with authenticated `comparison` and named-arm `logs`.
Use a complete single-block diagnostic comparison without warmups. The independent
oracle checks every greedy prefix, correction/bonus and EOG boundary, then reconciles
proposed/accepted totals. These are verified-token counters; work past an emitted output
cap can still count. Requests with no speculative work may omit response draft fields;
they count as zero only subject to reconciliation against the final native counters.

`verify-sampler` takes schema 1 with authenticated `source`, `binary`, `libraries`, `target`
and `trials`, plus optional `device` (`cpu` or `cuda`). It runs the actual common native
sample-and-match/rejection functions against controlled target logits, including zero
probability support and EOG. The helper checks fixed probability/acceptance margins over
10K–100K trials per case. CPU distribution tests qualify these host-side algorithms;
neural GPU requests and deployed cache behavior still need separate evidence.

A comparison can declare `sampling`: `temperature`, `top_p`, `top_k`, `min_p`, `seeds`
and an authenticated `qualification` result. Tier `sampled` requires 120 requests and three
fixed seeds, with qualification bound to the same target/runtime and both native sampler
functions. Smaller unqualified pilots are explicit. Outputs need not equal target-only
streams under paired seeds. Sampled throughput is descriptive; it cannot rank checkpoints
by producing longer/different answers. Sampled live application behavior is not qualified
by greedy workflow tests.

## Live workflows and harness sensitivity

`prepare-panels` consumes schema 1 with authenticated `workload` and `live_definitions`
references. It creates 90 harness requests from 30 distinct DEV groups under three
system instruction sets. Task content and existing tool definitions stay fixed. Variants
retain their source groups for paired analysis. It also materializes the twelve controlled
DEV workflow fixtures in [the maintained definition](../benchmarks/fixtures/live-dev.json).
They cover bug fixes, features, diagnosis, documentation followups, tool workflows and
revisions. These small authored scenarios test the execution mechanism; they do not
establish coverage of real repositories or a particular coding application.

`live` accepts the comparison schema with tier `live` (12 DEV or 24 confirmation tasks),
or an explicit smaller pilot. Tasks add an authenticated `fixture` and optional `followup`.
Each arm/repeat gets a fresh workspace, protected acceptance checks and deterministic
read/write/list/test tools. Tests run in Bubblewrap without host home or network access,
with read-only checks and bounded CPU, memory, file size and wall time. Install the host
`bubblewrap` and `util-linux` packages; Linux user namespaces must work. The CPU fixture
test checks actual sandbox execution, failing originals, accepted repairs and input reset.

The first request is cold; later turns reuse the prefix within that task. Model wait,
tool time, overhead and whole-task elapsed time are separate. Every task remains in the
completion denominator, including caps, failed checks and deadlines. Tool errors omit
host-specific paths so all arms receive identical observations. Greedy trajectory and
final filesystem/check fingerprints must match target-only for a valid comparison.
Keep failures as data; invalid measurement and unsuccessful model tasks are different.

`suite` consumes `full-suite-v1` with named `targets`, each supplying `requests`, `live`
and `harness` comparison artifacts. Panels must use the same target and candidate arms,
correct tier counts and at least two repeat blocks. Three rotated blocks balance three
arm positions; two alone do not. Completed panels can be reused only under the same
bound inputs/source; incomplete timing blocks require an explicit whole-panel rerun.
Scientific conclusions remain separate from operational suite completion.

## DEV loss and periodic training

`evaluate` runs in the numerical environment. `development-loss-v1` binds `source`,
`vocabulary`, a DEV-only `corpus`, named checkpoint `states`, `windows_per_task`,
`memory_bytes`, `training_corpora` and hard-heldout `exclusions`. It authenticates inputs,
refuses training/DEV identity overlap, and reports task-mean weighted loss overall and
by category. All candidate states use the same reference features and window anchors.
This teacher-forced loss is not native acceptance.

An intent may provide `development`: a `training-diagnostics-v1` artifact binding a
DEV `corpus`, hard-heldout `exclusions` and exactly 120 distinct `{id, anchor}` windows.
The worker checks these at cursor zero and every 250 updates by default, separately from optimizer
timing. It reports loss and top-1, checks training/DEV separation and preserves RNG.
An explicit positive `development_cadence_updates` changes cadence without shrinking
the panel. Measure its cost along with native evaluation overhead.

`campaign` consumes `training-campaign-v1`: authenticated `intent`, `comparison` and
`initial_drafter`, positive `optimizer_seconds_per_evaluation`, and explicit
`evaluation_failure` (`stop` or `continue`). A durable initial parent is required. It
evaluates at zero updates, runs the public supervisor until the next optimizer-time
boundary, then unloads training before native evaluation. Each continuation restores
the latest Adam/RNG/cursor; a provisional fast-DEV leader is a separate pointer.
Potential leader changes compare against the prior leader in the same timing block;
ties retain the earlier leader. Failed/partial evaluations cannot select a state.
The full-DEV incumbent remains separate and is not assigned by fast checks.

Campaign status and attempt journals are the operational evidence. A segment's product
report describes that segment; campaign status owns cross-segment selection. Record
measured evaluation overhead and lengthen cadence toward the planned ten-percent goal
instead of shrinking the frozen panel. An individual test run does not establish that
a long campaign, failure recovery or retention across many segments is qualified.

## FP16 capture storage

The numerical reader accepts metadata `feature_dtype: float16` with authenticated
`layer-N.f16` arrays; absent dtype retains legacy `layer-N.f32` behavior. It materializes
FP32 features for unchanged numerical training. Both stored and materialized geometry,
finite values, byte budgets and optional token/label arrays are checked. No automatic
conversion or capture deletion occurs. FP16 numerical loading support is separate from
scientific suitability. The [scoped HE4/Q4KM qualification](../results/fp16-storage-adoption/result.md)
passed matched training, separate DEV loss and native acceptance gates. The
[initial screen](../results/fp16-feature-screen/result.md) remains earlier bounded evidence.

The bounded screen command runs in a numerical environment:

```sh
python -m w1a1_eagle.numerical.precision FEATURE.f32 --sha256 DIGEST --bytes SIZE --device cpu
```

`--device cuda` explicitly uses that host's visible CUDA device. Check task ownership
first. It checks one authenticated F32 array up to 32 MiB, reports rounding error,
overflow and underflow, and never claims learning/native acceptance qualification.

## Development and evidence

Run the repository quick check while editing and full gate before committing. Tests
exercise cross-target refusal, role leakage, storage arithmetic, correction/bonus
accounting, fallback and FP16 feature round trips. GPU screens remain explicit external
experiments. No GPU access is needed for normal pre-commit.

Use [result closeout](results.md) and [close-results](../skills/close-results/SKILL.md)
for durable investigations. Preserve raw execution evidence externally, compact outcomes
in results and current conclusions in FINDINGS. Existing CPU check success never means
the remaining native runtime, sampled verifier or live application path passed.

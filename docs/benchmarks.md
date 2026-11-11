# Benchmark tools

The [implementation task](../tasks/benchmark-implementation.md) owns the remaining
training, native comparison, live workflow and harness qualification work. The commands
below include implemented metadata/trace tools and native request comparisons.
The live workflow, sampled verification and periodic selection paths remain in progress.

```sh
uv run --locked --no-sync --offline python -m benchmarks plan SPEC.json
uv run --locked --no-sync --offline python -m benchmarks verify-inputs SPEC.json
uv run --locked --no-sync --offline python -m benchmarks reduce-trace TRACE.json
uv run --locked --no-sync --offline python -m benchmarks compare COMPARISON.json NEW_OUTPUT
```

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

## FP16 capture storage

The numerical reader accepts metadata `feature_dtype: float16` with authenticated
`layer-N.f16` arrays; absent dtype retains legacy `layer-N.f32` behavior. It materializes
FP32 features for unchanged numerical training. Both stored and materialized geometry,
finite values, byte budgets and optional token/label arrays are checked. No automatic
conversion or capture deletion occurs. FP16 numerical loading support is separate from
scientific suitability. See [screen result](../results/fp16-feature-screen/result.md).

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

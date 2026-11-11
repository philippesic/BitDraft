# Benchmark tools

The [implementation task](../tasks/benchmark-implementation.md) owns the remaining
training, native comparison, live workflow and harness qualification work. The commands
below are implemented metadata/trace tools. They do not yet launch native evaluations.

```sh
uv run --locked --no-sync --offline python -m benchmarks plan SPEC.json
uv run --locked --no-sync --offline python -m benchmarks verify-inputs SPEC.json
uv run --locked --no-sync --offline python -m benchmarks reduce-trace TRACE.json
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

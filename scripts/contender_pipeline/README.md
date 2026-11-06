# Candidate training and evaluation workflow

This package executes trusted adapters with explicit costs and evidence. It does
not select an architecture, invent a learning-curve predictor, or qualify an
unknown export format. The current adapter is the existing Qwen3.5-4B Q4_K_M /
fixed-sign row-W1A8 branch, RTX5080, frozen mode32 correctness runtime.

## Check the workflow without training

From the repository root, `make workflow-check` runs the maintained CPU contract
suite with `.venv/bin/python`; override `PY` explicitly for another prepared
Python environment. It covers accounting, stale inputs, failure recovery,
comparison integrity and artifact packaging. Linux process-identity tests require
Linux. This command does not train or allocate a GPU and does not qualify a new
candidate. Respect an active owner's quiet measurement window even for CPU tests.

## Entry requirements for a candidate

Write a candidate record before training: architecture/weight/activation/head
contract; target/runtime pins; trainable parameter names; initializer; source/data
ancestry; objective and masks; optimizer, sampler, schedule horizon; exporter and
native loader; intended workload and resource limits. Shared target captures are
reusable only when target/taps/history semantics agree. Actor caches and replay
banks require their own compatibility checks.

An adapter is an ordinary argument-vector command plus machine-readable JSON
results. It must expose meaningful update counts/data exposure, finite loss and
gradient checks, parameter changes, exact checkpoint state, exporter output pins,
actual peak memory and update time. It must honor SIGTERM between complete updates,
checkpoint safely, and clean native processes. An exception during an optimizer
update is **not** proof of a resumable completed update. Preserve a known complete
checkpoint; failed outputs require explicit review/recovery and are never promoted.

The current trainer integrates export at completion. Its original 19200 schedule
horizon is preserved by the rehearsal wrapper; the bounded ablation CLI still
caps cursor7680. Long allocations use a specifically qualified long-run adapter,
not a silent cap increase. Original native/data/source admission checks remain.

## Plan and budget contract

`runner.py` consumes `contender_plan_v1` JSON. All executable/cwd paths are
absolute, commands are argv arrays (no shell interpolation), jobs are sequential,
and dependencies must precede jobs. A successful exit requires declared JSON
checks and optional artifact pins. New output paths must be unused.

```json
{
  "schema": "contender_plan_v1",
  "budget_seconds": 3600,
  "candidate_budgets": {"candidate_a": 3000, "shared_evaluation": 600},
  "grace_seconds": 180,
  "inputs": [{"path": "/absolute/frozen-input", "bytes": 123, "sha256": "..."}],
  "gpu": {
    "control": "/home/philip/.config/binary-eagle-decoding/gpu-control.json",
    "lock": "/home/philip/binary-eagle-decoding/runs/contender-gpu.lock"
  },
  "jobs": [{
    "id": "candidate_a_pilot",
    "candidate": "candidate_a",
    "stage": 0,
    "depends_on": [],
    "timeout_seconds": 600,
    "argv": ["/absolute/python", "/absolute/qualified_adapter.py", "--config", "/absolute/config.json"],
    "cwd": "/home/philip/binary-eagle-decoding",
    "checks": [{"path": "/absolute/new-result.json", "keys": ["status"], "equals": "TRAINED_EXPORTED_NOT_NATIVE_ADMITTED"}],
    "artifacts": ["/absolute/new-model.gguf"]
  }]
}
```

This is a schema example, not a runnable experiment allocation. Put every source,
configuration and immutable input needed by the adapter in `inputs`; the adapter
must verify its transitive scientific dependencies. Hashes are rechecked across
stage resumes and around execution. Baseline/evaluation work can be charged to an
explicit shared account; never call update-only time total GPU cost.

The runner reserves timeout + cleanup grace +5s before starting. Actual command
wall (including setup, native evaluation, audit/cleanup) is charged after exit.
A hard crash retains the whole reservation until explicit recovery. Cleanup may
need to overrun a compute deadline to stop safely; these are supervised wall
budgets, not a hardware power/utilization meter. Logs retain at most two8MiB files
per job. No failed job is silently retried or overwritten.

Use `launch.py` to check GPU occupancy and start the existing local supervisor:

```
.venv/bin/python scripts/contender_pipeline/launch.py \
  --project /home/philip/binary-eagle-decoding \
  --plan /absolute/plan.json --output /absolute/execution \
  --through 0 --run-id unique-local-run --receipt /absolute/launch.json
```

Repeat with a new supervisor run ID/receipt and a higher `--through` to continue
an unchanged plan. Completed job evidence must still hash identically. A changed
plan requires a new execution directory. A recovery/extension plan should set
`prior_state` to the pinned terminal previous state so failures and earlier work
remain charged to the same campaign. Budget extension is an explicit allocation,
not an automatic retry feature. Use identical cumulative budget ceilings unless
the user allocated more.

The GPU lock excludes other instances of this runner. External jobs must also
respect the project's quiet-window policy. Initial GPU occupancy is checked by
the launcher; another unrelated scheduler is not controlled by this package.

## Recovery

Normal stop: authenticate launch boot/start ticks, TERM the supervisor, allow its
recorded grace, then inspect execution state and native release receipts. Runner
records descendant identities and cleans separately-sessioned descendants too.
For an abandoned runner, `recover.py --state ... --job ...` locks the state,
checks boot/birth identities and releases recorded processes. It refuses PID reuse
and ambiguous leaderless groups. This is intentionally fail-closed, not a command
to kill all GPU processes. A child that escapes before being observed still needs
its adapter's own process receipts; no arbitrary-process cleanup is promised.

Inspect the checkpoint and exception cause before creating an explicit recovery
plan. The current rehearsal adapter accepts only its deliberate cooperative-stop
receipt and pinned interrupted checkpoint. Do not reuse a partially updated state
after OOM/nonfinite/device failure merely because a file exists.

## Data mechanism

`sharded_rows.pack` authenticates original F32 bytes and packs whole matrices into
bounded shards. `ShardRows` implements the existing `matrix(spec)` interface with
an LRU of mapped shards, original source identities, segment hashes and read-only
views. Caller row/finite/ancestry checks remain mandatory. Limits cover internal
mappings; retained caller views and OS page cache are separate. A source matrix
larger than the configured shard size is rejected; choose a measured explicit
bound. Failed packing does not publish an admitted manifest.

`ExposureSampler` provides shuffled complete epochs and resumable exposure for
future adapters. Its state is tied to seed and ordered IDs; pin NumPy/environment
as well. It does not replace the current proven schedule during this qualification.

`SnapshotAudit` uses bounded immutable copies while delegating row geometry,
counting, digest and finite checks to the original union auditor. Other filesystem
admission checks remain active. `optimized_current.py` explicitly opts into these
helpers; its freeze must pin the wrapper, helpers and shard index. Historical
trainer/source files are unchanged. Dataset content/order is unchanged.

Larger corpus selection/capture and complete train/dev/confirmation alias exclusion
remain scientific dataset preparation. Packing600MB successfully is evidence of
bounded streaming, not proof that48FIT groups are enough for long training.

## Evaluation tiers

1. **Smoke:** actual export/loader, native output parity and process release;
   no ranking. Current adapter runs14 requests per new endpoint.
2. **Contender:** current adapter includes target-only, frozen DFlash, initializer,
   existing paid candidate and proposed endpoints;12 consumed DEV cases plus2
   warmups, five alternating repetitions. Raw native metrics, requests, responses,
   settings, models, loaded libraries and release receipts are audited.
3. **Confirmation:** freeze selected model/settings before new input selection.
   The existing `modern_natural_confirmation.py` pipeline supplies source/alias
   exclusion, target context admission and protected evaluation. Do not open a
   final pool merely to exercise infrastructure. `comparison.py` checks complete
   paired panels and exact greedy output parity with configurable case count.
   Synthetic tests exercise confirmation reservation and consumption on failure.

`comparison.py` rejects missing/duplicate cases, changed pins, nonfinite timings,
unequal requests/tokens and correctness failures. Its paired case/repetition
bootstrap is exploratory and unadjusted for multiple candidates. Smoke has no
ranking; no automatic promotion or forever-rejection is implemented. Selection
uses actual native outcomes across multiple training checkpoints. Identical model
exports in the rehearsal are timing-repeatability controls, not independent model
ideas or evidence that one architecture beat another.

`reserve_confirmation` requires an authoritative ledger of canonical case/group
IDs that already includes historical training, development and previous evaluation
consumption. The existing source/alias audit must produce these identities. The
reservation helper is not a replacement for content/alias deduplication. It marks
cases consumed before requests, including failed attempts, under an exclusive lock.

Future adapters may produce `contender_comparison_v1`; they must separately prove
native execution, source/target/verifier semantics and canonical source grouping.
The common comparator cannot prove these from a model name or a JSON assertion.
Stochastic sampling, different context/concurrency, other targets or a changed
verifier need their own runtime admission before extending the supported scope.

## Local deployment bundle

`bundle.py` copies the chosen drafter and settings into a verifiable local bundle,
records target/runtime dependencies and evidence, and supports relocation/hash
checks. It does not upload anything. A reference rehearsal cannot silently claim
an independently confirmed winner. A winner claim requires a protected complete
confirmation panel, the selected model and frozen matched DFlash baseline.

A new runtime or machine still needs its installation/native smoke. Target weights
and runtime libraries are external pinned dependencies, not silently bundled or
redistributed. The supported workload and correctness contract travel with the
artifact. The actual user-selected design must pass these same gates before a
long allocation and before a product-level performance claim.

## Tests and evidence

```
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover \
  -s scripts/contender_pipeline -v
```

See `experiments/contender-infra-20261104/README.md` for measured GPU qualification,
exact checkpoint/export comparisons, data coverage, costs and limitations.

For metadata that also exceeds a single resident dataset, `ManifestStream` keeps
an explicitly bounded LRU of already-admitted member manifests and exposes stable
sample IDs to `ExposureSampler`. Only selected existing `role=fit` windows may
materialize. Pins are checked when a manifest is reopened; changed targets,
held-out windows, oversized metadata and changed sampler IDs fail closed. Use
one shared `row_reader=ShardRows(...)` across manifest loads so eviction does not
repeat array authentication and index construction. This preserves every sample
and its order while bounding active metadata. Registry size and individual
manifest size have explicit limits; total Python RSS is measured separately.

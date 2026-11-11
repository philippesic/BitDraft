# Supported behavior and qualification limits

The public workflow supports CPU metadata admission, authenticated capture reading,
HE4 numerical updates, supervised attempts, transactional checkpoints, exact same-recipe
resume and explicit branches, and streamed GGUF export/readback. See [execution](execution.md),
[data](data.md) and [model](model.md) for the actual contracts.

## What current checks establish

Reduced CPU fixtures exercise arithmetic, state restoration, ownership, recovery and
export. The retained modern metadata index reconciles its records and window counts.
Separate RTX 5080 evidence establishes the final HE4broad checkpoint’s next-update parity
and a three-request native target/baseline/candidate token-and-stopping pilot. See
[migration](../results/he4-checkpoint-migration/result.md). These bounded measurements
do not establish long-run equivalence or a serving-speed improvement.

Both check and numerical environments install from the lock. A fresh numerical wheel
install passed the CPU suite on the current host without importing from the old checkout.
The full gate uses the configured interpreter; this does not qualify other hosts or CUDA.
[Checks](checks.md) describes actual lane coverage and unavailable stages.

## Unsupported or unqualified paths

| Area | Current boundary |
| --- | --- |
| Training device | CPU and CUDA intents are implemented; CUDA obtains an exclusive host lease and refuses existing compute jobs. |
| Historical checkpoints | Authenticated legacy BSZ import is implemented; the final HE4broad state reproduces its next original-data update exactly. |
| Native runtime | Complete rebuild and maintained native-build/native-helpers commands were exercised on RTX 5080; build receipts remain external. Broader host compatibility is unqualified. |
| Serving correctness | No exported model from this repository has completed native target-only/speculative output, stopping and finite-logit qualification. |
| New data capture | Resumable capture and bounded format-conversion commands are implemented; diverse TRAIN/DEV qualification captures have run. Production format adoption remains under matched qualification. |
| Scientific selection | Category/length selection, opaque identity admission and paired request statistics are implemented. Serving-based checkpoint selection and live workflow integration remain pending. |
| End-to-end path | Public FP16 capture → optimizer-preserving CUDA continuation → export completed; schema-4 GPU resume and complete serving-based periodic workflow qualification remain pending. |
| Exposure across branches | Aggregate counters and lineage survive; a per-position lifetime ledger across changed recipes/corpora is not implemented. |
| Data recovery | The historical eight-prompt discrepancy remains unresolved; see the deduplication result. |

These are present capability limits, not an active task list or a scientific allocation.
New work should open a scoped task and define observable acceptance criteria. Historical
model families, training budgets and consumed cohorts are evidence, not automatic new
requirements. HE4 is the development candidate; broader model support remains a choice.

## Reusing historical implementations

The [source inventory](../evidence/source-inventory.json) and [evidence access guide](evidence.md)
identify preserved references. Recover and verify all source/build inputs before relying
on an old runtime. Check retained tensors, masks, losses, gradients, updates and exports
at the required scale; a metadata or small-fixture pass cannot stand in for that work.
Never silently reset optimizer or exposure state to make a historical import succeed.

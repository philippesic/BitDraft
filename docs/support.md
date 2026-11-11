# Supported behavior and qualification limits

The public workflow supports CPU metadata admission, authenticated capture reading,
HE4 numerical updates, supervised attempts, transactional checkpoints, exact same-recipe
resume and explicit branches, and streamed GGUF export/readback. See [execution](execution.md),
[data](data.md) and [model](model.md) for the actual contracts.

## What current checks establish

Reduced CPU fixtures exercise arithmetic, state restoration, ownership, recovery and
export. The retained modern metadata index reconciles its records and window counts.
These checks do not establish full-size equivalence to a historical trained state,
CUDA execution, native token parity or a serving-speed improvement.

The check environment installs from the lock. The full gate uses the configured
numerical interpreter; a clean rebuild of that numerical environment has not been
qualified. [Checks](checks.md) describes actual lane coverage and unavailable stages.

## Unsupported or unqualified paths

| Area | Current boundary |
| --- | --- |
| Training device | Public intents accept CPU; the CLI masks CUDA. GPU state fields are not GPU support. |
| Historical checkpoints | The worker restores its own schema 3. Historical locators do not implement conversion or import. |
| Native runtime | No rebuilt, source-complete runtime is qualified here. Historical runtime04 reused objects from external run trees. |
| Serving correctness | No exported model from this repository has completed native target-only/speculative output, stopping and finite-logit qualification. |
| New data capture | Existing captures can be authenticated and read; no public fresh-capture workflow is implemented. |
| Scientific selection | Workload selection, comparison statistics and serving-based checkpoint selection are not implemented. |
| End-to-end path | A real capture/import → train → stop/resume → export → native verification workflow is not qualified. |
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

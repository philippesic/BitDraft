# Supported behavior and qualification limits

The public workflow supports CPU metadata admission, authenticated capture reading,
HE4 numerical updates, supervised attempts, transactional checkpoints, exact same-recipe
resume and explicit branches, and streamed GGUF export/readback. See [execution](execution.md),
[data](data.md) and [model](model.md) for the actual contracts.

## What current checks establish

Reduced CPU fixtures exercise arithmetic, state restoration, ownership, recovery and
export. The retained modern metadata index reconciles its records and window counts.
Separate RTX 5080 evidence establishes the final HE4broad checkpoint’s next-update parity
and bounded native target/baseline/candidate token-and-stopping comparisons. See
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
| Serving correctness | Both matched precision descendants passed 120 target-token/stopping requests. An independent greedy target-argmax oracle passed 664 verification rounds, including EOG and short caps. This is bounded fidelity evidence, not universal runtime qualification. |
| New data capture | Resumable TRAIN/DEV capture and streamed conversion ran. FP16 passed the scoped HE4/Q4KM qualification; other targets require separate storage checks. |
| Scientific selection | Category/length selection, opaque identity admission and paired request statistics ran. Controlled live tools passed 12 tasks × three arms × two repeats. Full-suite/campaign coordinators are implemented but their complete GPU campaigns, recovery and selection qualification remain pending. |
| End-to-end path | Public FP16 capture → optimizer-preserving CUDA continuation → export completed. Schema-4 GPU resume and periodic 120-window diagnostics ran. Production hybrid continuation has not yet launched. |
| Sampled verification | Actual native sample-and-match and rejection functions passed controlled distribution/EOG tests for both targets after a measured rejection-EOG repair. Neural sampled request qualification remains separate. |
| Long inputs | Real FP16 capture and training ran on a 13,974-token prompt. The current bank's maximum prompt has been captured; its memory proof is tracked in the implementation task. |
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

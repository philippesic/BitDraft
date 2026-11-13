# Supported behavior and qualification limits

Public entries cover prompt admission, teacher capture, HE4 CPU/CUDA updates,
optimizer-preserving recovery, GGUF export, native requests, tool episodes and campaign
coordination. See [benchmarks](benchmarks.md), [execution](execution.md), [data](data.md)
and [model](model.md) for contracts; [FINDINGS](../FINDINGS.md) for measured conclusions.

## What current checks establish

The full local gate tests reduced CPU fixtures, required coverage and negative/control
rule specimens. Numerical dependencies install independently from the old checkout.
CPU passes do not certify hardware, native outputs, useful learning or performance.
[Checks](checks.md) names each lane; GPU evidence lives with its result and frozen source.

Separate RTX 5080 measurements establish original HE4broad next-update parity and scoped
[FP16 adoption](../results/fp16-storage-adoption/result.md). Native greedy outputs/stopping,
CPU sampler controls with neural GPU requests, CUDA kernel activation, tool trajectories,
train/eval recovery and retention have execution receipts. Read each cohort and condition;
one successful path does not qualify arbitrary inputs or configurations.

## Unsupported or unqualified paths

| Area | Boundary |
| --- | --- |
| Models | Current HE4/DFlash/Qwen3.5-4B family. Q4KM and NVFP4 targets have separate identities. New families need graph, taps, vocabulary, checkpoint and native qualification; no model plug-in registry exists. |
| Storage | FP16 teacher features qualified for HE4/Q4KM; training materializes FP32. NVFP4 capture and bounded continuation used FP32; its FP16 storage needs separate evidence. |
| Hardware | GPU evidence is RTX 5080 with the pinned toolchain. Other hosts/devices are unqualified; the numerical wheel does not support the 2080 Ti's sm_75. |
| Context | Real 13,974/14,651-token inputs passed bounded CUDA updates. Protocol limits are not proof of every length or workload. |
| Selection | Short DEV leader, full-request experimental incumbent and latest recovery are separate. Workflow/harness regressions cannot be hidden by a request score. No automatic deployment promotion. |
| Generalization | Public source exposure is unknown. Fixed DEV mixtures and one-project workflows do not prove real-traffic prevalence or unseen application transfer. Independent confirmation needs new curation and access receipts. |
| Exposure | Dataset-local counters and lineage survive; per-position lifetime history across changed recipes/corpora is incomplete. |
| Historical data | The eight-prompt discrepancy remains unresolved; see [deduplication](../results/null-topic-deduplication/result.md). |

These are capability boundaries, not new allocations. Active work belongs in `tasks/`;
old pending/active labels do not authorize work. Preserve healthy jobs and verify owners
before recovery. GPU workers refuse competing compute jobs and take a host lease.

## Reusing historical implementations

[Evidence access](evidence.md) identifies independent archives and remaining raw-input
limits. Verify required source, dependencies and operands before reuse. Recovering a
report is distinct from reproducing its experiment. Never silently reset Adam, RNG,
source initialization or exposure to make a historical import succeed.

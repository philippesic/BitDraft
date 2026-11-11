# Decision record

Current unresolved choices live in [STATUS](STATUS.md) and its active campaign,
not in the dated pending sections of the [decision archive](history/DECISIONS_20261105.md).

Current decisions already have authoritative homes:
- [Candidate stages](CANDIDATE_STAGES.md): foundation, precision, then feature tiers.
- [Candidate protocol](CANDIDATE_BENCHMARKING.md): external baseline, fixed references,
  recovery controls, frontier and incumbent.
- [Component campaign](COMPONENT_CAMPAIGN.md): current scope and allocation.

Add a short dated entry here when a consequential decision changes: choice,
reason, evidence and which earlier decision it supersedes. Keep detailed execution
and measurements in the campaign/run record rather than copying them here.
No unresolved historical question is silently declared resolved by this index.

## 2026-11-06: prioritize HE4; retain Csmallbroad fallback

**User decision:** prioritize HE4 as the primary binary-drafter development
candidate; revisit Csmallbroad as a fallback. This supersedes treating the two
retained frontier endpoints as equal development priorities. External DFlash
Q4_K_M remains the deployment incumbent until matched throughput evidence
supports a promotion.

HE4 uses W1 for all 18 FFN matrices, A4 gate/up and A8 down inputs, Q4_0 attention
and fusion, one private tied Q4_K output-head/embedding matrix, F16 draft K/V,
and F32 row scales, midpoints, gains, clipping bounds and norm storage. Use its
qualified 4,096-update retained state, reproduced with the original history and
verified for actual GPU resume and exact evaluated native bytes. The
[frontier retention receipt](../runs/component-campaign-20261105-01/frontier-retention.json)
identifies the full state and native export.

Reason: lower complete draft latency provides more room for an eventual
end-to-end throughput win if acceptance improves. This is a development choice,
not a claim that HE4 currently leads binary request throughput or has matched
external Q4 acceptance. [Formal evidence](../experiments/component-campaign-20261105/report.md).

Csmallbroad retains target-shared Q6_K head/embeddings, Q4_0 draft K/V and reduced
gain/clip/norm storage. Preserve its original formal 4,096-update endpoint as the
fallback reference and its separate extra-30-minute state/evidence. The latter
showed no consistent throughput improvement and is not automatically a promoted
replacement. Revisit this family if HE4's private vocabulary/cache memory limits
the target workload or if memory/context tradeoffs outweigh its latency advantage.
[Continuation evidence](../experiments/dflash2-smoke-20261106/report.md).

No new training, feature sweep, deployment change or artifact retirement is
authorized by this prioritization alone. Preserve the qualified frontier states
and their required parents/data/source dependencies.

## 2026-11-06: revisit mixed FFN weight precision later

User requested recording, then deferring, selective ternary/2-bit FFN weights.
HE4 stays the primary candidate. Keep its A4 gate/up and A8 down activation
settings, vocabulary, cache and essential features fixed while testing weight
precision across the 18 individual FFN projections.

Proposed future sequence: qualify packed W2 training/export/native execution;
screen six individual down upgrades, then six paired gate/up upgrades; shortlist
allocations at matched added bytes/native cost and give them matched short QAT
against an unchanged HE4 continuation. Compare ternary at promising allocations
and validate combinations rather than assuming isolated gains add together.
Start with down for implementation isolation, not presumed higher sensitivity.

Keep exact HE4 warm-start expansion and initialization from pinned original
higher-precision source tensors as distinct recipes. Binary sign latents are not
original dense weights; wider storage alone recovers no missing information.
Define learned scale/midpoint and codebook semantics explicitly. Straightforward
ternary packing also costs two bits; a zero code with a nonzero midpoint is not
a zero weight. Measure actual bytes, packing-inclusive native draft/request cost
and acceptance. Current native exporter admits only all-18 W1 FFN matrices;
mixed-weight support still requires qualification. Bound training/native drift
before attributing small recovery differences to precision.

This entry authorizes no implementation, GPU run or training allocation.

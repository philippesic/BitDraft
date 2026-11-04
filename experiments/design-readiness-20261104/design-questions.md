# Questions for the joint architecture and precision discussion

The backlog audit is a map of evidence and implementation dependencies. It does
not select a final architecture, precision, scale granularity or training budget.
New compatible-feature results belong in the eventual readiness report.

## Decisions that determine which features are applicable

1. **Deployment objective and constraints.** Keep the selected four modern target
   configurations and frozen DFlash comparison policy. Decide the intended prompt
   lengths, output lengths and concurrency before a final deployment claim. Our
   current 12-prompt/80-token DEV screen is an inexpensive development workload,
   not a universal deployment proxy.
2. **Proposal structure and information access.** Which target features can the
   drafter use, how many tokens are proposed together, and which draft histories
   recur during training? These choices determine cache/label alignment and whether
   historical sequential depth, survival and feature-matching losses have the
   intended meaning. The user will choose the architecture.
3. **Representation and exceptions.** Decide the weight/activation representation,
   scale granularity, and treatment of sensitive layers and the vocabulary head.
   Include actual kernels and packing in the cost. Per-row and within-row group
   scales are distinct representations; their storage and partial-sum arithmetic
   are mapped in decision-feature-map.md. No group-size experiment is authorized
   before the joint decision.
4. **Trainable capacity.** Decide which signs, scales, normalization gains or
   corrections can change. The measured recent transfer comes from fixed signs
   and 76,880 row scales, not complete sign QAT. Selected attention-output sign
   training has separate meaningful-flip/native/resume evidence; all-sign training
   needs further contract and resource qualification.
5. **Teacher and training states.** Preserve exact target precision, feature taps,
   history/position/cache semantics and source partitions. Raw native target
   features may be reusable when those inputs stay identical; actor-specific KV,
   reached masks and margin banks need reevaluation for the chosen drafter. Soft
   probability distillation needs an authentic current probability teacher. A
   hidden-MSE substitute does not test a KL hypothesis.

## Feature evidence to bring to those decisions

- Data/replay and reachable/depth objectives: conceptually broad, with actor-
  dependent masks and coefficients. Compare actual decoding, not training score.
- Group scales, extra planes, affine changes and rotations: representation
  choices with explicit tensor/metadata/kernel and optimizer implications.
- Low-rank/sparse corrections and factor/shortlisted heads: capacity and runtime
  changes; retained local gains are scoped and do not establish modern benefit.
- Adaptive proposal policies and true-tree verification: proposal/verifier
  semantics change. Their historical smoke tests do not qualify a new verifier.
- Transfer/packing/catch-up improvements: inspect the actual resolved graph before
  crediting savings already implemented. Exact native rounds help prioritize the
  remaining work; source-level wasted work alone does not prove a speedup.

The detailed family and F01–F32 mappings are in backlog-inventory.json and
decision-feature-map.json. Historical short negatives remain scoped; no
permanent elimination or calibrated universal early-stopping rule is established.

## What must be demonstrated after choosing a design

A written model/state contract, actual updates of intended parameters, resumable
optimizer/RNG/exposure state, deployed export, native trajectory/correctness
admission, measured memory/checkpoint costs, and a repeated native evaluation
hook. Current proven components can be reused where their contracts still apply.
A changed architecture or representation can require new implementation; the
preparation should identify that work before a long allocation is committed.

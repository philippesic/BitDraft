# Model card (draft)

Draft for the release retrain ([task](../release-retrain.md)). Bracketed items wait on the
run, the frozen gate and owner decisions.

## Model

- **What it is:** an HE4 speculative-decoding drafter for Qwen3.5-4B (Q4_K_M), loaded by the
  project's llama.cpp fork (runtime v10). One-bit feed-forward weights, four-bit gate/up
  inputs, eight-bit down-projection inputs, Q4_K head; other parts keep higher precision.
- **Origin:** trained from converted weights of the released DFlash drafter (Baseline), not
  from scratch, then trained 825,000 updates on 100,000 prompts with target-written responses
  ([data card](data-card.md)).
- **File:** [size] MB shortlist-only file with [rows] vocabulary rows, against Baseline's
  381 MB.
- **Licence:** [owner decision; must be compatible with the Baseline weights' licence and the
  data attributions].

## Use

The target verifies every drafted token, so outputs match the target's own decoding rules;
only speed changes. Greedy output is identical to target-only; sampled output follows the
target's distribution (sampler verified for Q4_K_M and NVFP4 on runtime v10).

## Results (frozen release gate)

[Fill from `gate-verdict` v2: general-scenario speed against stock Baseline with its 95% lower
bound and the measured session drift; the five guardrails (thinking-on, sampled, long input,
agentic replay, tool calls); the comparison with the 4-bit control and its reached update
count and curve; the held-out ReAct convention as transfer evidence; NVFP4 as a secondary
result.] Claims cover batch-one request workloads on an RTX 5080 only.

## Comparators

Target-only, stock Baseline (DFlash Q4_K_M as released) and the 4-bit control (Baseline's
architecture fine-tuned on the same token streams). [Same-shortlist attribution arm.]

## Limitations

- Measured on one GPU model and one target; other targets need their own drafter.
- [Scenarios where it is slower than Baseline, from the gate.]
- Multilingual acceptance depends on the vocabulary shortlist; [chosen variant and its DEV
  multilingual speed].

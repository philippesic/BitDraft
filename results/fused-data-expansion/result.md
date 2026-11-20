# Fused storage and 1.5× data make a drafter that passes the owner release policy

Fused F16 storage holds TRAIN features 8× smaller than eight-tap FP16. Its cost was
within noise: fused minus FP16 was −0.18 acceptance points, pooled over three seeds. That
freed enough disk to grow TRAIN from 9,047 to 13,394 records, aimed at the weakest
scenarios. Two passes over the larger set (335,812 updates, about 23.6 GPU hours) produced
export `ef068eba…`:
- it beats the prior leader `ae6569f6…` by 2.1–2.9 acceptance points on every screen;
- it is the first export to pass the owner release policy against Baseline;
- the owner promoted it to provisional acceptance leader on 2026-11-19.

## Outcome

### Fused storage retest

The panel is the 120-request Q4KM panel, with three seeds per storage, each a fresh
8,209-update calibrated run on `train-expansion-01`. The rule was fixed before running:
pass if the pooled 95% lower bound is at least −1.10 points (owner, 2026-11-18).

| Seed | Fused − FP16, points (95%) |
| --- | --- |
| 8101 | −0.14 (−0.50, +0.20) |
| 8102 | −0.24 (−0.57, +0.06) |
| 8103 | −0.17 (−0.52, +0.18) |
| Pooled | **−0.18 (−0.39, +0.03)**: pass |

Panel acceptance was 47.43% fused against 47.62% FP16. The supplementary 60-request screen
gave −0.43 (−1.02, +0.14).

### Expansion

| Component | Records | Fused size | Source |
| --- | ---: | ---: | --- |
| `train-final-01` (unchanged content) | 9,047 | 68 GB | earlier TRAIN |
| Capture A | 3,564 | 34 GB | MATH thinking-on (640) and plain (250); UltraChat thinking-on reasoning (360), writing, multi-turn; oasst2; QASPER 2–8K papers (300); CommitPack edits (400) |
| Capture B | 783 | 40 GB | 405 long-project agentic sessions (11–20K-token starts; 389 newly generated), 378 short sessions |
| `train-final-02-fused` | 13,394 | 142 GB | 8 components, `fused-mixture-v1` |

Pools were smaller than planned:
- commit-edit prompts were nearly used up, so there is no thinking-on code;
- QASPER has no 8K+ papers left;
- OpenAssistant writing gave 24 of 60.

All selections passed the existing exclusion checks: heldout24, confirmation-v2, DEV
identities, workloads and agentic project families.

### Long run learning curve

The schedule was cosine over 335,812 updates (2 × 167,906 groups per pass), with warm-up
6,716 and data-reuse cap 20. It ran as 24 segments of 14,000 updates, each resumed exactly
from the previous state. Acceptance is in % and every output was exact.

| Updates | Screen (60) | Panel (120) | Screen-v2 (60) |
| ---: | ---: | ---: | ---: |
| 14,000 | 48.72 | 40.98 | 49.13 |
| 28,000 | 50.59 | 42.73 | 51.70 |
| 56,000 | 52.69 | 44.48 | 52.87 |
| 112,000 | 54.11 | 46.32 | 55.66 |
| 168,000 | 56.07 | 47.85 | 57.53 |
| 224,000 | 58.63 | 49.69 | 58.94 |
| 280,000 | 58.23 | 50.39 | 59.60 |
| 335,812 | **59.75** | **50.81** | **60.30** |
| Prior leader | 57.64 | 48.73 | 57.38 |
| Baseline | 60.6 | 53.4 | 62.70 |

Every segment is in `result.json`. Gains ran at about 1.5–2 points per 14,000 updates
early, and about 0.3 over the last five segments.

### Selection against the prior leader

These are paired source-group bootstraps, candidate-only, against saved references.

| Panel | Difference, points (95%) |
| --- | --- |
| Screen (60) | +2.11 (+0.74, +3.50) |
| Panel (120) | +2.08 (+1.59, +2.52) |
| Screen-v2 (60) | +2.92 (+1.63, +4.31) |

### Release comparison against Baseline

Fresh three-arm timing (target, Baseline, candidate) ran at depth 3 under source
`8180f208…`. Greedy, thinking and sampled were repeated in reversed arm order and pooled.

| Scenario | Speed vs Baseline (95% lower) | Acceptance ours / Baseline | Prior leader |
| --- | --- | --- | --- |
| Agentic coding (60 tasks) | 1.108 (1.091) | 81.7% / 63.4% | 1.105 (1.089) |
| Greedy, long | 1.013 (1.002, merged strata) | 60.0% / 61.1% | 1.002 (0.990) |
| Sampled | 1.011 (1.003) | 47.7% / 50.0% | 0.994 (0.988) |
| Greedy, short | 1.010 (1.000) | 49.6% / 52.3% | 0.989 (0.979) |
| Thinking-on | 1.007 (0.998) | 63.0% / 66.6% | 0.978 (0.969) |

The owner policy with merged strata for `greedy-long` (owner decision 2026-11-16) has
**passed**: there are two tolerated shortfalls, greedy-short and thinking-on, with lower
bounds of at least 0.998. Agentic completion held, and every greedy output was exact. The
prior leader failed this policy. Under the maintained grouping, `greedy-long` has no
interval and the verdict reads `not-demonstrated`.

## Scope and limitations

- **Selection data:** all panels are reused DEV cohorts used for selection, so they lean
  optimistic. A speed claim needs one run on fresh, protected confirmation data, which was
  not used here.
- **Confounded causes:** data size, data mix, storage format and run length changed
  together. There is one run and one seed. The gain is not attributed to targeting or to
  more data alone. The fused retest covers storage only, on `train-expansion-01`;
  transfer to the larger set is assumed.
- **Agentic coding:** the agentic suite uses the same harness format as TRAIN sessions, so
  its 1.10× advantage may not transfer to other agent tools.
- **Panel coverage:**
  - long inputs reach 16K tokens, with 6 requests at 8–16K;
  - thinking-on covers reasoning and code only;
  - everything ran on one RTX 5080, Q4KM, batch 1. NVFP4 was not run.
- **Mid-run screens:** they were not annealed.
  - The 52%-at-28K kill bar was miscalibrated: it came from a run whose step size had
    already decayed.
  - The owner set it aside on 2026-11-18. The stop rule then became "two panel falls over
    0.3 points" (owner, 2026-11-19).
- **Screen-v2:** this new 60-request screen (thinking-on 14, agentic-coding 12,
  chat-writing 16, reasoning 6, coding 6, long documents 6) was introduced mid-run. It was
  measured retroactively on every segment from the same exports.

## Decision

- **Retain** fused F16 storage for TRAIN features.
- **Retain** `train-final-02-fused` as the full TRAIN set.
- **Promote** `ef068eba…` to provisional acceptance leader (owner, 2026-11-19).
- **Unresolved:** a release speed claim, which still needs protected confirmation data.
- **Unresolved:** whether further data or length gains remain. The curve was flattening
  but had not stopped.

## Evidence and reproduction

All paths are under `artifacts/feature-validation-01/` unless stated.

**Retest:**
- arms `arms/g3v2-calibrated-{,s8102,s8103}` and `arms/g3v2-calibrated-fused-{r2,s8102,s8103}`;
- panels in `fused-retest/panels/`, verdict in `fused-retest/verdict.json`;
- executed source `8c3d9db5…` for the new arms.

**Data:**
- build specs in `builds/expansion-03/` (`make_specs.py` seed 8303, `fuse_specs.py`,
  `final02.py`, `long_intent.py`);
- datasets `artifacts/datasets/train-final-01-fused`, `expansion-03/capture-{a,b}-fused`
  and `train-final-02-fused/mixture.json`;
- prompts `expansion-03/bank-a5`, `bank-b-405`, `sessions`.

**Run:**
- `arms/s16-fused-long/seg-{1..24}`, with `intent.json`, `run-report.json`, journals and
  per-segment `screen/`, `panel/` and `screen2/` results;
- recipe `7f0f5124…`, executed source `8c3d9db5…`;
- admission in `logs/s16-preflight.json`;
- segment 2 was interrupted by an owner shutdown at 17,311 updates and resumed exactly from
  that state (`seg-2/interrupted-17311/`).

**Screen-v2:**
- workload and builder in `artifacts/benchmarks/screen-v2/`, and the `screen-v2` tier
  (commit `38311841`);
- establish result `screen-v2/runs/q4km-screen2/result.json`, frozen source `060915bf…`;
- registry records: target, Baseline and prior leader in `screen2/reference-*.json`; the
  new leader in `closeout/leader-ef068eba-screen2-reference.json`.

**Selection and release:**
- `step16/selection-{screen,panel,screen2}.json`;
- release runs `step16/{final,reversed}/runs/*` from pueue tasks 0–9;
- policy result `step16/policy-pooled-output/result.json`;
- prior-leader comparison in `step14/policy-pooled-output/result.json`.

**Leader references:**
- `artifacts/benchmarks/references/93bd262f…json` (Q4KM screen) and `6d64cd3f…json`
  (screen-v2);
- specs `runtime-v8-linux-01/specs/q4km-screen-leader-ef068eba.json` and
  `screen-v2/q4km-screen2-leader-ef068eba.json`.

## Artifact disposition

**Retained:**
- export `ef068eba…` and the seg-24 recovery state `d64e9a3c…`, the selected and
  recovery pair, in `arms/s16-fused-long/seg-24/output/`. Its dependency closure is
  `train-final-02-fused` and its components, plus executed source `8c3d9db5…`;
- all per-segment screen, panel and screen-v2 results, intents, journals and run reports;
- the retest evidence and the step 16 release runs and policy result;
- the screen-v2 inputs, references and frozen source;
- prior leader `ae6569f6…`, under `arms/s12-long/output/exports/` until the next leader
  decision, as the paired control for regression checks.

**Retired** through authenticated plans in `retirement/closeout-20261119-*`:
- the exports of segments 1–23 (12.5 GB), which are screened and superseded. Exact resume
  from intermediate points is lost; only the final state remains;
- the aborted step 16 run on the segment-9 export, a partial output;
- refused selection attempts `expansion-03/bank-a`–`bank-a4`, `bank-b-392` and `bank-b-412`;
- the per-campaign queue scripts (`scripts/queue*.sh`, `screen2/setup.sh`,
  `screen2/swap-v4.sh`) and `logs/queue.log`. GPU jobs now go through pueue (owner,
  2026-11-19).

**Earlier in the run:**
- FP16 tap files were retired under the owner's narrow-deletion go of 2026-11-18: fused-4,
  he4broad, bank-v8, `train-expansion-01` and captures A and B. Their receipts are in
  `retirement/*-receipt.json`.
- Eight-tap FP16 replay of those records now needs token replay;
  `train-final-02/mixture.json` (eight-tap) names retired files and is not loadable.
- `long-sessions-01` taps, manifests, tokens, labels and DEV captures were kept.

Intermediate training states were retired by Store retention after each segment
(`logs/state-retirement-fused-long.log`).

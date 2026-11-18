# Fused storage retest, then data expansion or the two-pass run

## Goal

Decide whether fused F16 storage is good enough to train on. If it is, shrink the
TRAIN set 8×, roughly double it with data aimed at our weakest scenarios, and train
longer. If not, run the two-pass release recipe on the existing FP16 data.

## Authorized

Nothing yet. Proposed 2026-11-18; waiting for the owner's go. Deleting the FP16
originals of `train-final-01` needs its own explicit OK.

## Done when

- The fused verdict is recorded under `results/` with the rule below.
- Pass: the expanded fused corpus is admitted and a two-pass run on it is screened
  against leader `ae6569f6…` and on the release scenarios.
- Fail: the two-pass FP16 run is screened the same way.

## State

### Fused retest (proposal)

Question: does training on fused F16 storage cost at most 1 acceptance point against
FP16 storage? One point of acceptance is about 1.1% request speed at depth 3. The
exported drafter and runtime are identical either way; only training inputs change.

- Arms: a third seed (8103) of the earlier pair, each a fresh 8,209-update calibrated run
  on the full TRAIN set: fused (`fuse-full-train-r20.json`, corpus rebuilt about 15 GPU
  minutes) and FP16 control. Recipe and source as the seed 8101/8102 arms.
- Evaluation: all six exports (three seeds × two storages; the four earlier exports are
  retained) on the 120-request Q4KM panel, candidate-only against saved runtime-v8
  references. The 60-request screen is 59/60 inside this panel.
- Rule, fixed before running: pooled paired source-group bootstrap of fused minus FP16
  acceptance over three seeds on the 120-request panel; pass if the 95% lower bound is
  ≥ −1.0 point. The screen-only pooled result is reported, not used to decide.
- Detectability: two seeds on 60 requests gave ±0.70; 120 requests and three seeds should
  give about ±0.45, so a true cost near −0.36 passes and one near −1 fails.
- Cost: about 2.5–3 GPU hours (two training runs, six panels, one fusion).

### Expansion if fused passes (proposal)

Current gaps, ours minus Baseline acceptance, final export, step 14 runs: writing −7.0 to
−7.5, thinking-on −6.7 (maths −7.1, code −5.8), conversation −5.5 to −5.9, reasoning
−4.3 to −5.9, documents −3.8 to −4.4 (−6.1 at 8–16K tokens), tool-call harness code −6.3,
sampled −4.3, greedy code about even, structured +27, agentic suite +17.

Current mix (4.49M supervised positions): reasoning 27%, conversation 21%, writing 21%,
single-turn code 18%, sessions 7%, documents 5%, structured 1%.

| Area | Records | Positions | Source |
| --- | ---: | ---: | --- |
| Thinking-on maths and code | 1,000 (600 / 400) | about 1.5M | MATH train (7.2K pool), CommitPack edit pool |
| Writing | 900 | about 1.05M | UltraChat writing (33K pool) |
| Conversation | 700 | about 0.6M | UltraChat multi-turn and oasst2 (97K pool) |
| Reasoning, thinking off | 300 | about 0.33M | MATH |
| Long documents | 300 | about 0.07M | QASPER (350 left) |
| Agentic long projects (11–20K start) | about 400 | about 0.17M | `plan-long-projects` pool: 162 prepared and unused, about 250 more from the 848 planned families not yet examined (30% yield) |
| Agentic short sessions | 381 | 0.12M | `train-expansion-01/sessions-2`, already generated |
| Single-turn code and tool calls | 600 | about 0.17M | CommitPack edits |
| Structured | 0 | 0 | already +27 over Baseline |
| **Total** | **about 4,600** | **about 4.0M** | |

All selection goes through the existing exclusion checks (heldout24, confirmation
sets, DEV identities and token prefixes, agentic project families). Sampling stays
proportional to positions, so one change is tested at a time.

Sequence and cost: fuse `train-final-01` (about 70 GB); retire the FP16 originals on the
owner's OK (rebuildable by token replay); capture in FP16 chunks, fusing and retiring each
chunk (about 12–14 GPU hours, about 50 GB fused); admit the new mixture; two passes over
about 8.5M positions (about 240,000 updates, 18–20 GPU hours).

Limits: a gap can reflect one-bit capacity rather than missing data. Every learning-curve
slice rose through 42,000 updates, but gains per category are not predicted. Only
the long-project pipeline yields new agentic sessions; more needs a re-plan beyond 1,850 families.

## Next

Owner: go or changes for the retest, the expansion, and deleting the FP16 originals.

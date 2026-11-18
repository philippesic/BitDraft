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
  on the 8,633-record `train-expansion-01` mixture the earlier arms used (not
  `train-final-01`): fused (`fuse-full-train-r20.json`, corpus rebuilt in about 15 GPU
  minutes, 56 GB) and FP16 control. Both new arms run one frozen source, executed source
  `8c3d9db5…`; the earlier controls ran `4bb823…`, which differs by fused calibration and
  disabled optional paths (Astra review 2026-11-18). Transfer to `train-final-01` is
  extrapolated.
- Evaluation: all six exports (three seeds × two storages; the four earlier exports are
  retained) on the 120-request Q4KM panel, candidate-only against saved runtime-v8
  references. The 60-request screen is 59/60 inside this panel.
- Rule, fixed before running: pooled paired source-group bootstrap of fused minus FP16
  acceptance over three seeds on the 120-request panel (`compare_screens.py` with matched
  `--control` lists); pass if the 95% lower bound is ≥ −1.0 point. A fail means "not
  demonstrated", and the two-pass FP16 run follows. Each seed is reported. Fresh 60-request
  screens of the two new arms give the supplementary screen result; it does not decide.
- Detectability: two seeds on 60 requests gave ±0.70. About ±0.45 on the panel is a scaling
  guess, not measured: the bootstrap resamples requests, not training seeds.
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
| Agentic long projects (11–20K start) | 152 sure, up to about 400 | 0.06–0.17M | `plan-long-projects` pool: 152 prepared and uncaptured (23 generated, 129 not), about 250 more if the 848 unexamined planned families keep the 30% preparation yield |
| Agentic short sessions | 381 | 0.12M | `train-expansion-01/sessions-2`, already generated |
| Single-turn code and tool calls | 600 | about 0.17M | CommitPack edits |
| Structured | 0 | 0 | already +27 over Baseline |
| **Total** | **about 4,600** | **about 4.0M** | |

All selection goes through the existing exclusion checks: heldout24, confirmation-v2
protection, DEV banks and workloads, DEV identities and token prefixes, agentic project
families and code content. Pools are ample historically; their exclusion counts are checked
fresh at selection. Sampling stays proportional to positions. Data and training length both
change, so gains are not attributed to the targeting itself.

Sequence and cost:
- Fuse `train-final-01` (about 70 GB).
- On the owner's OK, retire FP16 tap files only where token replay is qualified: the 6K
  bank and `train-expansion-01` (replay bit-identical on 40 records, including 22 from the
  6K bank and 9 Windows captures). That frees about 440 GB.
  - Keep `long-sessions-01` taps (120 GB) until replay is qualified on its records.
  - Keep every manifest, token stream, label, source record and helper, and all eight-tap
    DEV captures (DEV diagnostics read them).
  - Use authenticated retirement plans.
- Capture in FP16 chunks, fusing and retiring each chunk: about 12–14 GPU hours, about
  77 GB fused. Budget the peak (chunk plus fused output plus checkpoints plus reserve) per
  chunk.
- Consolidate chunk manifests into one fused corpus before admission: a mixture takes at
  most eight direct components and six are used.
- Two passes, with updates derived from the admitted `groups_per_pass` (about 240,000,
  18–20 GPU hours). Screen every 14,000 updates.
- Stop early on any inexact output, nonfinite loss, or two consecutive screens that fall.

Limits: a gap can reflect one-bit capacity rather than missing data. Every learning-curve
slice rose through 42,000 updates, but gains per category are not predicted. Only
the long-project pipeline yields new agentic sessions; more needs a re-plan beyond 1,850 families.

## Next

Owner: go or changes for the retest, the expansion, and deleting the FP16 originals.

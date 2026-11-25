# A 4-bit control trained on our data for 66 minutes matches the one-bit leader

The leader `ef068eba…` is the first drafter to pass the owner release policy against
Baseline. This check asks where that result comes from: one-bit feed-forward weights, or
our training data. Baseline's own 4-bit architecture, fine-tuned from the public drafter on
the same data, answers it.

The training data carries the result. One bit does not:
- after 21,000 updates (66 minutes) the 4-bit control runs as fast against Baseline as the
  leader does after 335,812 updates (23.6 GPU hours), on every release scenario;
- at equal data, order and updates the control accepts 7.8 points more drafts than the
  one-bit arm;
- the leader stays level on speed only because its rounds are cheaper, and about half of
  that saving is its smaller vocabulary table, which a 4-bit drafter could also use.

## Outcome

### Speed against Baseline

Fresh three-arm timing (target, Baseline, candidate) at depth 3 on runtime-v8, with greedy,
thinking-on and sampled repeated in reversed arm order and pooled. The leader's figures are
its own 2026-11-19 chain with the same specifications, host and source
([result](../fused-data-expansion/result.md)).

| Scenario | 4-bit control (95% lower) | One-bit leader (95% lower) | Acceptance: control / leader / Baseline |
| --- | --- | --- | --- |
| Agentic coding (60 tasks) | 1.110 (1.093) | 1.108 (1.091) | 85.4% / 81.7% / 63.4% |
| Greedy, long | 1.016 (1.007, merged strata) | 1.013 (1.002, merged strata) | 64.3% / 60.0% / 61.1% |
| Greedy, short | 1.011 (1.007) | 1.010 (1.000) | 53.4% / 49.6% / 52.3% |
| Sampled | 1.004 (1.002) | 1.011 (1.003) | 50.5% / 47.7% / 50.0% |
| Thinking-on | 1.010 (1.007) | 1.007 (0.998) | 67.8% / 63.0% / 66.6% |

The control passes the owner policy with merged strata and no shortfall. The leader passes
with two tolerated shortfalls. Every output was exact, and agentic completion was 37 of 60
for target, Baseline and both drafters. Baseline's own speed repeated within 0.2% between
the two days (agentic 131.0 tok/s on both), so the two chains are comparable.

Sampled is the one scenario where the leader is ahead (1.011 against 1.004).

### Acceptance at equal training

All arms use `train-final-02-fused`, seed 8101, the same grouping and 4-slot blocks, and a
cosine schedule over their own length. Checks are candidate-only against saved references;
acceptance is in %.

| Arm | Updates | Screen (60) | Panel (120) | Screen-v2 (60) |
| --- | ---: | ---: | ---: | ---: |
| Baseline, where the control starts | 0 | 60.64 | 53.35 | 62.70 |
| 4-bit control | 2,000 | 63.27 | 54.55 | |
| 4-bit control | 21,000 | 62.94 | 54.58 | 64.73 |
| One-bit | 2,000 | 50.67 | 43.71 | |
| One-bit | 21,000 | 55.04 | 46.75 | 55.66 |
| One-bit leader | 335,812 | 59.75 | 50.81 | 60.30 |

Paired source-group differences, in points:

| Comparison | Screen | Panel | Screen-v2 |
| --- | --- | --- | --- |
| Control − one-bit, 2,000 updates | +12.59 (10.85, 14.29) | +10.84 (10.24, 11.41) | |
| Control − one-bit, 21,000 updates | +7.90 (6.70, 9.10) | +7.84 (7.26, 8.39) | +9.07 (8.17, 9.91) |
| Control (21,000) − leader (335,812) | +3.19 (1.88, 4.53) | +3.78 (2.90, 4.57) | +4.43 (3.12, 5.65) |

Two trends matter for what comes next:
- One-bit keeps gaining with length (panel 43.7, 46.7, 50.8), so the gap to the control
  narrows from 10.8 to 7.8 points between 2,000 and 21,000 updates.
- The control is flat on ordinary requests between 2,000 and 21,000 updates (panel +0.04;
  −0.29, +0.37). Its further gain is in the agentic sessions: the DEV agentic first-choice
  rate rose from 57.2% to 68.9%.

### What one bit costs and buys

On short greedy requests the control accepts 3.7 points more than the leader and is 0.1%
faster. On agentic coding it accepts 3.6 points more and is 0.2% faster. The leader's
cheaper round is therefore worth about 3.5 acceptance points, close to the 4.3% of round
time predicted from the [round profile](../round-profile-baseline-s4/result.md): 0.16 ms
from the one-bit feed-forward kernel and 0.17 ms from scoring against our 341 MiB Q4_K
vocabulary table instead of the target's 497 MiB table. Only the first part needs one bit.

The control's file is 381 MB, Baseline's size. The leader's is 542 MB, because it carries
its own vocabulary table; its body alone is 184 MB.

### Control step size

Three 2,000-update pilots: step 3e-5 beat 1e-4 by 1.27 panel points (0.94, 1.60) and tied
1e-5 (+0.11; −0.11, +0.34), with the lowest DEV loss (2.428, against 2.464 and 2.440, from
2.490). The 21,000-update control used 3e-5.

### NVFP4 target

Candidate-only against saved NVFP4 references, exact outputs. Speed here is against a
saved Baseline run, so it is descriptive.

| Drafter | Screen | Panel |
| --- | --- | --- |
| Baseline | 58.23% | 53.02% |
| One-bit leader | 58.33% | 50.03% |
| 4-bit control | 62.24% | 54.37% |

Both drafters were trained on Q4KM target features and keep their standing on NVFP4.

## Scope and limitations

- **Length is not matched to the leader.** The control trained 16 times shorter. It is a
  floor for 4-bit, not a matched-length verdict; a 4-bit run of the leader's length was not
  made. The matched pairs stop at 21,000 updates.
- **One seed** per arm, and one step size for the full control. The one-bit recipe was
  tuned over several days; the control had three short pilots.
- **Reused DEV data.** Every panel was also used to select the leader. The protected
  confirmation sets and heldout24 were not touched.
- **The control starts ahead.** It begins at Baseline (53.35% on the panel); the one-bit
  arm begins from a calibrated one-bit conversion and has to recover.
- **Different vocabulary tables.** The control scores against the target's Q6_K table, as
  Baseline is served; the one-bit arms use our Q4_K conversion. A 4-bit drafter with a
  Q4_K table was not served.
- **Agentic coding** uses the tool-call convention of the training sessions, for both
  drafters. Transfer to other agent tools is untested.
- **Setup:** Qwen3.5-4B Q4KM, one RTX 5080, batch 1, native Linux, runtime-v8, depth 3.
  The agentic suite ran in one arm order.
- A policy report (a few CPU seconds at low priority) ran while the reversed-order request
  panel was starting. The single-order and pooled figures agree within 0.2%.

## Decision

- **Retain** the opt-in `"feed_forward": "plain"` training setting and its Baseline-layout
  export as the project's 4-bit control ([execution](../../docs/execution.md)).
- **No selection or promotion.** The leader `ef068eba…` stays the provisional acceptance
  leader among one-bit candidates; the control is a control.
- **Unresolved, for the owner:** whether the release should be one-bit at all. The
  measured speed advantage over Baseline is available to a 4-bit drafter in about an hour
  of training. What remains specific to one bit is the smaller body (184 MB against
  381 MB) and a kernel saving of about 2% of round time.
- **Proposed, not decided:** a 4-bit control of the leader's length, and a 4-bit drafter
  served with a Q4_K vocabulary table, to see how far ahead 4-bit ends; and test-list
  item 11, which is what would make the one-bit file smaller than Baseline's.

## Evidence and reproduction

All paths are under `artifacts/one-bit-attribution-01/`. Compact measurements, paired
intervals, policy summaries and export identities are in [result.json](result.json),
written by `scripts/assemble.py`.

**Build:** commit `4b50428d` (merged as `1bd92b47`), executed source `94852e01…` in
`executed-sources/`. At zero updates its export is byte-identical to the Baseline file
`83586a63…` (`builds/plain-control/update0-identity.txt`), so the control's starting point
is Baseline's saved reference.

**Arms** (`arms/NAME/` with `intent.json`, `run-report.json`, journal and check results):

| Arm | Recipe | Executed source | Export |
| --- | --- | --- | --- |
| `plain-21000-3e-5` | `0d93d0f4…` | `94852e01…` | `3d006c22…` |
| `plain-2000-3e-5` | `e7cb60ae…` | `94852e01…` | `77907db8…` |
| `plain-2000-1e-5`, `plain-2000-1e-4` | in `result.json` | `94852e01…` | `c7035617…`, `97751156…` |
| `w1-21000` | `9dc9621b…` | `8c3d9db5…` | `8c0e73aa…` |
| `w1-2000` | `de956819…` | `8c3d9db5…` | `9506230d…` |

The one-bit arms are the leader's recipe with only the schedule length changed.

**Checks:** screen and panel under benchmark source `8180f208…`, screen-v2 under
`060915bf…`, each with the saved runtime-v8 references. The first screen-v2 attempt for
`w1-21000` was refused because it was pointed at the wrong frozen source; the rerun is the
recorded one.

**Timing:** `arms/plain-21000-3e-5/release-final/` and `release-reversed/` (specifications
from the 2026-11-19 chain's builder with the control as candidate), policy reports in
`policy-final-output/` and `policy-pooled-output/`. The leader's single-order report was
recomputed from its saved runs into `leader-final-only/`.

**NVFP4:** `arms/leader-nvfp4/` and `arms/control-nvfp4/`.

**Commands:** `scripts/train.sh`, `scripts/arm.sh`, `scripts/checks.sh`,
`scripts/release.sh`; intents from `scripts/make_intent.py` and `scripts/make_plain.py`.

## Artifact disposition

**Retained** (41 GB in all), owner Philip:
- the control export `3d006c22…` and the one-bit exports `8c0e73aa…` and `9506230d…`, with
  every arm's intent, journal, run report, check results, timing runs and policy reports;
- the build deliverables and both executed sources;
- each arm's final training state (6.4 GB each, 38 GB together) and the three pilot
  exports. Plain-table states cannot resume, so the four control states have no use; they
  were retired through Store retention on 2026-11-25 with owner approval (receipt
  `receipts/control-disk-retirement-20261125/`). The three pilot exports remain.

**Dependencies:** `train-final-02-fused`, the public BF16 drafter, the Baseline file, the
target, runtime-v8 and the saved references. None was changed.

**Retired:** the four plain-control state stores (2026-11-25, about 27 GB); exact resume of
those arms is no longer available. No intermediate states were written.

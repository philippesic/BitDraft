# Training the attention and fusion tables added about one acceptance point in one-hour runs; DEV did not confirm it

Test-list item 10, 2026-11-16. The candidate's 4-bit attention and fusion tables are copied
from the source drafter and stay frozen while the one-bit feed-forward trains. This tests
whether letting them adapt to the one-bit feed-forward recovers acceptance.

- At the small table step size (1e-5) screen acceptance rose by +1.3 points (+0.6, +2.0)
  over the frozen control, which meets the preflight's bar (at least +1.0 with the interval
  excluding zero).
- At ten times that step size it rose by +0.9 (+0.1, +1.7), below the bar.
- DEV did not move beyond noise: exported top-1 76.1% and 76.1% against the control's 76.9%,
  survival 55.0% and 57.5% against 56.7%.
- All three arms saw identical data groups.

## Outcome

Three fresh runs from the same initialization, corpus, seed and order, 17,800 updates each,
`Q4_0` tables in all arms. Screen: 60-request Q4KM DEV screen, runtime-v8, native Linux,
paired source-group bootstrap against the frozen arm (2,000 draws). DEV: 120 windows,
top-1 pooled over positions 1–3, survival through position 3, values after Q4_0 export.

| Arm | Tables | DEV top-1 | DEV survival | Screen | Δ screen (95%) |
| --- | --- | ---: | ---: | ---: | --- |
| t0 | frozen | 76.9% | 56.7% | 50.14% | — |
| t1 | trained, step 1e-5 | 76.1% | 55.0% | 51.45% | +1.31 (+0.55, +2.03) |
| t2 | trained, step 1e-4 | 76.1% | 57.5% | 51.01% | +0.87 (+0.08, +1.71) |

Before export rounding the trained tables scored 77.2% / 57.5% (t1) and 76.4% / 56.7% (t2):
rounding them to Q4_0 costs t1 about one DEV top-1 point. DEV curves at six readouts are in
[result.json](result.json); the trained arms led the control at the first two readouts
(t1 69.2% against 65.3% top-1 at update 2,966) and converged by the end.

Gradient norms were about three times the control's in both trained arms, so the shared clip
acted more often on them. Peak allocation was 14.1 GiB (trained) and 11.5 GiB (frozen). A
trained-table update took 0.19 s against 0.17 s, and each run took about 57 optimizer-minutes
(trained) or 50 (frozen).

## Scope and limitations

- One run and one seed per arm. Data-order noise on the screen was 0.06 points in the feature
  screen, so +1.3 is beyond it, but the paired interval spans +0.6 to +2.0.
- All arms used `Q4_0` tables, which cost 0.13 ms per round over Baseline's `Q4_K` format
  (item 8, about 1.7% of request speed). The current recipe uses Baseline's tables; trained
  tables cannot be copied and need their own `Q4_K` conversion, which the pinned GGUF package
  lacks. At `Q4_0`, +1.3 acceptance points roughly offsets that speed cost.
- Activations were capped at 1.75 GiB (about half the windows per update of the S4 recipe), so
  these arms are not comparable with earlier runs.
- The first small-step attempt ran out of GPU memory at update 1,495: 13.4 GiB allocated plus
  1.5 GiB of fragmented reserve exceeded the 15.47 GiB device while the display holds 0.24 GiB.
  Both trained arms then ran unchanged with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (allocator only; recipe and data identical). The large-step attempt stopped at about update
  150 before reaching that group.
- Only acceptance on the reused DEV screen; no speed measurement of a trained-table export.

## Decision

Proposed, for the owner: the small-step arm cleared the preflight bar on the screen, and both
trained arms point the same way. DEV does not confirm it, and the gain is about the size of the
`Q4_0` speed cost. A longer confirmation run is worth it only together with a `Q4_K` export of
trained tables. Otherwise keep the frozen `baseline-k` tables.

## Evidence and reproduction

- **Code.** Main `7af81924`, executed source `3587d764…`, intent setting `trained_tables`
  ([execution](../../docs/execution.md#trained-tables)); no code change.
- **Inputs.** Initialization `2f852a41…`, corpus `3fbc0135…`, source `18cefa06…`, vocabulary
  `a68f55d9…`, DEV specification `54afbd49…`; screens bound to the
  [native Linux references](../native-linux-references/result.md).
- **Runs.** `artifacts/training/trained-tables-01/`: `preflight.md`, `arms/*/intent.json`,
  journals and run reports, `screens/`, `summary.json`, `scripts/` (intent generator, runner,
  screen specification builder, `analyze.py` that rebuilds the table).

## Artifact disposition

Retained in `artifacts/training/trained-tables-01/`: every arm's intent, journal, report and
error log; the final state and export of t0, t1 and t2; the pilot's state and export; the
interrupted t2 attempt's state and export; all screens; the frozen executed source. The failed
t1 attempt left no state. The pilot and interrupted-attempt states (about 19 GB) were retired
through Store retention on 2026-11-25 with owner approval (receipt
`artifacts/receipts/control-disk-retirement-20261125/`); their exports and screens remain.
The final states of t0, t1 and t2 (29.7 GB) were retired the same way on 2026-11-27 with owner
approval (receipt `artifacts/receipts/control-disk-retirement-20261127/`); exact resume of these
arms is no longer available, and their exports and screens remain.

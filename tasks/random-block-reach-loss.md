# Random block sizes and reach-weighted positions

## Goal

Train HE4 so it is not locked to one draft depth: random block sizes per update and
loss weights proportional to how often each position is reached. Compare against fixed
16-slot and fixed 4-slot training at equal optimizer time, scored at depths 2, 3 and 4.

## Authorized

- Assigned by Philip on 2026-11-13 (design approved the same day) to the implementation
  agent in worktree `t3-af01b17f`, branch `t3/random-block-reach-loss`.
- GPU: 3.5 hours in total through the project lease, shared with a draft-depth study and
  a runtime-fix build. Wait when the lease is held; never stop another process; do not
  hold the lease between runs. Stop and report rather than exceed the budget.
- Data: frozen 6K TRAIN corpus, 120 fixed DEV diagnostic windows, the 60-request Q4KM DEV
  screen. Never heldout24 or its aliases.
- Inputs read-only by absolute path from `/home/philip/BitDraft/artifacts`. Outputs in this
  worktree's ignored `artifacts/training/block-test/`. Keep the best arm's state and export
  in a new folder under `/home/philip/BitDraft/artifacts/training/`; retire the other arms'
  states through Store retention. No push or merge.

## Done when

S4 and SR are trained at arm D's optimizer time (1,814.5 s), every arm is screened at
depths 2, 3 and 4 with matched controls, DEV per-position top-1 and prefix survival are
extracted, the result page, FINDINGS entry and docs are updated, the kept state verifies
from the merged code, and other raw outputs have an explicit disposition.

## State

- Implementation committed (`e10976f4`): `blocks` intent setting, block-aware memory rule,
  per-group block sizes in the sampler, reach weights refreshed at each DEV diagnostic and
  stored in tensor payload 5.
- Planned GPU runs, each under the lease, outputs under `artifacts/training/block-test/`:
  1. Pilots (about 10 minutes): S16 for 100 updates to confirm the fixed 16-slot path
     reproduces arm D's journal; S4 and SR for 400 updates to measure update time and set
     their update budgets. Disposal: pilot states and exports retired after closeout.
  2. S4 and SR full runs (about 40 minutes each). Disposal: the better arm's final state and
     export move to the main checkout; the other's are retired through Store retention.
  3. DEV diagnostics of each final state at 3, 4, 5 and 16 slots (a few minutes).
  4. Screens: S4 and SR candidate-only at depth 2; target, Baseline, S16, S4 and SR at
     depths 3 and 4. Disposal: screen outputs retained as result evidence.
- S16 control: arm D (`artifacts/training/recipe-test-arm-d-01`, main checkout), reused if
  the pilot reproduces its journal.
- GPU hours used: 0.

## Next

Run the pilots when the GPU lease is free.

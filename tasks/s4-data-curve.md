# S4 data curve on the 6K corpus plus new old-corpus prompts

## Goal

Learn whether training data limits S4 and where extra training stops paying: continue S4
for about five GPU hours on the 6K corpus plus 6,000 new natural prompts from the original
15,647-record corpus, with a native screen every 30 optimizer-minutes and DEV diagnostics.

## Authorized

- Owner request (Philip, 2026-11-13), executed by this branch's agent.
- GPU: one campaign of about five hours, hard limit six GPU hours including screens and the
  final 120-request Q4KM panels of the final and best-screen states. Start only after the
  GPU and project lease have been idle for 10 minutes; take the GPU only through the lease.
- Data: TRAIN only. The old corpus is read in place and never modified or retired. DEV
  prompts only for evaluation; never heldout24 or aliases.
- Outputs: `/home/philip/BitDraft/artifacts/training/s4-data-curve-01/` (campaign, specs,
  frozen source, logs) and `/home/philip/BitDraft/artifacts/datasets/s4-data-curve-mixture-01/`
  (selection, subset index and intent, mixture). Keep the final and best-screen states with
  exports; retire other intermediate states through Store retention at closeout.

## Done when

The result page with chartable curve data, the FINDINGS entry and artifact disposition are
committed; kept states load from main's code; intermediate states are retired with receipts.

## State

- Branch `t3/8045f98e`, worktree `/home/philip/.t3/worktrees/BitDraft/t3-8045f98e`.
- Executed source `f88948bd…` (commit `2dce22af`), frozen in the run folder's
  `executed-sources/`; launched with `--frozen` so outputs land in the main checkout.
- Mixture `34a955a8…`; recipe `c59b855e…`; intent `440ac55b…`. Continue from S4 state
  `6977443f…` (update 8,209): 72,000 more updates, 2% warm-up, cosine to zero at 80,209,
  optimizer-time stop at 18,213.2 cumulative seconds, screen every 1,800 optimizer seconds.
- CPU training admission and campaign preflight passed (`logs/` in the run folder).
- Waiting for the GPU: another agent's runtime-v7 work holds the lease.

## Next

Launch the campaign detached once the GPU has been idle for 10 minutes; check the journal,
losses and screens every 30 minutes until it completes.

# Make the one-bit leader's file smaller than Baseline's (test-list item 11)

## Goal

Ship the leader `ef068eba…` with only the shortlisted rows of its vocabulary table, so the
file drops from 542 MB to about 255 MiB (Baseline 364 MiB) and uses about 260 MiB less VRAM,
with the same drafts and speed as the full-table file served with the shortlist.

## Authorized

Owner go on 2026-11-20 (00:00 PST): the four model steps the lead listed, bounded to work
that helps now and no new feature set, builds by an Opus subagent. The lead read that go as
the 4-bit control alone; the owner confirmed on 2026-11-20 (09:40 PST) that it covered every
listed step, item 11 included. This task is item 11. GPU work goes through the `gpu` queue.

Not covered here: the confirmation sets stay unspent until the owner names the drafter they
are spent on.

## Done when

- The shortlist-only file and its runtime pass, per request on the 60-request screen and the
  120-request panel: identical tokens, drafts proposed and drafts accepted against the
  full-table shortlist file on runtime-v9; existing files unchanged on the new runtime.
- File size, VRAM and speed against Baseline (as released and with the same shortlist) are
  recorded in a result entry, FINDINGS and the test list.
- The build is merged through the gate and raw outputs have a disposition.

## State

Owner: Philip. Lead worktree `/home/philip/.t3/worktrees/BitDraft/shortlist-only-table`,
branch `t3/shortlist-only-table` from main `f219d446`. Campaign folder
`artifacts/shortlist-only-table-01/` (brief in `briefs/`, build outputs in `builds/`, runs in
`runs/`; purpose: item 11 qualification; inputs: the leader export, the target,
`train-final-02-fused` labels, runtime-v9; disposal at closeout, keeping the small file, the
runtime and compact results). Timing marker: `artifacts/shortlist-only-table-01/TIMING-ACTIVE`.

2026-11-20 09:55 PST: Opus build delegated (shortlist-only file from `vocabulary-variants`,
runtime edit after runtime-v9's five, a CPU proof that single-row conversion of the target's
Q6_K rows reproduces our Q4_K rows, runtime-v10). The child may run short functional GPU
checks through the queue. No measurement yet.

## Next

Wait for the build report. Then freeze its source, run parity (runtime-v9 against
runtime-v10) and the five-arm screen and panel, and decide from those whether the
five-scenario timed chain is worth running on the small file.

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

2026-11-20: the Opus build is in (branch `t3/shortlist-only-vocabulary-table`, head
`dc296a41`, not yet merged; frozen source `29115117…` in `builds/executed-sources/`). The small
file `156b4f72…` is 284.0 MB (270.9 MiB) against Baseline's 381.5 MB: 68,728 Q4_K rows, one per
distinct TRAIN label. All 248,320 rows convert alone to the export's bytes.

Qualification passed on the screen and the panel: runtime-v10 reproduces runtime-v9 per request
for the five full-table arms, and the small file reproduces the full-table shortlist file. The
build's `summary.py parity` reported a panel failure because it compares the first block by file
name and the two runs have different block layouts; `scripts/parity_by_request.py` matches on
the request and passes all 120. Panel: small file / full-table shortlist file 1.032
(1.030, 1.034), / Baseline with the same shortlist 1.030 (1.021, 1.040), / Baseline as released
1.076; GPU memory 340 MiB lower. The 3% over the full-table shortlist file is unexplained
(drafting time fell only 0.05 ms per round).

Timed chain: preflight in `logs/chain-preflight.md`; pilot slices passed (agentic 1.135 against
Baseline, exact outputs, shortlist attached, no request-row overflow). The sampled runs needed a
sampler check bound to runtime-v10 (`release-final/verifier-v10/`, passed). The four-arm chain
(target, Baseline, Baseline with the sidecar, small file) started 11:31 PST in both arm orders,
watched as jobwatch job `item11-chain2`.

Owner decision 2026-11-20 (12:53 PST): skip the reversed pass and report after agentic. The
chain runs one arm order only; `SKIP-reversed` in the campaign folder makes the chain skip it.

## Next

When the chain ends: policy report against Baseline as released and against Baseline with the
shortlist, merge the build through the gate, then close with a result entry.

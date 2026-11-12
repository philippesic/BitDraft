# Capture the frozen 6K TRAIN bank in FP16

## Goal

Generate training captures for the existing 6,000-prompt TRAIN bank. Philip cancelled
the extra hybrid training and the proposed 12K expansion. The immediate handoff is a
cleaned-up training setup with the 6K capture job verified running.

## Authorized

RTX 5080 FP16 acquisition for exactly the existing bank, using Qwen3.5-4B Q4KM and
the qualified eight-tap, up-to-15 capture contract. Reuse compatible validated captures
without copying arrays. Preserve DEV/heldout roles, original teacher data, original
final220095 and selected states. Reclaim obsolete large artifacts with evidence and
recoverability checks. No training, bank expansion, new remote or unrelated job changes.

## Done when

All 6,000 unique TRAIN IDs have valid captures, each passing the product loader, and
the complete corpus manifest reconciles exactly with the frozen bank. Preserve failures
and execution/source receipts. Close out data evidence and this task when acquisition
actually completes; starting the job is not a completed dataset.

## State

Owner: this thread's existing RTX5080 capture allocation; project branch `main`.
Artifact root `/home/philip/binary-eagle-artifacts`. The detached controller under
`datasets/train-bank-v8-fp16-01` reauthenticated the completed 2,000-record supplement
and is generating the remaining 4,000 through public `benchmarks capture` entry points.
All 2,000 reused prompts exactly match the bank's messages, tools, token identity,
output cap, group, category and role. Target, FP16 storage and supervision policy match.
`plan.json`, `launch-1.json`, `controller.json`, `status.json`, `events.jsonl` and
`controller.log` describe ownership and actual progress. The Linux shard is active and
has published validated records. Verify birth/boot before recovery.

The first new shard has 1,789 records in `linux-captures`; the second has 2,211 in
`windows-captures`, linked to `/mnt/c/binary-eagle-artifacts/train-bank-v8-fp16-01/captures`.
Separate storage ceilings are about100 and122 GiB of raw features, with measured room
on each filesystem at launch. This is dataset planning, not a repository disk limit.
The original 102 GiB supplement remains in place. Final `corpus.json` will reference
all 6,000 records across these locations without copying them; `result.json` is published
only after both shards complete and all product validation receipts reconcile.

No production hybrid campaign started. The short qualification worker stopped after
2.057 optimizer seconds; its controller was interrupted, and its evidence is retained.
`training/two-hour-request-cancelled.json` records cancellation. Older schedules remain
disabled. Obsolete unselected states, intermediate checkpoints, old downloads and a
conversion environment were retired with receipts under
`/home/philip/binary-eagle-evidence/retirement-20261111/six-k-capacity`.

## Next

1. Inspect controller identity, actual GPU clients, `status.json`, shard `progress.json`
   and logs. Preserve healthy work. No training is authorized by this capture task.
2. For a routine failure, preserve it and resume the exact shard specification through
   `uv run --locked --no-sync --offline python -m benchmarks capture SPEC OUTPUT`.
   Completed matching records are verified and reused. Do not launch competing jobs.
   The one-time controller's recorded execution can also resume the same plan after
   confirming its previous owner is dead; completed shards are skipped.
3. At completion, verify the exact6K manifest, role boundaries, validation receipts and
   storage accounting; publish compact evidence and close this task.

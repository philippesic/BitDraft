# Expand the frozen 6K TRAIN bank toward agentic coding

## Goal

Add a TRAIN-only expansion to the frozen 6K FP16 bank, combined with it through a
`corpus-mixture-v1` manifest. Priorities: long agentic tool sessions in the live harness
format, other coding with longer answers, writing and conversation, then breadth
(competition maths, more long documents, longer answers).

## Authorized

- Philip approved the expansion (2026-11-13). Tranche 1 may use at most 200 GB of new
  FP16 captures and 8 GPU hours; tranche 2 is forecast only.
- New outputs go under `/home/philip/BitDraft/artifacts/datasets/train-expansion-01/`.
  Existing main-checkout files, the 6K bank identities, DEV cohorts, workloads and saved
  references stay unchanged. The 314 GB FP32 corpus must not be touched.
- GPU work starts only after 10 minutes without other GPU clients or lease holders,
  through the project lease, never stopping another process. No push or merge.

## Done when

The expansion capture passes product validation and Corpus admission, a mixture manifest
with the 6K corpus is admitted on CPU, and the result page, FINDINGS entry, data docs and
artifact disposition are written.

## State

- Owner: implementation sub-agent on branch `t3/3d1f358a`.
- Multi-span session supervision is implemented (`supervised_spans`, native `spans`
  requests, `capture_supervision`). Session windows end at each label/next-token
  disagreement because re-rendered history retokenizes the model's own output.
- Helpers are built against runtime-v6 in `train-expansion-01/native-helpers`.
- Sources are stored read-only in `train-expansion-01/sources`. Non-session quotas
  (`specs/quotas-nonsession.json`) select 1,759 prompts with a 193 GB maximum forecast.
  Legacy Dolly/Magicoder rows are excluded: they come from the old corpus that the S4
  data-curve mixture samples.
- GPU allocation began 2026-11-14 01:57. `sessions/` holds 640 complete TRAIN sessions
  (433 passed their checks); `bank/` selected 2,399 prompts (640 session records with
  201,729 span positions); `capture-a/` (1,844 IDs: all sessions, then an interleaved
  non-session prefix) runs from snapshot `c2f0a0c0…`. First session records: 98.4% of
  span labels equal the next history token and every turn ends with its end-of-turn label.
- `specs/after_capture_a.sh` then runs `specs/chain3.py` (snapshot `75e90cfa…`, log
  `logs/chain3.log`): 615 more sessions (`skip` 640), a session-only `bank-2/` sized to
  the room left under 200 GB, `capture-c/`, the non-session remainder `capture-b/`, and
  the mixture manifest with its CPU admission.
- Raw outputs: session transcripts and captures are TRAIN inputs retained until a later
  result retires them; executed source snapshots stay beside them.

## Next

Verify chain3's outputs, then write the result, FINDINGS entry and tranche-2 forecast.

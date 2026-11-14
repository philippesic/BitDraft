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
- Sources are stored read-only in `train-expansion-01/sources`. A 12-task session pilot
  waits for an idle GPU (`specs/sessions-pilot.json`, log `logs/sessions-pilot.log`).
- Raw outputs: session transcripts and captures are TRAIN inputs retained until a later
  result retires them; executed source snapshots stay beside them.

## Next

Measure the pilot's session lengths, span agreement and time, then fix session and
non-session quotas, run the main sessions, select, and capture in shards.

# Retire historical archives and qualification outputs

## Goal

Preserve useful historical measurements and source in the results format, retain inputs
needed for active development, and remove obsolete archive and qualification payloads.

## Authorized

Philip authorized this audit, evidence extraction and deletion in the current conversation.
Scope is historical evidence and artifacts, especially `artifacts/archive/` and
`artifacts/qualification/`. Preserve current datasets, model inputs, original220095 and
actual active development dependencies. No training, capture or new GPU benchmark.

## Done when

Compact evidence is verified against source; every retained large artifact has a concrete
development dependency; obsolete payloads are retired through their maintained owners;
results, findings and current guidance agree; checks pass and changes are pushed.

## State

Owner: current agent, `/home/philip/BitDraft`, existing branch. Working tree initially clean.
The archives include about170GiB of losslessly compressed Windows raw outputs; qualification
holds about120GiB including closed test checkpoints. No GPU client or training/capture
process was observed. Receipts and executed extraction scripts will live under
`artifacts/receipts/artifact-closeout-20261113/`; compact support will live under
`artifacts/archive/development-history/`. Their purpose is historical measurement/source
recovery after retiring raw tensors, old weights and redundant run trees.

## Next

Authenticate archive members while extracting compact support. Resolve current input and
saved-control dependencies before retiring qualification stores and other raw files.

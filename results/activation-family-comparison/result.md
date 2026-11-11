# Learned-small G4/D8 led trained activation endpoints but trailed baseline

## Outcome

In the November 4 modern 4B/RTX 5080 campaign, learned-small G4/D8 was faster on
complete heldout requests than the other retained trained activation-family endpoints,
but remained slower than external DFlash Q4_K_M. G4/D8 means four-bit gate/up inputs
and eight-bit down inputs with binary feed-forward weights; it is not a uniform
whole-model four-bit format. These learned-small endpoints are not HE4. Clipping and training choices also differ
across these endpoints; this is not an isolated causal estimate of activation width.

## Scope and limitations

the recorded campaign's recipes and training budgets, one target/hardware track
and a cohort now consumed. It does not prove that lower precision cannot recover under
another recipe or establish a fresh heldout result in this repository.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[precision summary](https://github.com/philippesic/binary-eagle-decoding/blob/7d9ab32066584edbb715ddbce38d0386b2efb0d6/experiments/precision-campaign-20261104/summary.json).

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. Missing reproduction details must be recovered before artifact
retirement; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

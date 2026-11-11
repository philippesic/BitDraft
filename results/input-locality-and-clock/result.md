# Remote input materialization dominated the audited HE4broad training clock

## Outcome

The early audit found distinct shuffled windows rather than small-dataset cycling and no observed optimizer discontinuity. Mean materialization was about 0.85 seconds/update versus 0.14 seconds model work. In a separate 32-capture replay, reads consumed 97.51% and SHA-256 hashing 0.55% of materialization time.

## Scope and limitations

November 10 investigation snapshot; the replay ran concurrently and is not a matched storage benchmark. Later localization changed storage and loading. Three repeated DEV prompts cannot establish a general learning plateau. The wider training investigation remains unfinished.

## Decision

Retain the input-locality lesson and separate clocks; no model-capacity conclusion or new allocation.

## Evidence and reproduction

- [he4broad-20261110/setup-audit.md](../sources/experiments-he4broad-20261110-setup-audit-485d9fd6.txt)
- [he4broad-20261110/localization.md](../sources/experiments-he4broad-20261110-localization-b1bccc40.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

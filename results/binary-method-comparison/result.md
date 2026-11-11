# No tested binary method beat the retained W1A8 pipeline

## Outcome

No. A1 XOR requests ran 137.95 tok/s against 197.45 for A8. A8-trained weights run with
an A1 activation override are not trained W1A1. The tested binary-MMA PTX lowered to
integer MMA.

## Scope and limitations

Qwen3 F16 with DSpark D5, two repetitions. No modern DFlash inference and no
general hardware conclusion.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[experiments/binary-methods-20261104/README.md](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/binary-methods-20261104/README.md) [experiments/binary-methods-20261104/analysis.json](https://github.com/philippesic/binary-eagle-decoding/blob/7cbc9f1833841c7f394789a209b23b1ad5a77af2/experiments/binary-methods-20261104/analysis.json)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. Missing reproduction details must be recovered before artifact
retirement; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

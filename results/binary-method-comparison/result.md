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

[experiments/binary-methods-20261104/README.md](../sources/experiments-binary-methods-20261104-readme-c03c241f.txt) [experiments/binary-methods-20261104/analysis.json](../sources/experiments-binary-methods-20261104-analysis-json-dd9413a4.json)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. [Evidence access](../../docs/evidence.md) describes archive recovery and remaining
raw-input limits; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

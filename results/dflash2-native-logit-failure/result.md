# Both original DFlash2 native paths produced nonfinite logits

## Outcome

No, both produced nonfinite logits. The accepted fallback replaced exactly five down
projection matrices with released BF16 values converted to F32 and left the other 76
tensors unchanged. A bounded search found no matching 9B release.

## Scope and limitations

a mixed Q4 and F32 result, not a fully Q4 DFlash2. The search was bounded and
historical.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[experiments/dflash2-smoke-20261106/report.md](../sources/experiments-dflash2-smoke-20261106-report-7e0e0aa5.txt) [experiments/dflash2-smoke-20261106/results.json](../sources/experiments-dflash2-smoke-20261106-results-json-939ce214.json)

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. [Evidence access](../../docs/evidence.md) describes archive recovery and remaining
raw-input limits; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

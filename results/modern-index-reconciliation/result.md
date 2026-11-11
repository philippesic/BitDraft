# The retained modern index reconciles within metadata scope

## Outcome

Yes. The selected modern index reconciles 15,647 records, 1,568,435 windows and
1,786,275 unique supervised positions; 87 records have no windows. Prior use is
historical training for 9,813 records, unknown for 5,658 and unobserved for 176.
Intake on the real metadata exposed and fixed two port errors: first-anchor and
end-of-generation handling, and the original sampler order.

## Scope and limitations

metadata and source-chain reconciliation only. No raw-feature hashing, teacher
replay or data adequacy. The F005 recovery difference is still open.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[complete intake](../../evidence/modern-index-complete-intake.json),
[pinned intake](../../evidence/modern-index-qualified-intake.json),
[earlier refusals](../../evidence/modern-index-intake.json),
[snapshot refusal](../../evidence/modern-index-snapshot-intake.json),
[transport](../../evidence/metadata-snapshot-transport.json),
[initializer header](../../evidence/he4-initializer-header.json).

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. [Evidence access](../../docs/evidence.md) describes archive recovery and remaining
raw-input limits; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

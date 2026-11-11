# Repository overhaul for agent work

## Goal

Make the correct path the easy one for agents: short instructions, one home per fact,
fast feedback, useful diagnostics and rules that earn their cost.

## Authorized

Philip authorized routine overhaul repairs and verification without repeated approval.
Preserve the full commit gate, no hosted CI, scientific semantics and existing protections.
Safeguard simplification is approved: remove redundant third-party tool proofs and
manual policy checksum/configuration copies, retaining actual checks and project proofs.
Audit documentation for port/overhaul leftovers and historically useful evidence.
Bring remaining scientific and operational choices to Philip with concrete examples.
No GPU work, training, artifact transfer, remote creation or publication is authorized.

## Done when

Routine defects are repaired and verified, remaining policy and operations decisions
are resolved or explicitly deferred, and matched fresh-agent exercises assess the final
workflow without claiming general productivity from a few trials.

## State

Owner: Codex in the current conversation, branch main.

Completed: instructions and docs reorganized; task template enforced; quick partial
check and full commit gate; executable local hook; no hosted CI. Diagnostics now use
the fixes table as their single repair source, with repair coverage for every registered
rule and Python comment locations. Doctor detects missing and non-executable hooks.
The validation policy points to tracked registries rather than an external scratch
proposal; CPU verification no longer prints a blanket approval request. README explains
the check-only setup boundary and host numerical interpreter configuration.

Checks and limitations: [diagnostic probes](../evidence/overhaul-diagnostics-20261110.json)
record three actual rejected mistakes and hook detection in a disposable clone. This
first probe also exposed an inadequate hook repair message. The [verified probes](../evidence/overhaul-diagnostics-20261110-verified.json)
confirm the corrected repair messages. Existing fresh-agent exercises are linked in
[port](port.md); they are not a matched before/after comparison. The [clean check-only setup](../evidence/overhaul-setup-20261110.json) passed
offline using cached wheels. Numerical environments were reused; a complete numerical
reinstall was not verified. No remote is configured.

Safeguards simplified: third-party tools still check the full source; 92 project rules
retain bad/control proofs. Direct versions and settings come from pyproject.toml;
the installed dependency closure is checked against uv's locked export with platform
markers. Manual policy and specimen checksums and dead policy metadata were removed.
Frozen source/data evidence hashes remain. Ordinary check imports are permitted only
in tools and tests. Documentation audit is in progress.

Remaining choices for Philip:

1. Future architecture. Runtime, comparison, selection and discovery contracts precede
   complete serving implementations. Recommend retaining scientific contracts, but
   reviewing speculative discovery/layout restrictions when that component is designed.
2. Numerical environment. Choose whether a clean standalone CPU installation or reuse
   of host numerical libraries is the supported setup. Recommend a reproducible CPU
   environment as the default, with host reuse explicit. Existing environments stay intact.
3. Operations. Remote name/visibility and license, independent artifact backup, and
   shared-GPU coordination remain in [agent operations](agent-operations.md) and
   [port](port.md). No repository resource manager or agent disk cap is proposed.
   Product capacity checks remain intact; their requirements are a separate decision.

## Next

Present the remaining choices with recommendations. After the chosen policy changes,
compare fresh agents on identical bounded tasks against the baseline recorded in the
[setup audit](../evidence/overhaul-setup-20261110.json) and retain failures, interventions and limitations. Do not start the product
port's pending scientific stages from this overhaul task.

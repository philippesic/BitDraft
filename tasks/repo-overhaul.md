# Repository overhaul for agent work

## Goal

Make the correct path the easy one for agents: short instructions, one home per fact,
fast feedback, useful diagnostics and rules that earn their cost.

## Authorized

Philip authorized routine overhaul repairs and verification without repeated approval.
Preserve the full commit gate, no hosted CI, scientific semantics and existing protections.
Bring changes to guarantees and operational choices to Philip with concrete proposals.
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

Remaining choices for Philip:

1. Safeguard simplification. The corpus has 34 externally executed tool specimens
   (6 Ruff, 23 mypy, 1 Vulture, 4 Deptry), plus project-authored proofs. Keep actual
   lint/type/dependency stages. Review removing redundant tool-behavior specimens
   while retaining project-specific integration checks, and replacing manual policy
   digests/config copies with checks of the intended invariants. These alter the
   safeguards and should be reviewed before implementation. Keep the full hook meanwhile.
2. Future architecture. Runtime, comparison, selection and discovery contracts precede
   complete serving implementations. Recommend retaining scientific contracts, but
   reviewing speculative discovery/layout restrictions when that component is designed.
3. Numerical environment. Choose whether a clean standalone CPU installation or reuse
   of host numerical libraries is the supported setup. Recommend a reproducible CPU
   environment as the default, with host reuse explicit. Existing environments stay intact.
4. Operations. Remote name/visibility and license, independent artifact backup, and
   shared-GPU coordination remain in [agent operations](agent-operations.md) and
   [port](port.md). No repository resource manager or agent disk cap is proposed.
   Product capacity checks remain intact; their requirements are a separate decision.

## Next

Present the remaining choices with recommendations. After the chosen policy changes,
compare fresh agents on identical bounded tasks against the baseline recorded in the
[setup audit](../evidence/overhaul-setup-20261110.json) and retain failures, interventions and limitations. Do not start the product
port's pending scientific stages from this overhaul task.

# Repository overhaul for agent work

## Goal

Make the correct path the easy one for agents: clear instructions and documentation,
useful feedback and checks that protect current work without unnecessary maintenance.

## Authorized

Philip authorized routine repairs and verification, removal of redundant third-party
tool proofs and manual policy checksum/configuration copies, and a documentation audit
covering port/overhaul leftovers and historically useful evidence. Keep the full local
commit gate, no hosted CI and scientific semantics. GPU work follows task scope and ownership in AGENTS.md; no new training allocation,
artifact deletion or publication follows from this task. Philip will link the remote.

## Done when

Approved repairs are committed and verified, remaining choices are resolved or explicitly
deferred, and matched fresh-agent exercises assess the final workflow with their limits.

## State

Owner: Codex in the current conversation, branch main. No remote is configured.

Current workflow: quick partial check, full local hook, enforced task sections,
actionable diagnostics and doctor checks for hook configuration and executability.
Actual third-party tools still check source; 92 project rules retain bad/control proofs.
Settings and versions come from pyproject.toml and the marker-aware locked export.
Manual policy/fixture hashes, copied dependency closure and dead foundation policy
metadata are removed. Frozen source/data evidence authentication remains intact.
[Safeguard receipt](../evidence/safeguards-simplification-20261110.json).

Documentation now has a project explanation, [current model](../docs/model.md),
[research context](../docs/research.md) and scoped historical findings. The
[documentation audit](../evidence/documentation-audit-20261110.json) inventories 335
historical Markdown paths and records selected-source dispositions; it is not a
line-by-line review of every archive. Historical plans and job ownership stay outside
current instructions. The older interactive tutorial is linked separately as teaching
material. Existing data/execution docs were reviewed and retained.

The clean check-only install passed offline; numerical tests reuse the existing
interpreter. A clean numerical reinstall remains unverified. Existing agent exercises
are linked in [port](port.md), but no matched before/after comparison is claimed.
Earlier diagnostics and setup receipts remain in evidence; their rule counts describe
those snapshots rather than today's corpus.

Philip's current priority is documentation housekeeping before benchmark design.
The approved [result convention](../docs/results.md) now separates present docs, active
tasks and durable results. All thirteen findings live in descriptive result folders;
FINDINGS links to them. Navigation, task conventions, the tracked closeout skill and
repository checks use that layout. No historical artifacts were deleted.
Functional-port gaps remain recorded in [port](port.md), outside this documentation task.

## Next

Recover missing executed-source and reproduction details before proposing artifact
retirement. Use the approved closeout convention for future investigations. Broader agent
operations remain in [agent operations](agent-operations.md).

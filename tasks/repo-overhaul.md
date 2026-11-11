# Repository overhaul for agent work

## Goal

Make the correct path the easy one for agents: clear instructions and documentation,
useful feedback and checks that protect current work without unnecessary maintenance.

## Authorized

Philip authorized routine repairs and verification, removal of redundant third-party
tool proofs and manual policy checksum/configuration copies, and a documentation audit
covering port/overhaul leftovers and historically useful evidence. Keep the full local
commit gate, no hosted CI and scientific semantics. No GPU work, training, artifact
transfer, remote creation or publication is authorized.

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

Open choices: rules already prescribe filenames and discovery for future benchmark
plug-ins even though that system has not been built. Recommend designing those layout
rules with the feature, while keeping data and comparison protections. This does not
change the neural-network architecture. Native/GPU/scientific work remains in
[port](port.md); remote, artifact backup and machine sharing remain in
[agent operations](agent-operations.md).

## Next

Review the remaining software-layout restriction with Philip using a concrete example.
After the chosen changes, run matched fresh-agent tasks against the baseline in the
[setup audit](../evidence/overhaul-setup-20261110.json). Do not resume scientific port
stages merely because their task file is open.

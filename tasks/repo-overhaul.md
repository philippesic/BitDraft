# Repository overhaul for agent work

## Goal

Make the correct path the easy one for agents in this repository: short instructions,
one home per fact, fast feedback, rules that earn their cost.

## Authorized

Instructions and documents restructure, the fast check lane. Other components need
Philip's decision as they come up.

## Done when

Every component has been reviewed with Philip, approved changes are committed and
verified, and fresh agents have been compared on before and after commits.

## State

Done: AGENTS.md, README, FINDINGS, topic docs, `evidence/` and `tasks/` restructured;
decision records and port plans folded into topic docs and [port](port.md);
`tools/check.py --fast` added. The baseline commit for comparison is `b4413ac`.

Remaining components:

1. Diagnostics: name the line and the fix in each message, not only the code and path.
2. Rules: about 40 proofs test third-party tool behavior, comparison and selection
   rules guard code that does not exist yet, and configuration digests in
   `tools/config.py` force a hash edit for every policy change.
3. Task-file template enforced by the check.
4. CI, container and remote, after the remote exists; see [agent operations](agent-operations.md).
5. Dependency integration: host Torch through `.host.toml`, the lock digest repeated in
   `policy/toolchain.toml`.

## Next

Review diagnostics and rules with Philip.

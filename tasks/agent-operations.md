# Agent operations

## Goal

Design project operations after the functional port and benchmark overhaul.

## Authorized

Record decisions and proposals. The tracked closeout procedure and document convention are implemented. Artifact
purge, backup automation and remote creation are not assigned now. Philip will link the remote. No hosted CI.

## Done when

The later operations phase defines and verifies evidence preservation, artifact backup,
cleanup and reusable agent procedures.

## State

Disk handling belongs to Philip's separate system. The approved
[result convention](../docs/results.md) and [closeout skill](../skills/close-results/SKILL.md)
cover producing results, updating current docs and assessing authorized cleanup.
Automated backup, artifact retirement and cross-project skill distribution remain
future operations work. A report alone cannot replace everything needed to recompute
a claim or resume a state. The procedure requires preserving source and inputs first.

CPU work is autonomous. GPU work is allowed within task scope, commonly explicitly
assigned by Philip. Check existing GPU activity and ownership first. Once a task owns
an allocation, its agents may retain it for the whole task, coordinating their use.
Other tasks' jobs and explicit limits remain protected. AGENTS.md holds the rule.

## Next

Finish the functional port, then overhaul benchmarks, then design these operations.
Do not treat the unlinked remote or future retention automation as a port blocker.

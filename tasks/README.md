# Task convention

One `task-name.md` describes one active, multi-session task. This README is the convention,
not an active task. An existing task does not authorize an agent to resume it unasked.

Use a descriptive title followed by these sections in order:

- `Goal`: intended outcome.
- `Authorized`: assignment, resources, budgets, stopping conditions and explicit permissions.
- `Done when`: observable outcome and necessary verification/closeout criteria.
- `State`: owner, branch/worktree, current facts and linked execution/source receipts;
  rewrite instead of appending. Distinguish decisions from pending proposals.
- `Next`: the next useful action and any input it depends on.

Read relevant current docs and [FINDINGS](../FINDINGS.md) before investigating. Put durable
learning in [results](../docs/results.md), not in a task diary. Update current docs as
behavior changes. Invoke [close-results](../skills/close-results/SKILL.md) when closing an
investigation. Delete a task once its criteria are satisfied; Git retains its history.

After interruption, reconcile this state with the latest user instruction and actual
process ownership before acting. Preserve the prior allocation rather than inventing a
new budget or silently resetting counters. Questions and status requests do not cancel
the assignment; an explicit cancellation does. A finished attempt is not necessarily a
finished task. Keep unfinished work and the concrete next action visible.

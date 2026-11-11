# Task convention

One `task-name.md` describes one active, multi-session task. This README is the convention,
not an active task. An existing task does not authorize an agent to resume it unasked.

Use a descriptive title followed by these sections in order:

- `Goal`: intended outcome.
- `Authorized`: scope, boundaries and explicit permissions.
- `Done when`: observable completion criteria.
- `State`: current owner, relevant work and unresolved issues; rewrite instead of appending.
- `Next`: the next useful action and any input it depends on.

Read relevant current docs and [FINDINGS](../FINDINGS.md) before investigating. Put durable
learning in [results](../docs/results.md), not in a task diary. Update current docs as
behavior changes. Invoke [close-results](../skills/close-results/SKILL.md) when closing an
investigation. Delete a task once its criteria are satisfied; Git retains its history.

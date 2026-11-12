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

Before creating raw artifacts, record their purpose, required inputs and disposal point
in the task or existing run recipe. Closeout retains verified compact evidence and
needed inputs, then retires obsolete outputs through [the retirement command](../docs/results.md#retirement-command).
An unexplained pile of raw outputs does not satisfy completion.

Read relevant current docs and [FINDINGS](../FINDINGS.md) before investigating. Put durable
learning in [results](../docs/results.md), not in a task diary. Update current docs as
behavior changes. Invoke [close-results](../skills/close-results/SKILL.md) when closing an
investigation. Delete a task once its criteria are satisfied; Git retains its history.

For a long job, keep the executed source, owner/birth/boot receipt, allocation and consumed
clocks in `State`. The [execution](../docs/execution.md) and [benchmark](../docs/benchmarks.md)
guides define the recovery commands and guards. Keep only the current state and concrete
next action; Git preserves earlier task versions.

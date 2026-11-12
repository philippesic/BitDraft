# Development

Start with [setup](../README.md#setup), [architecture](architecture.md), [support limits](support.md), relevant topic
docs and [FINDINGS](../FINDINGS.md). The findings index leads to prior results; it does
not contain their full evidence. Read the linked result before reusing a conclusion.

Confirm this checkout and the assignment before changing files. Legacy source is available from the independent archives described in [evidence](evidence.md);
it does not require the old checkout. Read the relevant task, topic and result;
the findings index and architecture map are entrances, not a requirement to load every
document. For multi-session work, use the [task convention](../tasks/README.md). Existing
tasks are context, not new assignments. Latest user steering overrides a stale task.

Maintain the affected behavior and its useful tests with the change. Update commands,
the [architecture map](architecture.md#entries-and-flows), topic docs and current task
state when their facts change. A new public entry also belongs in `policy/commands.toml`.
Remove obsolete guidance rather than accumulating amendments. Existing tests and docs
may already suffice; routine changes need no new ceremony. Git preserves implementation
history.

Use the quick partial check while editing and the full pre-commit gate before committing:

```sh
uv run --locked --no-sync --offline python tools/check.py
uv run --locked --no-sync --offline python tools/check.py --full
```

[Checks](checks.md#choosing-verification) explains which evidence the change needs,
diagnostics, hook setup and the separate numerical
interpreter. There is no hosted CI. CPU verification does not qualify GPU execution,
serving parity or performance. Run only the execution lanes within the assigned task.

For an experiment, preserve the question, model/data identities, allocation and stopping
conditions before execution. Use maintained [benchmark](benchmarks.md) and
[training](execution.md) entries. A valid unfavorable result changes the conclusion;
it is not an execution error to repair until the score improves. Keep recovery state,
acceptance leader and broader-comparison incumbent separate. New allocations or weaker
scientific guarantees need a decision; routine repair within an existing allocation does
not require repeat approval.

For long jobs, inspect the recorded owner and actual GPU activity before adoption or
recovery. A lost tool session does not prove a worker stopped. Preserve healthy work
through questions and unrelated edits. Update the current task with the executed source,
receipts and next action; do not infer completion from a checkpoint or successful launch.
When asked to leave a run working, use persistent host execution and verify actual progress
before handoff. Scheduled agent follow-up requires an explicit request; do not invent one.

Shared `benchmark-design` is for designing or repairing measurements, and `correct` is
for recurring agent/workflow mistakes. Ordinary candidate runs and isolated bug fixes
use the maintained commands and relevant tests. Add another skill only when a recurring
procedure has a demonstrated gap.

At investigation closeout, read and follow [close-results](../skills/close-results/SKILL.md)
and the project's [result format](results.md). The tracked skill is the reusable procedure;
this pinned project copy governs closeout even when a shared version is installed. The
link makes it available to agents without assuming automatic skill discovery from
`skills/`. Routine fixes need no result unless they produce a durable lesson or decision.

Keep small results in Git and large raw artifacts at their external location. Preserve
needed source and inputs before authorized cleanup. Update FINDINGS after writing a
result and delete completed task files. Commit coherent checkpoints and push to an
established user-owned remote; report local-only work when none is configured.

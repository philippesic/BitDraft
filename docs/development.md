# Development

Start with [setup](../README.md#setup), [architecture](architecture.md), [support limits](support.md), relevant topic
docs and [FINDINGS](../FINDINGS.md). The findings index leads to prior results; it does
not contain their full evidence. Read the linked result before reusing a conclusion.

For multi-session work, use the [task convention](../tasks/README.md). Existing tasks are
context, not new assignments. Make changes within the authorized scope, keeping present
behavior in `docs/` and updating it alongside code. Git preserves implementation history.

Use the quick partial check while editing and the full pre-commit gate before committing:

```sh
uv run --locked --no-sync --offline python tools/check.py
uv run --locked --no-sync --offline python tools/check.py --full
```

[Checks](checks.md) explains coverage, diagnostics, hook setup and the separate numerical
interpreter. There is no hosted CI. CPU verification does not qualify GPU execution,
serving parity or performance. Run only the execution lanes within the assigned task.

At investigation closeout, read and follow [close-results](../skills/close-results/SKILL.md)
and the project's [result format](results.md). The tracked skill is the reusable procedure;
this link makes it available to agents without assuming automatic skill discovery from
`skills/`. Routine fixes need no result unless they produce a durable lesson or decision.

Keep small results in Git and large raw artifacts at their external location. Preserve
needed source and inputs before authorized cleanup. Update FINDINGS after writing a
result and delete completed task files. Commit coherent checkpoints and push to an
established user-owned remote; report local-only work when none is configured.

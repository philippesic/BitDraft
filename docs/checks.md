# Development verification

The full commit gate is:

```sh
uv run --locked --no-sync --offline python tools/check.py
```

It verifies the locked small environment, forbidden repository content, comments and
exceptions, Ruff, strict core/numerical typing, dependency direction, Vulture/Deptry,
document links and active map symbols, required CPU scenarios, and every bad/control
specimen. It installs and downloads nothing and executes no GPU work. The numerical
wrapper runs pinned CPU subprocesses and rejects missing cases, skips or failures.
Current required names live in [CPU policy](../policy/cpu-tests.toml) and
[numerical policy](../policy/numerical.toml). Rule IDs/group totals derive from
[rules](../policy/rules.toml), not duplicated literal counts. Digest pins detect drift;
changes to policy remain reviewable and do not grant approval.

A normal new authored file, test or document requires no file-allowlist/checker edit.
Critical entries remain mandatory. Normal test discovery runs every collected test;
the critical scenario roster additionally prevents removal, skipping and xfail from
producing a green gate. There is one authored critical roster, with a pinned digest.

For a focused diagnosis, use `ruff check PATH`, `ruff format --check PATH`, or a named
CPU test through `.venv/bin/python -B -m pytest tests/test_training.py`. The numerical
interpreter can run `-B -m unittest tests.numerical.test_model -v` with CUDA hidden and
TMPDIR=/tmp. Focused checks issue no full verification receipt. `--self-test` runs the
paired guardrail corpus with bootstrap prerequisites; it is not the full commit gate.

Production core and numerical source retain all strict typing settings. Only numerical
test modules permit Any expressions and untyped external calls. One numerical analysis
checks both profiles. Mypy runs uncached without writing unused caches. No hardware
measurements are cached. Exact exceptions are enumerated in
[exceptions](../policy/exceptions.toml); ordinary comments/docstrings have no exemption.

The full receipt binds authored source and policy content. Doctor reports it stale
after either changes. It also checks the installed pre-commit interpreter path after
a checkout relocation. The normal hook checks the staged worktree and independently
compares index contents. A green receipt means the named CPU scenarios executed; it
does not prove authority, scientific adequacy, actual model parity, native correctness
or serving performance. Hosted CI and an optional container remain unexercised choices.

[Decision 007](decisions/007-checks-and-review.md) records intentional policy changes and
reference-semantic corrections. [Execution](execution.md) describes runtime refusal,
reporting, identity and resource scope. [Agent exercise evidence](evidence/agent-readiness.json)
is one fresh readiness/refusal task with explicit isolation and attribution limits.

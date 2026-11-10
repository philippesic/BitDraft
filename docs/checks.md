# Checks

The exact public command is `uv run --locked --no-sync --offline python tools/check.py`.
Setup is explicit: `uv sync --locked --only-group check --no-build` with authenticated
CPython 3.11.15 and uv 0.11.32. The complete closure is [uv.lock](../uv.lock).

Ten enforcing stages run in order, followed by qualification status. Bootstrap checks
versions, offline lock freshness, fixed tool settings and mandatory rosters. A setup
failure stops dependent checks. Later failures aggregate. Diagnostics carry a code and
relative path, never a secret value. Check execution fixes locale C.UTF-8, timezone UTC,
hash seed zero, UTF-8, no bytecode and no automatic pytest plugins by re-executing Python.

The rule contract is [rules](../policy/rules.toml); the exact specimen roster is
[cases](../policy/cases.toml). There are 125 atomic rules across 19 groups. Every rule
requires one intended rejection and one passing counterpart. Multi-syntax witnesses
remain one atomic rule. A missing tool, timeout, crash or extra diagnostic fails proof.
[CPU inventory](../policy/cpu-tests.toml) requires every named test to run and pass;
empty collection, skip, xfail, deselection and missing nodes fail.

Ruff checks format and exactly F401/F821/I001/B006/UP006. mypy uses pinned strict mode
plus all explicit settings in pyproject. Tests and tools are included. Import Linter
and AST checks enforce dependency, side-effect, owner and future-boundary restrictions.
Vulture uses confidence 60; Deptry enables DEP001–DEP004 with tests included. Exact
necessary exceptions live only in [exceptions](../policy/exceptions.toml), including
pytest discovery and legitimate check-only imports; broad/stale exceptions fail.

The optional [CPU image](../Dockerfile.checks) pins the authenticated Python manifest.
Its runtime needs a read-only mount of the checkout .git directory for Dulwich inventory.
No image is built in this bounded local stage. [CI](../.github/workflows/checks.yml)
uses pinned checkout, read-only credentials and a network-disabled check container.
No remote, hosted execution or branch protection exists. A local verification receipt
is fresh, external to tracked content and explicitly REVIEW_REQUIRED; it cannot approve
its own policy change. Source, native, runtime verify and benchmark status is unavailable.

Local CPU validation on 2026-11-09 passes all ten enforcing stages, 26 required CPU
nodes, 125 intended bad rejections and 125 passing controls. The independent outer
checks reject disabled comment scanning, a missing executable, one removed specimen,
duplicated rule IDs, unrelated/additional diagnostics and missing/skipped/xfail tests.
The ordinary-comment scratch proof returns COMMENT_PYTHON with exit 1; restoring the
original file restores exit 0. Pre-commit uses its staged worktree plus an independent
index/content comparison; a changed unstaged worktree cannot claim that index identity.

No atomic rule was removed or weakened. Pinned mypy 1.18.2 uses `misc` for unchecked Any
and untyped-decorator diagnostics; fixtures preserve that actual code. It warns with
exit 0 for unused config sections, so the wrapper promotes that exact warning to a
failure. Focused external-call/decorator witnesses allow untyped definitions in their
tiny external stub to isolate the intended call/decorator diagnostic. The full repository
keeps all strict settings. All specimen output checks preserve real relative filenames.

There are eleven exact live exceptions: six legitimate checker/test dependency edges,
four pytest discovery hooks and one Hypothesis decorator typing directive. No ordinary
comment or docstring is exempt. Selected small old evidence objects were hashed from
Git; external checkpoints/captures were neither copied nor freshly hashed.

Early implementation pytest runs used the default /tmp/pytest-of-philip directory.
Subsequent runs use the repository temporary root. External temporary directories were
left untouched under the explicit outside-cleanup prohibition; details remain in the
external acceptance evidence. Optional image and hosted CI execution are unrun.


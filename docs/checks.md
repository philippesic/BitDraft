# Checks

## Commands

| Command | Time | Use |
| --- | --- | --- |
| `uv run --locked --no-sync --offline python tools/check.py --fast` | about 3 s warm, 30 s cold | While editing |
| `uv run --locked --no-sync --offline python tools/check.py` | about 90 s | Commit gate; the pre-commit hook runs it |
| `uv run --locked --no-sync --offline python tools/check.py --self-test` | about 25 s | Rule proofs only, after changing a checker or rule |
| `uv run --locked --no-sync --offline python -m tools.doctor` | seconds | Environment, hook and receipt inspection |

None of them installs or downloads anything, formats files or uses the GPU.

`--fast` runs every static stage and the ordinary CPU tests. It skips the numerical
test (about 37 s), the required-test roster and the rule proofs, keeps a type-check
cache of about 300 MB in `/tmp`, writes no receipt and is never a commit result.

For one file: `.venv/bin/ruff check PATH`, `.venv/bin/ruff format --check PATH`, or
`.venv/bin/python -B -m pytest tests/test_training.py`. The numerical tests run in the
numerical interpreter: `CUDA_VISIBLE_DEVICES= TMPDIR=/tmp PYTHON -B -m unittest
tests.numerical.test_model -v`.

## Stages

| Stage | Covers |
| --- | --- |
| bootstrap | Python, tool and lock pins; configuration digests; hook configuration |
| repository | Tracked outputs, large files, secrets, ignored escapes, instruction files, diary names |
| comments | The comment ban and `policy/exceptions.toml` |
| ruff | Format and selected lint codes |
| mypy | Strict typing; the numerical profile allows Any only in `tests/numerical` |
| architecture | Import direction, pure modules, owners, environment and clock access, entries |
| dependencies | Deptry, Vulture and declared tool use |
| documentation | Relative links and anchors, pinned evidence links, system-map symbols |
| cpu | All CPU tests, the numerical subprocess test and the required roster in `policy/cpu-tests.toml` |
| proofs | Each rule in `policy/rules.toml` rejects its bad specimen and accepts its control |

A green full check means the named CPU scenarios ran. It does not show native
correctness, retained-artifact parity, scientific adequacy or serving performance.

## Comment ban

Python, shell, TOML, YAML, C/C++/CUDA, Dockerfile and ignore-file comments and all
docstrings fail. Markdown prose is allowed; fenced code in Markdown is checked. Only
a necessary shebang, a legally required license header or a genuine tool directive can
have an exception, and each entry in `policy/exceptions.toml` names the exact path and
content with a reason. Wildcards and file-wide suppressions are refused. Unsupported
syntax fails closed. Patches are checked hunk by hunk against their pinned base.

## Adding things

- A new test, source file or document needs no checker edit.
- Add a test to `policy/cpu-tests.toml` only when its removal must fail the gate.
- A new public command must be declared in `policy/commands.toml`.
- A new evidence link to the predecessor repository must use the pinned revision and
  be registered in `policy/evidence.toml`.

## Pins and digests

`tools/config.py` pins the tool versions and the digests of `policy/rules.toml`,
`policy/commands.toml`, `policy/boundaries.toml`, `policy/cases.toml`,
`policy/cpu-tests.toml`, `policy/check-stages.toml`, `policy/validation.toml`, the hook
configuration, the workflow and the container file. A digest mismatch means a safeguard
changed. If the change is intended and authorized, update the digest in the same commit
and say so in the report. If not, revert it. Never update a digest only to get a pass.

## Fixes

| Diagnostic | Fix |
| --- | --- |
| `COMMENT_*` | Delete the comment; move rationale to the topic doc |
| `LINT` | `.venv/bin/ruff format PATH` and fix the printed lint errors |
| `TYPES` | Fix the printed mypy errors; do not add `Any` or `type: ignore` |
| `ENV_OWNER` | Read environment variables only in `w1a1_eagle/config.py` or `tools/config.py` |
| `TIMING_OWNER`, `SEEDED_RANDOM` | Take an injected clock or seeded generator as an argument |
| `ARCH_*`, `IMPORTS` | Move the code so imports follow the direction in `policy/boundaries.toml` |
| `OWNER_*` | Write through the owner module listed in [system map](system-map.md) |
| `PUBLIC_ENTRY` | Declare the entry in `policy/commands.toml` or remove the main guard |
| `DOC_LINK`, `MAP_ENTRY` | Fix the link, anchor or system-map symbol |
| `EVIDENCE_PIN` | Use the pinned revision URL and register it in `policy/evidence.toml` |
| `REPO_DIARY` | Put the content in FINDINGS, a topic doc or a task file |
| `INSTRUCTIONS_BUDGET` | Shorten AGENTS.md to 80 lines and 4,000 bytes |
| `REQUIRED_TESTS` | Read the pytest output above it; a test failed, was skipped or disappeared |
| `CHECK_CONFIG`, `PIN_*`, `STAGE_GATE` | A pin or digest changed; see Pins and digests |

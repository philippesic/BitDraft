# Checks

## Commands

| Command | Time | Use |
| --- | --- | --- |
| `uv run --locked --no-sync --offline python tools/check.py` | about 3 s warm, 30 s cold | While editing |
| `uv run --locked --no-sync --offline python tools/check.py --full` | about 70 s | Commit gate; the pre-commit hook runs it |
| `uv run --locked --no-sync --offline python tools/check.py --self-test` | seconds | Rule proofs only, after changing a checker or rule |
| `uv run --locked --no-sync --offline python -m tools.doctor` | seconds | Environment, hook and receipt inspection |

None of them installs or downloads anything, formats files, changes Git configuration
or uses the GPU.

The plain command runs every static stage and the ordinary CPU tests and labels itself
partial. It skips the numerical test (about 37 s), the required-test roster and the
rule proofs, keeps a type-check cache of about 300 MB in `/tmp`, writes no receipt and
is never a commit result. `--full` runs everything portable and writes the receipt.

There is no hosted CI. Verification is the local hook plus the checks agents run.
The hook is `.githooks/pre-commit`. Enable it once per clone with
`git config core.hooksPath .githooks`; the doctor reports whether it is enabled.

For one file: `.venv/bin/ruff check PATH`, `.venv/bin/ruff format --check PATH`, or
`.venv/bin/python -B -m pytest tests/test_training.py`. The numerical tests run in the
numerical interpreter: `CUDA_VISIBLE_DEVICES= TMPDIR=/tmp /absolute/path/to/numerical/bin/python -B -m unittest
tests.numerical.test_model -v`.

## Stages

| Stage | Covers |
| --- | --- |
| bootstrap | Python, tool and lock pins; configuration invariants; executable full-gate hook |
| repository | Tracked outputs, large files, secrets, ignored escapes, instruction files, diary names, task-file sections |
| comments | The comment ban and `policy/exceptions.toml` |
| ruff | Format and selected lint codes |
| mypy | Strict typing; the numerical profile allows Any only in `tests/numerical` |
| architecture | Import direction, pure modules, owners, environment and clock access, entries |
| dependencies | Deptry, Vulture and declared tool use |
| documentation | Relative links and anchors, pinned evidence links, system-map symbols |
| cpu | Ordinary CPU tests; with `--full` also the numerical subprocess test and the required roster in `policy/cpu-tests.toml` |
| proofs | With `--full`: each rule in `policy/rules.toml` rejects its bad specimen and accepts its control |

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
- A new evidence link to the predecessor repository must use a full commit revision and
  be registered in `policy/evidence.toml`.

## Configuration and evidence

`pyproject.toml` owns tool settings and direct dependency versions. `uv.lock` owns
the resolved dependencies and download hashes. The gate compares the installed check
environment with uv's locked export, including platform markers; it does not keep a
second copy of the dependency tree. Python and uv bootstrap identities live in
`policy/toolchain.toml`.

Rule rationale links point to current topic docs and are checked for broken targets.
Project validators reject weakened typing, broad lint/dependency suppressions, a
missing mandatory stage and a hook that omits the full gate. They check those
properties directly rather than requiring a new checksum after every policy edit.
Tests and checker modules may import declared check dependencies; product modules may not.
The required-test list lives only in `policy/cpu-tests.toml`.

The proof corpus tests project-authored validators, with one rejecting and one
accepting specimen per rule. Ruff, mypy, Vulture and Deptry run on the actual source;
the corpus does not retest their built-in error codes on toy programs.

Hashes still authenticate frozen source evidence, model/data artifacts, exact machine
line exceptions and verification receipts. A policy edit cannot authorize a run or
change scientific semantics. Preserve those identities and use the user's authorized
scope when changing checks.

## Fixes

Project diagnostics print a repair from this table after the stable rule code and
location. Tool failures retain the tool's own line numbers and detailed output.
The table is the single source for repair text. CPU receipts record verification,
not user approval; a successful check does not require a new policy approval.

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
| `REPO_TASK` | Start with a `# ` title and keep `## Goal`, `## Authorized`, `## Done when`, `## State`, `## Next` in that order; other sections go after |
| `INSTRUCTIONS_BUDGET` | Shorten AGENTS.md to 80 lines and 4,000 bytes |
| `REQUIRED_TESTS` | Read the pytest output above it; a test failed, was skipped or disappeared |
| `CHECK_CONFIG`, `PIN_*`, `STAGE_GATE`, `TYPE_CONFIG` | Inspect the named configuration and tool output; restore the required properties described above or repair the locked environment without weakening checks |
| `TYPE_*` | Fix the type error in the tool output; preserve strict typing and avoid suppression |
| `FORMAT`, `F401`, `F821`, `I001`, `B006`, `UP006` | Run `.venv/bin/ruff check PATH` and `.venv/bin/ruff format --check PATH`; repair the reported source |
| `DEP*`, `DEAD_CODE`, `UNUSED_TOOL` | Inspect the dependency tool output; declare a direct dependency in its proper group or remove the unused import, code or declaration |
| `EXCEPTION_*` | Remove a broad, stale or duplicate exception; necessary machine directives require an exact reasoned entry in `policy/exceptions.toml` |
| `INSTRUCTIONS_ROOT`, `CLAUDE_POINTER` | Keep the sole authority in root AGENTS.md and make CLAUDE.md contain only `@AGENTS.md` followed by a newline |
| `REPO_OUTPUTS`, `REPO_LARGE` | Keep generated outputs and large artifacts outside tracked source; preserve valuable files before changing their Git tracking |
| `REPO_SECRET` | Remove the credential from authored source without printing it; report any credential already committed so it can be revoked |
| `REPO_IGNORED` | Remove the ignore or symlink escape hiding authored source; preserve secrets and generated artifacts outside authored paths |
| `RULE_PAIR`, `PROOF_*` | Inspect the named specimen and actual diagnostic; restore missing tools or fixtures and repair the validator without accepting unrelated failures |
| `POLICY_REVIEW` | Run the full check for a current CPU receipt; a receipt cannot approve policy changes or scientific work |
| `VERIFY_*` | Supply matching finite token, label and mechanism evidence from the qualified runtime; CPU fixtures alone cannot qualify it |
| `DISCOVERY_*` | Follow the declared discovery protocol and unique IDs in `policy/boundaries.toml`; report a legitimate extension blocked by this policy |
| `NULL_TOPICS`, `TOPIC_*`, `GROUP_PROPERTIES` | Preserve distinct missing topics and namespaced relationships; fix grouping without merging unrelated prompts |
| `HELDOUT_ROLE`, `UNKNOWN_ANCESTRY` | Correct the declared role or ancestry evidence; never admit protected evaluation data into training, calibration or selection |
| `COUNT_*`, `CAPTURE_*`, `EXPOSURE_*` | Reconcile source counts, label boundaries and parent exposure against the declared intent; retain refused inputs and failures |
| `DISK_*` | Supply separate current filesystem and backing-space observations and satisfy the run's declared capacity requirement |
| `SELECTION_*` | Preserve the earlier durable valid serving best until a complete, comparable serving result improves it |
| `COMPARE_*` | Match the declared machine, target, workload, controls and evidence before computing a comparison; retain incompatible results separately |

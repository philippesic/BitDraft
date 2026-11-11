# Binary Eagle

Binary Eagle investigates whether a small, low-precision drafter can make a larger
language model answer faster. The drafter proposes several tokens; the target model
verifies them. Success means faster complete requests with the target's output
semantics preserved. Fast draft kernels or good training loss alone do not establish it.

The current development candidate is HE4 for a Qwen3.5-4B Q4_K_M target. Its
feed-forward weights use one bit, gate/up inputs use four bits, and down inputs use
eight bits. Other components retain their declared formats; it is not a whole-model
one-bit system. The [model](docs/model.md) explains what is trained and frozen.

This repository implements CPU data admission, numerical updates, supervised attempts,
recoverable training states and GGUF export. Native runtime builds, real serving
verification and benchmarks remain to be implemented and qualified. The historical
external DFlash Q4_K_M comparator remains the deployment reference; retained results
here do not establish an HE4 win. See [research context](docs/research.md) for the
reasoning and evidence behind the development direction.

## Setup

Requires CPython 3.11.15 and uv 0.11.32 on Linux x86_64; exact archive identities are in
[toolchain](policy/toolchain.toml). Install the check environment from wheels only:

```sh
uv sync --locked --only-group check --no-build
git config core.hooksPath .githooks
```

The second line enables the pre-commit hook, which runs the full check. Run
`uv run --locked --no-sync --offline python -m tools.doctor` to confirm the hook
is configured and executable. The check-only install does not provide the numerical
interpreter required by the full gate.

Numerical code runs in a separate interpreter with versions declared by the
`numerical` extra in [pyproject.toml](pyproject.toml); its fixture scope and retained
GGUF build identity are in [numerical policy](policy/numerical.toml). Point to it with `BINARY_NUMERICAL_PYTHON`,
`numerical_python` in an untracked `.host.toml`, or a `.venv/numerical` environment.
For a host-provided interpreter, the configuration is:

```toml
numerical_python = "/absolute/path/to/numerical/bin/python"
```

The file contains host paths only; do not put credentials in it. The numerical lane
runs with GPU visibility disabled. Check it independently with the command in
[checks](docs/checks.md#commands). A successful check-only install is not a complete
numerical setup; rebuilding the numerical environment from scratch has not been
verified from scratch here.

## Commands

```sh
uv run --locked --no-sync --offline python tools/check.py
uv run --locked --no-sync --offline python tools/check.py --full
uv run --locked --no-sync --offline python -m tools.doctor
python -m w1a1_eagle preflight INTENT
```

The first is a quick partial check, the second is the full commit gate, the third
inspects the environment and hook. The last family runs and reports CPU training;
see [execution](docs/execution.md).

## More

- [System map](docs/system-map.md), [checks](docs/checks.md), [data](docs/data.md),
  [domain terms and metrics](docs/domain.md), [model](docs/model.md),
  [research context and source guide](docs/research.md)
- [Findings](FINDINGS.md) from this project and its predecessor
- License: not yet chosen.

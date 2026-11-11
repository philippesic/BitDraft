# Binary Eagle

Binary Eagle trains small low-bit drafter models for speculative decoding with
Qwen3.5-4B Q4_K_M targets. The current model family is HE4: 1-bit weights with
4-bit activations, an 8-bit down projection and a Q4_K output head. Trained states
export to GGUF for a native runtime.

This repository holds the CPU source path: admitting captured training data,
HE4 numerical updates, durable and resumable training state, a supervised run
command and verified GGUF export. Native runtime builds and benchmarks are not here yet.

## Setup

Requires CPython 3.11.15 and uv 0.11.32 on Linux x86_64; exact archive identities are in
[toolchain](policy/toolchain.toml). Install the check environment from wheels only:

```sh
uv sync --locked --only-group check --no-build
git config core.hooksPath .githooks
```

The second line enables the pre-commit hook, which runs the full check.

Numerical code runs in a separate interpreter with the versions in
[numerical policy](policy/numerical.toml). Point to it with `BINARY_NUMERICAL_PYTHON`,
`numerical_python` in an untracked `.host.toml`, or a `.venv/numerical` environment.

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
  [domain terms and metrics](docs/domain.md)
- [Findings](FINDINGS.md) from this project and its predecessor
- License: not yet chosen.

# Binary speculative decoding

This project develops compact speculative drafters and evaluates their **native
end-to-end decoding** against frozen target and DFlash baselines. Current work
includes DFlash W1A8 and component precision/recovery; the original W1A1 EAGLE
scaffold below is historical, not the current execution guide.

Start with [STATUS](docs/STATUS.md) for the current owner, active work and pauses.
Use [candidate stages](docs/CANDIDATE_STAGES.md) and
[candidate benchmarking](docs/CANDIDATE_BENCHMARKING.md) for the current scientific
contracts. The maintained [training/evaluation workflow](scripts/contender_pipeline/README.md)
documents candidate records, full elapsed budgets, compatible data reuse,
checkpoint recovery, native export/admission and evaluation tiers.

The current [component campaign](docs/COMPONENT_CAMPAIGN.md) owns its GPU lane.
Reading this guide does not start or resume a campaign. Preserve active training
and use the current owner's recorded handoff before competing GPU work. Earlier
runtime-only speedups do not establish a trained-model improvement; training
throughput, exact output correctness, acceptance and native request latency are
separate measurements.

## Original scaffold reference

A bounded systems/ML experiment: can a genuinely binary EAGLE drafter improve
end-to-end LLM throughput enough to offset lower speculative-token acceptance?

Start with the [project overview](docs/history/PROJECT_OVERVIEW_20261105.md), which defines the
goal, scope, milestones, and decision criteria. See [setup](docs/SETUP.md) for
build instructions and [evaluation](docs/EVALUATION.md) for measurement rules.
Historical operations are described in [agent operations](docs/AGENT_OPERATIONS.md);
current ownership and execution contracts above take precedence.

This checkout runs directly in WSL on the RTX 5080 machine at
`/home/philip/binary-eagle-decoding`. Use the local shell and GPU; current
work and historical results are distinguished in [status](docs/STATUS.md).

## Quick start

Requires Git, Python 3.11+, [uv](https://docs.astral.sh/uv/), and a C/C++ compiler.
Run from this repository's root:

```sh
make setup
make check
make doctor
make build-cpu
./build/llama-cpu/bin/llama-server --version
```

`make build-metal` is available for Apple development machines.
`make build-cuda` targets SM75 on a Linux machine with NVIDIA's CUDA toolkit.
CPU and Metal runs are development checks, not evidence for the GPU hypothesis.

## Layout

```text
configs/                 Versioned experiment plans (not yet a runner interface)
data/                    Local prompts, calibration data, and hidden states
docs/                    Project guidance, setup, and evaluation protocol
experiments/             Experiment notes and small, reviewable reports
kernels/                 Future packed binary CUDA prototypes
models/                  Local checkpoints and converted GGUF files
results/                 Local raw runs, logs, and environment manifests
scripts/                 Bootstrap checks, builds, and environment capture
src/w1a1_eagle/           Future simulation, QAT, export, and analysis code
third_party/llama.cpp/   Pinned upstream Git submodule
```

At initial scaffold creation, only the upstream runtime and build tooling existed.
That historical milestone description is superseded by the current status and
maintained workflow linked above; it does not describe today's implementation.

Development tools are pinned in `uv.lock`. The llama.cpp revision is pinned by
the submodule gitlink; see [third-party notes](third_party/README.md). Large
artifacts stay out of Git. Do not use `git submodule update --remote` for routine
setup, because it changes the experimental runtime version.

# BitDraft

![BitDraft](assets/bitdraft.svg)

Experimental low-bit draft models for faster language-model generation.

In speculative decoding, a small draft model proposes several tokens at once and a
larger target model verifies them. BitDraft explores whether reducing the drafter's
precision can make this process faster while preserving the target's output rules.
A cheaper drafter helps only when enough of its proposals are accepted.

## Status

BitDraft is a research project under active development. It provides training data
capture, drafter training, GGUF export and comparisons against released DFlash drafters,
including checks during training and evaluations with workflows that use tools.

Current work focuses on HE4 drafters with one-bit feed-forward weights and Qwen3.5-4B
targets. Other model components retain higher precision. See [findings](FINDINGS.md)
for measured results and [supported configurations](docs/support.md) for their limits.

## Requirements

- Linux x86-64, Python 3.11.15 and uv.
- An RTX5080 for the currently tested GPU configuration.
- Compatible target/drafter weights and the project's pinned native runtime.

Model weights, compiled runtimes and training datasets are not bundled with the source.
Support for other GPUs, model families and precisions needs separate validation.

## Getting started

Follow the [setup instructions](docs/development.md#setup) to install the environment.
The main workflows are:

- [Prepare training data](docs/data.md) from prompts and a target model.
- [Train or continue a drafter](docs/execution.md) and export it as GGUF.
- [Compare drafters](docs/benchmarks.md) by acceptance, request speed and tool workflows.

Model inputs and generated files live under `artifacts/`, which Git ignores. The
[storage guide](docs/storage.md) explains the layout and what needs a separate backup.

## Development

Build, validation and contribution guidance is in the [development guide](docs/development.md).
The [architecture](docs/architecture.md) and [model description](docs/model.md) explain
how the implementation works.

## License

A license has not yet been selected.

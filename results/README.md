# Local run outputs

Use a unique directory per run for logs, timings, counts, resolved configuration,
and environment manifests. All contents except this README are ignored. Preserve
raw data externally when sharing a result and link it from `experiments/`.

Ignored outputs can include unique reports, analysis scripts and recovery evidence;
they are not disposable solely because this directory is ignored. Preserve compact
outcomes and reusable code in `experiments/` or `scripts/` at milestones. The
[Git/artifact guide](../docs/GIT_AND_ARTIFACTS.md) records this boundary; retained
[precision charts](../experiments/precision-history-20261105/README.md) are one example.

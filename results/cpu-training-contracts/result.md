# Matching forward values did not guarantee matching learned-quantizer gradients

## Outcome

The learned-head batching audit found activation VJPs scaled by 1/sqrt(valid_depth) despite identical hard logits and loss. Other bounded audits passed recurrence/cache VJP equivalence, 81 exact serialized projection comparisons, and staged optimizer-state validation. These answer distinct questions.

## Scope and limitations

Historical reduced synthetic CPU fixtures, learned A1/A4/A8 or fixed activation variants as specified in each report. No full-size HE4, GPU fit or speed is established.

## Decision

Retain separate forward, gradient, export and resume checks; do not infer one from another.

## Evidence and reproduction

- [parallel20261002/lsq_batching/report.md](../sources/experiments-parallel20261002-lsq-batching-report-f473661b.txt)
- [parallel20261002/recurrent_vjp/report.md](../sources/experiments-parallel20261002-recurrent-vjp-report-07036e93.txt)
- [parallel20261002/export_function/report.md](../sources/experiments-parallel20261002-export-function-report-82903f4c.txt)
- [parallel20261002/resume_validation/README.md](../sources/experiments-parallel20261002-resume-validation-readme-d5eabdf0.txt)
- [parallel20261002/sign_inertia/report.md](../sources/experiments-parallel20261002-sign-inertia-report-d8132d83.txt)
- [parallel20261002/step_bookkeeping/report.md](../sources/experiments-parallel20261002-step-bookkeeping-report-42674677.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

# Scale fitting and selective precision rescue improved acceptance without earning promotion

## Outcome

Fitted group scales reached 0.424965 accepted drafts per round, versus 1.0415 for Q4_0. Selective attention Q8_0 rescue reached 0.6093; restoring the original FP16 head reached 0.5112. The later body/head investigation promoted no candidate.

## Scope and limitations

RTX 5080, historical Qwen3-4B F16 target, 96 TRAIN and 24 reused DEV prompts. These acceptance values are not percentages or serving-speed ratios.

## Decision

Retain diagnostics showing distributed model error; a head-only explanation is insufficient.

## Evidence and reproduction

- [binary-scale-fitting-5080.md](../sources/experiments-binary-scale-fitting-5080-cd55bc2f.txt)
- [binary-rescue-head-5080.md](../sources/experiments-binary-rescue-head-5080-a988921c.txt)
- [qat-head-pilot-results.md](../sources/experiments-qat-head-pilot-results-913aa533.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

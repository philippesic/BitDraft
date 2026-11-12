# Verifier parity and runtime speed depend on the exact arithmetic contract

## Outcome

An isolated SM120 runtime improved request throughput 32.0–51.0% over the conservative mode32 runtime on four pinned Qwen3.5 targets; 1,920 final requests preserved matched target-only output. Separately, a historical F16 native alternative passed reused prompts but failed two fresh prompts in every repetition. The stock vLLM batch-invariant BF16 experiment passed its own bounded qualification.

## Scope and limitations

Different targets, engines and controls. The maintained-engine result does not port binary HE4 or authorize changing target precision. API-call timing may include compilation; read the individual reports before comparing rates.

## Decision

Retain exact verifier controls and fresh confirmation requirements; no universal lossless-runtime claim.

## Evidence and reproduction

- [local-resume-20261101/runtime-and-transfer.md](../sources/experiments-local-resume-20261101-runtime-and-transfer-206177e6.txt)
- [clean-baseline-20261101/report.md](../sources/experiments-clean-baseline-20261101-report-ed355a25.txt)
- [upstream-cuda-consistency-2026-11-01/report.md](../sources/experiments-upstream-cuda-consistency-2026-11-01-report-5439def0.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

# Exact bounded CPU replay required matching native arithmetic and cache history

## Outcome

The integrated native CPU diagnostic matched captured taps and K/V writes at 13 draft depths across three cases, including accepted-prefix reconstruction. It required ordered projections and pinned native RoPE, attention and SiLU operators. It refused gradient-enabled calls and did not select a training derivative.

## Scope and limitations

Apple M3 Max, historical candidate D W1A16 and F16 target; captured probes rather than the full vocabulary, no free-running CUDA or training qualification.

## Decision

Retain numerical and cache-boundary evidence; do not equate this diagnostic with a deployable training path.

## Evidence and reproduction

- [recurrent-binary-integrated-cpu-diagnostic.md](../sources/experiments-recurrent-binary-integrated-cpu-diagnostic-5db00866.txt)
- [recurrent-binary-target-feature-parity.md](../sources/experiments-recurrent-binary-target-feature-parity-6505eca6.txt)
- [native-verifier-trace-5080.md](../sources/experiments-native-verifier-trace-5080-8362fa0e.txt)
- [recurrent-target-block14-stage-intervention.md](../sources/experiments-recurrent-target-block14-stage-intervention-b3c88981.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

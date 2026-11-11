# The packed W1A1 head was slower than ordinary EAGLE on RTX 5080

## Outcome

Packed-head EAGLE measured 115.3805 request tokens/s versus ordinary EAGLE 123.9626: ratio 0.931. All five repetitions favored ordinary EAGLE, and both speculative variants produced identical decoded outputs across their 60 matched requests. Their apparent advantage over target-only is not a clean lossless-speedup claim: both differed from target-only on two prompts.

## Scope and limitations

Historical F16 Qwen3-4B target, 12 fixed prompts, five alternating orders, 128-token cap, one sequence. Only the draft head was binary; this is real packed CUDA execution, not a fake-quantization or SM75 result.

## Decision

Retain the negative head comparison and verifier-parity failure.

## Evidence and reproduction

- [native-end-to-end-5080.md](../sources/experiments-native-end-to-end-5080-2e29d556.txt)

Imported reported measurements, not rerun. [Evidence access](../../docs/evidence.md)
describes independent source recovery and the remaining raw-artifact limits.

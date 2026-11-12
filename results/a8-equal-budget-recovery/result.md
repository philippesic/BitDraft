# The combined A8 recipe lost to its reference and both trailed Q4_0

## Outcome

Both arms completed 7,200 trainer-accounted seconds. Reference measured 95.689 request tokens/s (70.47% of paired Q4_0), versus 69.623 (51.68%) for the combined candidate. The reference consumed 531 TRAIN prompts and the candidate 426; equal accounted time did not mean equal data exposure.

## Scope and limitations

RTX 5080 fixed DEV comparison. Failed attempts remained charged. This combined recipe cannot identify individual feature effects.

## Decision

Retain the negative comparison and its accounting limits; no component is universally rejected.

## Evidence and reproduction

- [a8-qat-recovery/comparison.md](../sources/experiments-a8-qat-recovery-comparison-34e1416c.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

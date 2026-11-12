# The pipeline profile did not establish general binary acceleration

## Outcome

No general binary-acceleration claim followed. The retained DSpark W1A8 implementation
already used integer Tensor Core work for the tested multi-position feed-forward path.
Replacing it with the available bit-serial path preserved the request's tokens and
speculative decisions but made its measured dot products slower. Preparation cost also
mattered in the separate standalone projection diagnostic. A fast isolated binary dot
cannot establish a faster complete request.

## Scope and limitations

one instrumented native request on a consumed prompt and separately timed
synthetic projections, RTX 5080, historical Qwen3 F16/DSpark. Binary MMA was not
benchmarked. These timings do not predict modern HE4 serving performance.

## Decision

Retain this scoped finding for future decisions. No new experimental or deployment decision is made by this documentation migration.

## Evidence and reproduction

[binary pipeline profile](../sources/experiments-binary-pipeline-profile-20261104-readme-8c46fef4.txt).

Imported from the linked records; no experiment was rerun for this documentation
migration. Complete executed-source, dependency and artifact availability have not
been reverified. [Evidence access](../../docs/evidence.md) describes archive recovery and remaining
raw-input limits; report revisions alone are not execution provenance. No original source,
checkpoint or capture was deleted.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

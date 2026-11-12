# A large catch-up penalty was not inherent to DSpark

## Outcome

On the modern Qwen3.5 4B Q4_K_M diagnostic, DSpark Q4_K_M catch-up cost about 25% less per round than DFlash Q4_K_M. Separately, historical DSpark W1A8 parent3504 was 16.2% faster than DFlash Q4_K_M and 7.2% slower than DSpark Q4_K_M in a quick matched-depth screen.

## Scope and limitations

The first comparison has no W1A8 arm. The second uses original Qwen3-4B F16 on RTX 5080. These are separate model/target eras, not a common leaderboard.

## Decision

Retain the counterexample to an architecture-wide catch-up explanation.

## Evidence and reproduction

- [dspark-dflash-catchup-20261104/README.md](../sources/experiments-dspark-dflash-catchup-20261104-readme-e527a691.txt)
- [dspark-w1a8-dflash-20261104/README.md](../sources/experiments-dspark-w1a8-dflash-20261104-readme-185a81c1.txt)

Imported historical measurements; no experiment was rerun. Full source/history and
non-copied supporting records are recoverable from the verified external archive
described in [evidence access](../../docs/evidence.md). Raw models/captures are not
part of the Git archive; past retirement and missing inputs limit exact reproduction.

## Artifact disposition

Retain the linked compact reports, measurements and independently archived source.
These imported findings do not promise raw checkpoint, capture or runtime availability.
Current availability and later authorized disposals are recorded in
[the evidence guide](../../docs/evidence.md); historical observations stay unchanged.

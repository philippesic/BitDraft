# FP16 passed bounded capture and training checks; broad qualification remains pending

2026-11-11 followup: the broader investigation completed and qualified the scoped storage
format. See [FP16 storage adoption](../fp16-storage-adoption/result.md). The original
bounded outcome and unresolved decision below describe the earlier evidence stage.

## Outcome

The RTX 5080 completed a matched 72-update probe from the final original HE4broad state,
using three sequences, including an 8,394-token document prompt. Both storage arms kept
finite training values and exported through the new producer. Four native requests
matched target-only tokens and stopping in both arms, with independently reconciled
native draft counters. Here FP16 describes stored teacher features; both deployed models
use the same HE4 weight/activation formats.

| Probe | F32 features | FP16 features |
| --- | ---: | ---: |
| Completed updates | 72 | 72 |
| Mean training loss | 0.535918 | 0.524494 |
| Native accepted/proposed tokens | 119/233 | 121/229 |
| Native requests matching target output/stopping | 4/4 | 4/4 |

Native capture also matched serving trajectories for three short prompts, retained EOS
as a hard label without adding it to the input stream, wrote exact F32-to-F16 rounded
arrays, and reproduced a cold-context request byte for byte. A public capture command
then produced a three-sequence FP16 corpus with explicit partial supervision, and the
public supervisor continued the original optimizer through three new CUDA updates.

## Scope and limitations

The 72-update probe repeatedly used three sequences. Some native requests reused those
inputs. It establishes bounded execution and storage behavior, not generalization,
long-run equivalence or a speed advantage. The arms have different complete source hashes
because acquisition code changed; their numerical graph, optimizer and export sources
are compared in the compact evidence. The broader qualification below must finish before
production adoption. Earlier 2080 Ti results remain in the
[initial screen](../fp16-feature-screen/result.md).

## Decision

Unresolved for production adoption. Retain F32 references and run the frozen
[qualification policy](qualification-policy.json): 64 diverse TRAIN captures, 32 separate
DEV captures, matched training and the actual short native request panel. Passing this
qualifies the tested Q4KM target/HE4 recipe only; new targets and architectures need their
own checks. Failed and inconclusive checks remain evidence.

## Evidence and reproduction

[Compact measurements](result.json). Full drivers, inputs, states, exports and raw native
responses remain under `/home/philip/binary-eagle-artifacts/qualification/` and
`/home/philip/binary-eagle-artifacts/benchmarks/precision-pilot-native-02/`. Executed
source snapshots are beside each training attempt. Native serving used the complete
rebuild; artifact identities and commands are retained in the comparison specification.
The initial capture verification attempt expected serving to omit EOS; the corrected
verifier accounts for the retained EOS label. Both attempts remain in their logs.

## Artifact disposition

Retain the compact measurements and executed source linked above. Reusable saved
controls and active data/model inputs retain their dependency closure outside source.
Qualification-only descendants are disposable after their measurements and failure
receipts are verified; exact retired states cannot be resumed.

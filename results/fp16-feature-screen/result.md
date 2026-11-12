# FP16 feature rounding passed a small GPU screen; training suitability remains unresolved

## Outcome

On an RTX 2080 Ti, 24 F32 feature arrays from three existing code/prose/reasoning
captures round-tripped through FP16 with zero overflows. Maximum relative L2 error
across arrays was 0.000219235 (0.021924%). Maximum absolute error was 0.00780869.

The same short captures also passed through the actual initializer's fusion projection
on the 2080 Ti. All outputs stayed finite; relative output L2 errors were 0.0137-0.0145%.
[Fusion measurements](fusion.json) and [executed driver](executed-fusion.txt) preserve
source identity and scope. This did not run downstream layers or optimizer updates.
The first fusion invocation failed due to missing PyYAML; installing PyYAML 6.0.3 in
the isolated qualification dependency directory resolved it. No shared environment changed.

## Scope and limitations

Three short sequences, eight taps each, 52,674,560 source bytes. This does not qualify
long contexts, gradients, accumulated training changes, native acceptance or speed.
The remote environment used Torch 2.11.0+cu128, which supports sm_75; this differs from
the new repository's Torch 2.14.0 lock. No training update or model serving ran.
The active RTX 5080 job was left alone. No new prompts or production captures were generated.

## Matched-training attempts

A bounded six-update-per-arm experiment was attempted, reusing these existing captures.
The first attempt incorrectly supplied the target GGUF as the Q4_K head source and was
refused: its tied embedding is Q6_K. The retained initializer has a separate Q4_K output
head. A second attempt selected that authenticated artifact, then both F32/F16 arms were
refused by the existing host-memory budget check before updates. The 2080 Ti WSL host has
about 16 GB RAM and 14.2-14.4 GB available during these attempts. No memory guard was
relaxed. This is a setup/resource refusal, not evidence against FP16 learning quality.

[First refusal](training-refused.json), [second refusal](training-with-head.json),
[first driver](executed-training-refused.txt), [second driver](executed-training-with-head.txt).
Exact source, target and separate head identities are recorded. The modern target and
BF16 initializer were copied into the remote qualification directory as reusable inputs;
this was not prompt generation or a new production training dataset.

## Decision

Unresolved. Keep FP32 references and require matched training/native evaluation before
adopting FP16 captures. Loader support and successful rounding alone are insufficient.

## Evidence and reproduction

[Array measurements and authenticated input identities](result.json) and
[executed batch driver](executed-screen.txt) and [executed numerical screen](executed-precision.txt). The driver called the initial numerical
screen implementation; raw transferred source and inputs remain under
`pstack-2080:/home/philip/binary-eagle-benchmark-qualification-20261110/`.
The later screen revision adds a 32 MiB input limit and expresses L2 reduction using
explicit tensor operations; it was not the executed remote revision. Preserve the remote
source with the execution record. No figure or model-selection claim is made.

## Artifact disposition

Retain the compact measurements and executed source linked above. Reusable saved
controls and active data/model inputs retain their dependency closure outside source.
Qualification-only descendants are disposable after their measurements and failure
receipts are verified; exact retired states cannot be resumed.

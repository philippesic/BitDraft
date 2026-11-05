# Modern admission patch (not execution qualification)

Apply `modern-admission.patch` to llama.cpp revision
`5eb221ec53cb4317aef963e3bdec3cc69401eca4` with `git apply`.
No parent gitlink change is required.

Adds a separate ffn18/A8 contract with six layers, eight exact taps,
2560 embedding width, 9216 FFN width, 32/8 attention heads, 16 slots,
rope base 10000000 and the five sliding/one full attention pattern.
Requires shared target head/embedding, excludes DSpark/selector tensors,
retains packed-pair shape/type/dense-shadow checks and legacy guards.
Uses existing packed decoder FFN operations. This does not qualify their
GPU execution or target-shared graph.

CPU CMake build and actual frozen zero.gguf model load succeeded November4.
Raw build/load logs and standalone loader source are in
`/home/philip/binary-eagle-decoding/runs/dflash-modern-20261104-02/`.
Load exit success only establishes tensor admission; no inference was performed.
Frozen artifact was not changed. CUDA graph, capture, QAT, resume and full
1800 update seconds remain outstanding.

# Lightweight binary candidate screening

Zero training updates. GPU jobs use the common contender launcher/supervisor;
plans and evidence live under `runs/binary-screen-20261104-01/`.

- `kernels_v2.cu`: standalone SM120 DP4A, A1 XOR/popcount, A4/A8 bit-serial
  operators. Not a ggml backend, GGUF exporter or native serving candidate.
- `bench_v2.py`: current actor's six unique projection shapes; five graph timing
  repetitions with packing included, independent integer/code/scale checks.
- `bench_dflash_shapes.py`: same operators on released DFlash body geometries;
  does not test a binary DFlash model's quality.
- `select_inputs.py`: freezes12 already-TRAIN groups before precision outcomes.
- `quality.py`: unadapted activation policies and separate original-BF16 local
  row/group128/two-plane reconstruction. Reuses authenticated native input data.
- `quality_followup.py`: explicitly adaptive, same-input follow-up motivated by
  fusion sensitivity; verifies the same A8 control and reuses original local
  reconstruction evidence. Not fresh confirmation.
- `activation_range.py`, `summarize.py`: recomputable range and evidence summaries.

The successful CUDA build used:

```
/usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -arch=sm_120 --ftz=false --fmad=false \
  -I/home/philip/binary-eagle-decoding/results/cuda-glibc-compat/include \
  -Xcompiler -fPIC -shared kernels_v2.cu -o kernels-v2.so
```

The existing local compatibility header fixes the two CUDA/glibc rsqrt noexcept
annotations; no system headers were changed. SASS is retained in the raw run.
The v1 sources remain because completed/superseded plans pin their exact bytes.
All selected source/worktree paths must remain intact for evidence replay.

Do not interpret warm-device matrix timing as total draft speed, teacher-forced
TRAIN top1 counts as native acceptance, or reconstruction improvement as a model
promotion. The controls favor the already A8-trained state; every new precision
needs adaptation and its own native qualification before a substantive decision.

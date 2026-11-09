#!/usr/bin/env bash
set -euo pipefail
root=/home/philip/he4-capture-2080ti
export PATH="$root/venv/bin:$PATH"
cmake -S "$root/runtime" -B "$root/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/nvcc -DCUDAToolkit_ROOT=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" --target llama -j 8
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I "$root/runtime/include" -I "$root/runtime/ggml/include" "$root/project/scripts/he4_capture/capture.cpp" -L "$root/build/bin" -Wl,-rpath,"$root/build/bin" -lllama -lggml -lggml-base -ldl -o "$root/capture"

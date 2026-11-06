#!/usr/bin/env bash
set -euo pipefail
copy="$PWD/results/cuda-glibc-compat/include"
source="/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h"
patched="$copy/crt/math_functions.h"
test "$(sha256sum "$source" | cut -d' ' -f1)" = "decdc28efcfaf0aaf806abc96d7bba9cb84b37c6e83cb82cda59b6aa59916ff8"
grep -F 'rsqrt(double x) noexcept (true);' "$patched"
grep -F 'rsqrtf(float x) noexcept (true);' "$patched"
nvcc -std=c++17 -arch=sm_120 -I"$copy" -E results/cuda-glibc-compat/probe.cu -o results/cuda-glibc-compat/probe.ii
grep -F "$patched" results/cuda-glibc-compat/probe.ii >/dev/null
nvcc -std=c++17 -arch=sm_120 -I"$copy" -c results/cuda-glibc-compat/probe.cu -o results/cuda-glibc-compat/probe.o
sha256sum "$source" "$patched" results/cuda-glibc-compat/probe.cu results/cuda-glibc-compat/probe.ii results/cuda-glibc-compat/probe.o

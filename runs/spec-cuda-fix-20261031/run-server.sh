#!/bin/sh
set -eu
CUDA_FIX_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export GGML_CUDA_DECODE_CONSISTENT=32
export LD_LIBRARY_PATH="$CUDA_FIX_DIR/patched-build/bin:$CUDA_FIX_DIR/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$CUDA_FIX_DIR/build/bin/llama-server" "$@"

#!/usr/bin/env bash
set -euo pipefail
results/convert-env/bin/python --version
results/convert-env/bin/python -c 'import torch, transformers, gguf; print("torch",torch.__version__,"cuda",torch.version.cuda); print("transformers",transformers.__version__); print("gguf",getattr(gguf,"__version__","unknown"))'
uv pip freeze --python results/convert-env/bin/python | sort > results/conversion-baseline-20260924/installed-packages.txt
sha256sum results/conversion-baseline-20260924/packages.txt results/conversion-baseline-20260924/installed-packages.txt

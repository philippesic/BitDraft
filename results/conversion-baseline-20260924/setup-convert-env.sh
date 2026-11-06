#!/usr/bin/env bash
set -euo pipefail
test ! -e results/convert-env
uv venv results/convert-env --python 3.11
uv pip install --python results/convert-env/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --requirement results/conversion-baseline-20260924/packages.txt
results/convert-env/bin/python -m pip freeze | sort \
  > results/conversion-baseline-20260924/installed-packages.txt
results/convert-env/bin/python --version
sha256sum results/conversion-baseline-20260924/packages.txt \
  results/conversion-baseline-20260924/installed-packages.txt

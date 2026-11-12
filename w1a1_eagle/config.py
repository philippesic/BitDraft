import os
import sys
import tomllib
import uuid
from pathlib import Path
from typing import cast

from w1a1_eagle.storage import project_root


def numerical_python(root: Path) -> str:
    explicit = os.environ.get("BINARY_NUMERICAL_PYTHON")
    if explicit:
        return explicit
    host = root / ".host.toml"
    if host.exists():
        data = cast(dict[str, object], tomllib.loads(host.read_text()))
        value = data.get("numerical_python")
        if not isinstance(value, str) or not value:
            raise ValueError("host numerical_python must be an executable path")
        return value
    local = root / ".venv/numerical/bin/python"
    return str(local) if local.is_file() else sys.executable


def artifact_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    cache = project_root() / "artifacts/cache"
    temporary = cache / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    environment["PYTHONPYCACHEPREFIX"] = str(temporary / ("bytecode-" + uuid.uuid4().hex))
    for key, folder in (
        ("XDG_CACHE_HOME", "xdg"),
        ("HF_HOME", "huggingface"),
        ("HF_HUB_CACHE", "huggingface/hub"),
        ("HUGGINGFACE_HUB_CACHE", "huggingface/hub"),
        ("HF_DATASETS_CACHE", "huggingface/datasets"),
        ("TORCH_HOME", "torch"),
        ("TORCH_EXTENSIONS_DIR", "torch/extensions"),
        ("PYTORCH_KERNEL_CACHE_PATH", "torch/kernels"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        environment[key] = str(cache / folder)
    return environment


def cpu_environment() -> dict[str, str]:
    environment = artifact_environment()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def execution_environment(device: str) -> dict[str, str]:
    if device == "cpu":
        return cpu_environment()
    if device != "cuda":
        raise ValueError("unsupported execution device")
    return artifact_environment()


def native_environment(libraries: Path, device: str = "cuda") -> dict[str, str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("native device must be cpu or cuda")
    environment = {
        key: value
        for key, value in artifact_environment().items()
        if key == "CUDA_CACHE_PATH"
        or not key.startswith(("GGML_", "W1A8_", "LLAMA_", "CUDA_", "LD_", "OMP_"))
    }
    environment.update(
        LD_LIBRARY_PATH=str(libraries),
        GGML_CUDA_DECODE_CONSISTENT="32",
        CUDA_VISIBLE_DEVICES="0" if device == "cuda" else "",
        OMP_NUM_THREADS="4",
    )
    return environment


def benchmark_environment(root: Path, snapshot: Path) -> dict[str, str]:
    environment = execution_environment("cuda")
    environment["PYTHONPATH"] = str(snapshot)
    environment["BINARY_NUMERICAL_PYTHON"] = numerical_python(root)
    return environment

import os
import sys
import tomllib
import uuid
from pathlib import Path
from typing import cast


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


def cpu_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TMPDIR"] = "/tmp"
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/binary-no-bytecode-" + uuid.uuid4().hex
    return environment


def execution_environment(device: str) -> dict[str, str]:
    if device == "cpu":
        return cpu_environment()
    if device != "cuda":
        raise ValueError("unsupported execution device")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def native_environment(libraries: Path, device: str = "cuda") -> dict[str, str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("native device must be cpu or cuda")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GGML_", "W1A8_", "LLAMA_", "CUDA_", "LD_", "OMP_"))
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

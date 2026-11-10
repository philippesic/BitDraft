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

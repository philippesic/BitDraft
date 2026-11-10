import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path

from w1a1_eagle.contracts import Record
from w1a1_eagle.records import identity


def source_identity(root: Path) -> str:
    values: dict[str, str] = {}
    for path in sorted((root / "w1a1_eagle").rglob("*")):
        if path.suffix == ".py":
            if path.is_symlink():
                raise ValueError("source identity refuses symlinked executable inputs")
            values[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for name in ("uv.lock", "policy/numerical.toml", "pyproject.toml"):
        values[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return identity(values)


def loaded_identity() -> Record:
    executable = Path(sys.executable).resolve()
    paths = {executable}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
            path = Path(fields[5])
            if not path.is_file():
                raise ValueError("loaded executable was deleted or is unreadable")
            paths.add(path)
    files: dict[str, str] = {}
    for path in sorted(paths):
        with path.open("rb") as source:
            files[str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "python": sys.version,
        "executable": str(executable),
        "loaded_executables": files,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "gguf")},
        "pid": os.getpid(),
    }

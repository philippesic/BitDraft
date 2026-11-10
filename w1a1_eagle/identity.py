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


def loaded_identity(cache: dict[tuple[int, int, int, int, int], str] | None = None) -> Record:
    executable = Path(sys.executable).resolve()
    paths = {executable}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "x" in fields[1] and fields[5].startswith("/"):
            path = Path(fields[5])
            if not path.is_file():
                raise ValueError("loaded executable was deleted or is unreadable")
            paths.add(path)
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"w1a1_eagle", "torch", "numpy", "gguf"} and module is not None:
            filename = module.__file__ if hasattr(module, "__file__") else None
            if filename and Path(filename).is_file():
                paths.add(Path(filename).resolve())
    files: dict[str, str] = {}
    hashes = cache if cache is not None else {}
    reused = 0
    for path in sorted(paths):
        before = path.stat()
        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if key in hashes:
            reused += 1
        else:
            with path.open("rb") as source:
                hashes[key] = hashlib.file_digest(source, "sha256").hexdigest()
        after = path.stat()
        if key != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("loaded code file changed while its identity was captured")
        files[str(path)] = hashes[key]
    return {
        "python": sys.version,
        "executable": str(executable),
        "loaded_code_files": files,
        "reused_unchanged_file_hashes": reused,
        "hash_reuse_scope": "same attempt; device/inode/size/mtime/ctime checked; not cached execution or a hostile-filesystem proof",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "gguf")},
        "pid": os.getpid(),
    }

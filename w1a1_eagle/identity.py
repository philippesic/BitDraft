import errno
import hashlib
import importlib.metadata
import os
import shutil
import sys
import tempfile
from pathlib import Path

from w1a1_eagle.capacity import require
from w1a1_eagle.contracts import Record
from w1a1_eagle.records import identity


def source_files(root: Path) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    paths = sorted((root / "w1a1_eagle").rglob("*"))
    paths.extend(sorted((root / "benchmarks").rglob("*.py")))
    paths.extend(sorted((root / "native").rglob("*")))
    paths.extend(root / name for name in ("uv.lock", "policy/numerical.toml", "pyproject.toml"))
    total = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError("source identity refuses symlinked executable inputs")
        if path.is_file() and (
            path.suffix in {".py", ".c", ".h", ".cpp", ".hpp", ".cu", ".cuh"}
            or path.parent == root
            or path.name == "numerical.toml"
        ):
            total += path.stat().st_size
            if total > 2 * 1024 * 1024:
                raise ValueError("runtime source snapshot exceeds its explicit 2 MiB bound")
            values[path.relative_to(root).as_posix()] = path.read_bytes()
    if not values or not all(
        name in values
        for name in ("w1a1_eagle/__init__.py", "uv.lock", "policy/numerical.toml", "pyproject.toml")
    ):
        raise ValueError("runtime source closure is incomplete")
    return values


def source_identity(root: Path) -> str:
    return identity(
        {name: hashlib.sha256(data).hexdigest() for name, data in source_files(root).items()}
    )


def freeze_source(root: Path, destination: Path, expected: str, reserve: int) -> Path:
    values = source_files(root)
    if (
        identity({name: hashlib.sha256(data).hexdigest() for name, data in values.items()})
        != expected
    ):
        raise ValueError("source changed before runtime snapshot")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / expected
    if target.exists():
        if target.is_symlink() or source_identity(target) != expected:
            raise ValueError("existing source snapshot differs from admitted identity")
        return target
    require(destination, 2 * 1024 * 1024, reserve)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-source-", dir=destination))
    try:
        for name, data in values.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            path.chmod(0o444)
        if source_identity(temporary) != expected:
            raise ValueError("runtime snapshot verification failed")
        try:
            os.rename(temporary, target)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            if target.is_symlink() or not target.is_dir() or source_identity(target) != expected:
                raise ValueError(
                    "concurrent source snapshot differs from admitted identity"
                ) from error
        descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


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

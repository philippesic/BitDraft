import json
import shutil
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from dulwich.ignore import IgnoreFilterManager
from dulwich.index import IndexEntry
from dulwich.repo import Repo

from tools.contracts import Record
from tools.policy_review import content_digest as content_digest


def read_toml(path: Path) -> Record:
    return cast(Record, tomllib.loads(path.read_text()))


def read_json(path: Path) -> Record:
    return cast(Record, json.loads(path.read_text()))


def inventory(root: Path) -> tuple[dict[str, str], list[str]]:
    repo = Repo(str(root))
    try:
        index = repo.open_index()
        names = {p.decode() for p in index}
        ignore = IgnoreFilterManager.from_repo(repo)
        escapes: list[str] = []
        pending = [root]
        while pending:
            directory = pending.pop()
            for child in sorted(directory.iterdir()):
                relative = child.relative_to(root).as_posix()
                if relative.split("/")[0] in {".git", ".venv"}:
                    continue
                if child.is_symlink():
                    escapes.append(relative)
                elif child.is_dir():
                    if not ignore.is_ignored(relative + "/"):
                        pending.append(child)
                elif not ignore.is_ignored(relative):
                    names.add(relative)
        files: dict[str, str] = {}
        for name in sorted(names):
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                escapes.append(name)
                continue
            files[name] = (
                path.read_bytes().decode("utf-8", errors="replace") if path.exists() else ""
            )
        return files, sorted(set(escapes))
    finally:
        repo.close()


@contextmanager
def temporary_root(base: Path) -> Iterator[Path]:
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="foundation-", dir=base)).resolve()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def index_matches(root: Path, files: dict[str, str]) -> bool:
    repo = Repo(str(root))
    try:
        index = repo.open_index()
        if set(files) != {p.decode() for p in index}:
            return False
        for path in index:
            entry = index[path]
            if not isinstance(entry, IndexEntry):
                return False
            if (
                repo.object_store[entry.sha].as_raw_string()
                != files.get(path.decode(), "").encode()
            ):
                return False
        return True
    finally:
        repo.close()


def hooks_path(root: Path) -> str:
    repo = Repo(str(root))
    try:
        value = repo.get_config().get((b"core",), b"hooksPath")
        return value.decode()
    except KeyError:
        return ""
    finally:
        repo.close()

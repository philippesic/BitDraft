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

from tools.contracts import Record, sequence
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


def validate_yaml(text: str) -> bool:
    import yaml

    try:
        root = cast(object, yaml.compose(text, Loader=yaml.SafeLoader))
        pending: list[object] = [root] if root is not None else []
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            kind = cast(object, getattr(node, "id", None))
            value = cast(object, getattr(node, "value", None))
            if kind == "mapping":
                keys: set[str] = set()
                for item in sequence(value):
                    if not isinstance(item, tuple) or len(item) != 2:
                        return False
                    key, child = cast(tuple[object, object], item)
                    key_value = cast(object, getattr(key, "value", None))
                    if not isinstance(key_value, str) or key_value in keys:
                        return False
                    keys.add(key_value)
                    pending.append(child)
            elif kind == "sequence":
                pending.extend(sequence(value))
        return True
    except (yaml.YAMLError, ValueError):
        return False

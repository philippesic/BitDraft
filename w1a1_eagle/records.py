import hashlib
import json
from pathlib import Path
from typing import cast

from w1a1_eagle.contracts import Record, integer, record, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.json_stream import invalid_constant, unique
from w1a1_eagle.storage import relocated


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path: Path, limit: int = 16 * 1024 * 1024) -> Record:
    path = relocated(path)
    if path.stat().st_size > limit:
        raise ValueError("record byte limit exceeded")

    return record(
        cast(
            object,
            json.loads(
                path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid_constant
            ),
        )
    )


def artifact(value: object, root: Path, resolve: bool = True) -> Artifact:
    raw = record(value)
    if set(raw) != {"path", "sha256", "bytes"}:
        raise ValueError("unsupported artifact record")
    path = Path(string(raw["path"]))
    selected = relocated(root / path)
    return Artifact(
        selected.resolve() if resolve else selected, string(raw["sha256"]), integer(raw["bytes"])
    )


def reference(value: Artifact) -> Record:
    return {"path": str(value.path), "sha256": value.sha256, "bytes": value.size}

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from w1a1_eagle.contracts import record
from w1a1_eagle.data import Artifact


@dataclass(frozen=True)
class StateRef:
    artifact: Artifact
    config: str
    updates: int

    def __post_init__(self) -> None:
        if not self.config or type(self.updates) is not int or self.updates < 0:
            raise ValueError("invalid state identity")


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def atomic(self, path: Path, payload: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def commit(self, payload: bytes, config: str, updates: int) -> StateRef:
        digest = hashlib.sha256(payload).hexdigest()
        reference = StateRef(
            Artifact(self.root / (digest + ".bin"), digest, len(payload)), config, updates
        )
        with (self.root / ".writer").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.atomic(reference.artifact.path, payload)
            reference.artifact.verify()
            manifest = {
                "schema": 1,
                "sha256": digest,
                "bytes": len(payload),
                "config": config,
                "updates": updates,
            }
            self.atomic(self.root / "latest.json", json.dumps(manifest, sort_keys=True).encode())
        return reference

    def latest(self, config: str) -> StateRef:
        raw = record(cast(object, json.loads((self.root / "latest.json").read_text())))
        if set(raw) != {
            "schema",
            "sha256",
            "bytes",
            "config",
            "updates",
        }:
            raise ValueError("unsupported state manifest")
        if type(raw["schema"]) is not int or raw["schema"] != 1 or raw["config"] != config:
            raise ValueError("state schema or configuration mismatch")
        digest, size, updates = raw["sha256"], raw["bytes"], raw["updates"]
        if not isinstance(digest, str) or type(size) is not int or type(updates) is not int:
            raise ValueError("malformed state manifest")
        reference = StateRef(Artifact(self.root / (digest + ".bin"), digest, size), config, updates)
        reference.artifact.verify()
        return reference

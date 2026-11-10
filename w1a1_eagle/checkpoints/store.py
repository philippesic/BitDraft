import fcntl
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, cast

from w1a1_eagle.capacity import require
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
    def __init__(self, root: Path, reserve_bytes: int = 0) -> None:
        self.root = root.resolve()
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("invalid store disk reserve")
        self.reserve_bytes = reserve_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def atomic(self, path: Path, payload: bytes) -> None:
        require(self.root, len(payload), self.reserve_bytes)
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
        def write(output: BinaryIO) -> None:
            output.write(payload)

        return self.commit_with(write, config, updates, max(1, len(payload)))

    def commit_with(
        self, writer: Callable[[BinaryIO], None], config: str, updates: int, max_bytes: int
    ) -> StateRef:
        if not isinstance(config, str) or not config or type(updates) is not int or updates < 0:
            raise ValueError("invalid state identity")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("positive state byte budget required")
        with (self.root / ".writer").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
            path = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    writer(
                        cast(
                            BinaryIO,
                            BoundedSink(
                                output,
                                max_bytes,
                                lambda count: require(self.root, count, self.reserve_bytes),
                            ),
                        )
                    )
                    output.flush()
                    os.fsync(output.fileno())
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                size = path.stat().st_size
                reference = StateRef(
                    Artifact(self.root / (digest + ".bin"), digest, size), config, updates
                )
                os.replace(path, reference.artifact.path)
                reference.artifact.verify()
                manifest = {
                    "schema": 1,
                    "sha256": digest,
                    "bytes": size,
                    "config": config,
                    "updates": updates,
                }
                self.atomic(
                    self.root / (digest + ".json"), json.dumps(manifest, sort_keys=True).encode()
                )
                self.atomic(
                    self.root / "latest.json", json.dumps(manifest, sort_keys=True).encode()
                )
            finally:
                path.unlink(missing_ok=True)
        return reference

    def latest(self, config: str) -> StateRef:
        raw = record(cast(object, json.loads((self.root / "latest.json").read_text())))
        return self.parse(raw, config)

    def describe(self, artifact: Artifact) -> StateRef:
        raw = record(cast(object, json.loads(artifact.path.with_suffix(".json").read_text())))
        config = raw.get("config")
        if not isinstance(config, str):
            raise ValueError("missing parent configuration")
        reference = self.parse(raw, config)
        if reference.artifact != artifact:
            raise ValueError("parent manifest disagrees with explicit artifact")
        return reference

    def parse(self, raw: dict[str, object], config: str) -> StateRef:
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


class BoundedSink(io.BufferedIOBase):
    def __init__(self, output: BinaryIO, limit: int, before_write: Callable[[int], None]) -> None:
        super().__init__()
        self.output = output
        self.limit = limit
        self.count = 0
        self.before_write = before_write

    def write(self, data: object) -> int:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("state writer requires a byte buffer")
        if self.count + memoryview(data).nbytes > self.limit:
            raise ValueError("state byte budget exceeded")
        self.before_write(memoryview(data).nbytes)
        written = self.output.write(data)
        self.count += written
        return written

    def tell(self) -> int:
        return self.count

    def flush(self) -> None:
        if not self.output.closed:
            self.output.flush()

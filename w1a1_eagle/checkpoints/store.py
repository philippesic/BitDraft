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
        if path.parent.resolve() != self.root:
            raise ValueError("store publication escaped its owned root")
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

    def stream(self, name: str, writer: Callable[[BinaryIO], None], max_bytes: int) -> Artifact:
        if Path(name).name != name or name in {"", ".", ".."} or max_bytes < 1:
            raise ValueError("explicit bounded asset filename required")
        destination = self.root / name
        if destination.exists() or destination.is_symlink():
            raise ValueError("streamed asset must be new")
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as output:
                sink = BoundedSink(
                    output, max_bytes, lambda count: require(self.root, count, self.reserve_bytes)
                )
                writer(cast(BinaryIO, sink))
                output.flush()
                os.fsync(output.fileno())
            if Path(temporary).stat().st_size != sink.count:
                raise ValueError("streamed asset size differs")
            os.link(temporary, destination)
            artifact = Artifact(destination, sink.digest.hexdigest(), sink.count)
            artifact.verify()
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return artifact
        finally:
            Path(temporary).unlink(missing_ok=True)

    def commit(self, payload: bytes, config: str, updates: int) -> StateRef:
        def write(output: BinaryIO) -> None:
            output.write(payload)

        return self.commit_with(write, config, updates, max(1, len(payload)))

    def commit_with(
        self,
        writer: Callable[[BinaryIO], None],
        config: str,
        updates: int,
        max_bytes: int,
        validator: Callable[[Artifact], None] | None = None,
    ) -> StateRef:
        if not isinstance(config, str) or not config or type(updates) is not int or updates < 0:
            raise ValueError("invalid state identity")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("positive state byte budget required")
        with (self.root / ".writer").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.root / "latest.json").exists():
                self.latest(config)
            descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
            path = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    sink = BoundedSink(
                        output,
                        max_bytes,
                        lambda count: require(self.root, count, self.reserve_bytes),
                    )
                    writer(cast(BinaryIO, sink))
                    output.flush()
                    os.fsync(output.fileno())
                digest = sink.digest.hexdigest()
                size = path.stat().st_size
                if size != sink.count:
                    raise ValueError("state stream size mismatch")
                if validator is not None:
                    validator(Artifact(path, digest, size))
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

    def prune(self, keep: int, pinned: tuple[str, ...] = ()) -> tuple[str, ...]:
        if type(keep) is not int or keep < 1:
            raise ValueError("positive retained-state count required")
        with (self.root / ".writer").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            latest = record(cast(object, json.loads((self.root / "latest.json").read_text())))
            protected = {str(latest["sha256"]), *pinned}
            available: list[tuple[int, str, dict[str, object]]] = []
            for path in self.root.glob("*.json"):
                digest = path.stem
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    continue
                payload = self.root / (digest + ".bin")
                if not payload.exists():
                    continue
                metadata = record(cast(object, json.loads(path.read_text())))
                if (
                    metadata.get("sha256") != digest
                    or metadata.get("config") != latest["config"]
                    or type(metadata.get("updates")) is not int
                ):
                    raise ValueError("retention refuses foreign or malformed state metadata")
                available.append((cast(int, metadata["updates"]), digest, metadata))
            present = {digest for _, digest, _ in available}
            if str(latest["sha256"]) not in present:
                raise ValueError("retention cannot proceed without its latest payload")
            retained = protected & present
            if len(retained) > keep:
                raise ValueError("retention bound cannot preserve all pinned states")
            for _, digest, _ in sorted(available, reverse=True):
                if len(retained) < keep:
                    retained.add(digest)
            retired: list[dict[str, object]] = []
            for _, digest, metadata in available:
                if digest in retained:
                    continue
                path = self.root / (digest + ".bin")
                status = path.lstat()
                if (
                    path.is_symlink()
                    or status.st_uid != os.getuid()
                    or status.st_nlink != 1
                    or status.st_size != metadata["bytes"]
                ):
                    raise ValueError("retention refuses changed, shared or foreign payload")
                path.unlink()
                retired.append(metadata)
            if retired:
                receipt: dict[str, object] = {
                    "keep": keep,
                    "pinned": list(pinned),
                    "retained": sorted(retained),
                    "retired": retired,
                }
                self.atomic(
                    self.root / ("retired-" + str(latest["sha256"]) + ".json"),
                    json.dumps(receipt, sort_keys=True).encode(),
                )
            return tuple(str(row["sha256"]) for row in retired)

    def describe(self, artifact: Artifact) -> StateRef:
        raw = record(cast(object, json.loads(artifact.path.with_suffix(".json").read_text())))
        config = raw.get("config")
        if not isinstance(config, str):
            raise ValueError("missing parent configuration")
        reference = self.parse(raw, config)
        if reference.artifact != artifact:
            raise ValueError("parent manifest disagrees with explicit artifact")
        return reference

    def retire_closed(self, protected: frozenset[Path], recovery: Artifact) -> tuple[str, ...]:
        recovery.verify()
        retired: list[str] = []
        with (self.root / ".writer").open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in sorted(self.root.glob("*.bin")):
                if path in protected:
                    continue
                status = path.lstat()
                if path.is_symlink() or status.st_uid != os.getuid() or status.st_nlink != 1:
                    raise ValueError("closed retention refuses foreign or shared payload")
                metadata = record(cast(object, json.loads(path.with_suffix(".json").read_text())))
                candidate = Artifact(path, path.stem, status.st_size)
                self.describe(candidate)
                receipt: dict[str, object] = {
                    "state": metadata,
                    "recovery": str(recovery.path),
                    "recovery_sha256": recovery.sha256,
                    "reason": "closed unselected campaign segment; exact retired resume point unavailable",
                }
                self.atomic(
                    self.root / ("retirement-" + path.stem + ".json"),
                    json.dumps(receipt, sort_keys=True).encode(),
                )
                path.unlink()
                retired.append(path.stem)
        return tuple(retired)

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
        self.digest = hashlib.sha256()
        self.before_write = before_write

    def write(self, data: object) -> int:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("state writer requires a byte buffer")
        if self.count + memoryview(data).nbytes > self.limit:
            raise ValueError("state byte budget exceeded")
        self.before_write(memoryview(data).nbytes)
        written = self.output.write(data)
        if written != memoryview(data).nbytes:
            raise OSError("short state write")
        self.digest.update(data)
        self.count += written
        return written

    def tell(self) -> int:
        return self.count

    def flush(self) -> None:
        if not self.output.closed:
            self.output.flush()

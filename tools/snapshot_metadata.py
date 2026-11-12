import argparse
import gzip
import hashlib
import io
import json
import signal
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, cast

from w1a1_eagle.storage import output_path, relocated


def object_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected metadata object")
    return cast(dict[str, object], value)


def items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected metadata list")
    return cast(list[object], value)


def text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected metadata text")
    return value


def pack(
    index: Path, root: Path, output: BinaryIO, expected: str, maximum: int
) -> dict[str, object]:
    if index.stat().st_size > maximum:
        raise ValueError("index exceeds snapshot bound")
    raw = index.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("index hash mismatch")
    document = object_record(cast(object, json.loads(raw)))
    root = root.resolve()
    references: dict[str, tuple[Path, str, int]] = {}

    def include(value: object) -> None:
        ref = object_record(value)
        path = Path(text(ref["path"]))
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("metadata reference escaped snapshot root")
        if path.suffix != ".json":
            raise ValueError("snapshot accepts JSON metadata only")
        size = ref["bytes"]
        if type(size) is not int or not 0 <= size <= 16 * 1024 * 1024:
            raise ValueError("metadata file size outside bound")
        name = path.relative_to(root).as_posix()
        if name == "index.json":
            raise ValueError("capture metadata collides with index archive name")
        entry = (path, text(ref["sha256"]), size)
        if name in references and references[name] != entry:
            raise ValueError("conflicting metadata references")
        references[name] = entry

    include(document["exclusions"])
    for value in items(document["rows"]):
        row = object_record(value)
        include(row["receipt_pin"])
        receipt = object_record(row["receipt"])
        include(receipt["metadata"])
        sources = [
            ref
            for ref in items(receipt["files"])
            if Path(text(object_record(ref)["path"])).name == "source.json"
        ]
        if len(sources) != 1:
            raise ValueError("one source identity record required per capture")
        include(sources[0])
    total = len(raw) + sum(entry[2] for entry in references.values())
    if total > maximum:
        raise ValueError("metadata snapshot exceeds declared byte bound")
    with (
        gzip.GzipFile(fileobj=output, mode="wb", compresslevel=1, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):

        def add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(payload))

        add("index.json", raw)
        for name, (path, digest, size) in sorted(references.items()):
            payload = path.read_bytes()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("metadata changed or failed identity: " + name)
            add(name, payload)
    return {
        "schema": 1,
        "index_sha256": expected,
        "files": len(references) + 1,
        "bytes": total,
        "scope": "pinned index and JSON receipts/source identities only; no tensor files",
    }


def unpack(archive_path: Path, destination: Path, expected: str, maximum: int) -> dict[str, object]:
    destination.mkdir(mode=0o700, parents=True)
    destination = destination.resolve()
    total = count = 0
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or name.suffix != ".json"
                or member.name in seen
            ):
                raise ValueError("unsafe or duplicate metadata archive member")
            seen.add(member.name)
            total += member.size
            if (
                total > maximum
                or member.size < 0
                or member.name != "index.json"
                and member.size > 16 * 1024 * 1024
            ):
                raise ValueError("metadata archive exceeds byte bound")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("missing metadata archive payload")
            with source, target.open("xb") as output:
                while data := source.read(1024 * 1024):
                    output.write(data)
            count += 1
    index = destination / "index.json"
    with index.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError("unpacked index hash mismatch")
    return {
        "schema": 1,
        "files": count,
        "bytes": total,
        "index_sha256": expected,
        "scope": "transport integrity only; run metadata admission separately",
    }


class Arguments(argparse.Namespace):
    mode: str
    index: Path
    root: Path
    archive: Path
    destination: Path
    expected: str
    maximum: int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--maximum", type=int, default=512 * 1024 * 1024)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    packing = subcommands.add_parser("pack")
    packing.add_argument("index", type=Path)
    packing.add_argument("root", type=Path)
    unpacking = subcommands.add_parser("unpack")
    unpacking.add_argument("archive", type=Path)
    unpacking.add_argument("destination", type=Path)
    args = parser.parse_args(namespace=Arguments())
    if args.maximum < 1:
        raise ValueError("positive snapshot byte bound required")
    signal.alarm(180)
    if args.mode == "pack":
        result = pack(
            args.index, args.root, cast(BinaryIO, sys.stdout.buffer), args.expected, args.maximum
        )
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
    else:
        result = unpack(
            relocated(args.archive),
            output_path(args.destination, "datasets", args.archive.stem),
            args.expected,
            args.maximum,
        )
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

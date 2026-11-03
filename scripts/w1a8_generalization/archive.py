"""Lossless archival of this experiment's completed checkpoint publications.

Default archival preserves the original. Explicit removal requires a verified
gzip round trip, a durable recovery receipt and unchanged original/archive files.
Do not run compression during clean serving timing.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from recovery import atomic_json

BUFFER = 1024**2


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def copy_hash(source, destination):
    checksum, size = hashlib.sha256(), 0
    while block := source.read(BUFFER):
        checksum.update(block)
        size += len(block)
        if destination is not None:
            destination.write(block)
    return checksum.hexdigest(), size


def completed_owner(state_path):
    state_path = Path(state_path)
    state = json.loads(state_path.read_text())
    owner = json.loads(state_path.with_name("identity.json").read_text())
    if state.get("status") != "finished" or state.get("exit_code") != 0:
        raise ValueError("successful terminal supervisor required before archival")
    same_boot = owner["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if same_boot:
        for role in ("supervisor", "child"):
            ref = owner[role]
            process = Path("/proc") / str(ref["pid"])
            if process.exists():
                try:
                    birth = int((process / "stat").read_text().rsplit(")", 1)[1].split()[19])
                except FileNotFoundError:
                    continue
                if birth == ref["start_ticks"]:
                    raise ValueError("owned training process is still alive")
    return {"path": str(state_path.resolve()), "sha256": digest(state_path)}


def archive(ref, output, *, owned_root, owner_state, remove_source=False, floor_bytes=2 * 1024**3):
    source, output, root = Path(ref["path"]), Path(output), Path(owned_root).resolve()
    if source.is_symlink() or not source.resolve().is_relative_to(root):
        raise ValueError("only an owned non-symlink checkpoint may be archived")
    if source.suffix != ".pt" or output.exists() or output.is_symlink():
        raise ValueError("existing checkpoint and exclusive new archive required")
    owner = completed_owner(owner_state)
    state = json.loads(Path(owner_state).read_text())
    command = state.get("command", [])
    if "--output" not in command:
        raise ValueError("training output must be bound by the completed owner command")
    declared = Path(command[command.index("--output") + 1])
    if not declared.is_absolute():
        declared = Path(owner_state).resolve().parents[2] / declared
    if source.resolve().parent != declared.resolve():
        raise ValueError("checkpoint does not belong to this completed training output")
    before = identity(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    free_before = {str(path.resolve()): shutil.disk_usage(path).free
                   for path in (source.parent, output.parent)}
    if shutil.disk_usage(output.parent).free < source.stat().st_size + floor_bytes:
        raise OSError("reserve worst-case archive bytes plus filesystem floor")
    partial = output.with_name(output.name + ".partial")
    started = time.perf_counter()
    with source.open("rb") as stream, partial.open("xb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, compresslevel=1, mtime=0) as compressed:
            raw_sha, size = copy_hash(stream, compressed)
        destination.flush()
        os.fsync(destination.fileno())
    if raw_sha != ref["sha256"] or identity(source) != before:
        raise ValueError("original changed during archival")
    with gzip.open(partial, "rb") as stream:
        check_sha, check_size = copy_hash(stream, None)
    if (check_sha, check_size) != (raw_sha, size):
        raise ValueError("gzip round trip differs from authenticated checkpoint")
    archive_sha = digest(partial)
    os.link(partial, output)
    partial.unlink()
    archived_identity = identity(output)
    receipt = {"schema": "w1a8_checkpoint_gzip_v1", "original": ref,
               "archive": {"path": str(output.resolve()), "sha256": archive_sha},
               "original_bytes": size, "archive_bytes": output.stat().st_size,
               "gzip_roundtrip_verified": True, "source_identity": before,
               "owner_supervisor": owner, "source_removed": False,
               "remove_source_requested": remove_source, "wall_seconds": time.perf_counter() - started,
               "free_bytes_before": free_before}
    receipt_path = output.with_name(output.name + ".json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    atomic_json(receipt_path, receipt)
    if remove_source:
        if identity(source) != before or identity(output) != archived_identity:
            raise ValueError("source or verified archive changed before retirement")
        source.unlink()
        descriptor = os.open(source.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        receipt["source_removed"] = True
    receipt["free_bytes_after"] = {name: shutil.disk_usage(name).free for name in free_before}
    atomic_json(receipt_path, receipt)
    return receipt


def restore(receipt, output):
    if receipt.get("schema") != "w1a8_checkpoint_gzip_v1" or receipt.get("gzip_roundtrip_verified") is not True:
        raise ValueError("verified experiment archival receipt required")
    source, output = Path(receipt["archive"]["path"]), Path(output)
    if output.exists() or output.is_symlink() or digest(source) != receipt["archive"]["sha256"]:
        raise ValueError("unchanged archive and exclusive recovery destination required")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < receipt["original_bytes"] + 2 * 1024**3:
        raise OSError("reserve complete restored bytes plus filesystem floor")
    partial = output.with_name(output.name + ".partial")
    with gzip.open(source, "rb") as stream, partial.open("xb") as destination:
        checksum, size = copy_hash(stream, destination)
        destination.flush()
        os.fsync(destination.fileno())
    if (checksum, size) != (receipt["original"]["sha256"], receipt["original_bytes"]):
        raise ValueError("restored checkpoint differs from original publication")
    os.link(partial, output)
    partial.unlink()
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(output.resolve()), "sha256": checksum, "bytes": size}


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("archive")
    p.add_argument("--pin", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--owned-root", type=Path, required=True)
    p.add_argument("--owner-state", type=Path, required=True)
    p.add_argument("--remove-source", action="store_true")
    p = commands.add_parser("restore")
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "archive":
        result = archive(json.loads(args.pin.read_text()), args.output, owned_root=args.owned_root,
                         owner_state=args.owner_state, remove_source=args.remove_source)
    else:
        result = restore(json.loads(args.receipt.read_text()), args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

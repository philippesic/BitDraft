"""Bounded recovery publications and chunked frozen-operand verification.

Only checkpoints created by this publisher in its own output directory may be
retired. A verified newer state is published before the oldest recovery point
is removed. Existing inputs, selected endpoints and other runs are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def chunked_frozen(model, references, slots, *, chunk_bytes=4 * 1024**2):
    """Hash original tensor bytes without a whole embedding/head GPU->CPU copy."""
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("positive chunk byte budget required")
    result = {}
    for name, (owner, attribute) in slots(model).items():
        value = getattr(owner, attribute)
        if value.requires_grad:
            continue
        ref = references[name]
        if list(value.shape) != ref["shape"]:
            raise ValueError("ordinary frozen shape changed: " + name)
        digest = hashlib.sha256()
        tensor = value.detach()
        if tensor.ndim == 0:
            digest.update(tensor.cpu().contiguous().numpy().tobytes())
        else:
            row_bytes = tensor.element_size()
            for dimension in tensor.shape[1:]:
                row_bytes *= dimension
            if row_bytes > chunk_bytes:
                raise ValueError("single frozen row exceeds verification byte budget")
            rows = max(1, chunk_bytes // max(1, row_bytes))
            for start in range(0, tensor.shape[0], rows):
                chunk = tensor.narrow(0, start, min(rows, tensor.shape[0] - start)).cpu().contiguous().numpy()
                digest.update(memoryview(chunk))
                del chunk
        actual = digest.hexdigest()
        if actual != ref["actor_f32_sha256"]:
            raise ValueError("ordinary frozen buffer changed: " + name)
        result[name] = {"sha256": actual, "shape": list(value.shape), "dtype": str(value.dtype)}
    return result


class RecoveryCheckpoints:
    def __init__(self, output, publisher, verify, *, keep=2, bound_bytes=4 * 1024**3,
                 floor_bytes=2 * 1024**3):
        self.output = Path(output).resolve()
        if type(keep) is not int or keep < 1 or bound_bytes <= 0 or floor_bytes < 0:
            raise ValueError("valid recovery retention and storage bounds required")
        self.publisher, self.verify = publisher, verify
        self.keep, self.bound_bytes, self.floor_bytes = keep, bound_bytes, floor_bytes
        self.entries = []
        self.latest = None

    def publish(self, step, state):
        if type(step) is not int or step <= 0 or state["local_updates"] != step:
            raise ValueError("exact positive completed optimizer boundary required")
        if self.latest is not None and self.latest["local_updates"] == step:
            return {k: v for k, v in self.latest.items() if k != "local_updates"}
        if shutil.disk_usage(self.output).free < self.bound_bytes + self.floor_bytes:
            raise OSError("insufficient space for full recovery publication plus floor")
        path = self.output / ("checkpoint-%08d.pt" % step)
        ref = self.publisher(path, state, lambda saved: self.verify(state, saved))
        if Path(ref["path"]).resolve() != path or path.is_symlink() or path.stat().st_size > self.bound_bytes:
            raise ValueError("unexpected recovery publication path or size")
        self.latest = {**ref, "local_updates": step}
        self.entries.append({"checkpoint": self.latest, "retired": False})
        # Durable pointer is written before any older recovery state is retired.
        atomic_json(self.output / "latest.json", self.latest)
        active = [entry for entry in self.entries if not entry["retired"]]
        for entry in active[:-self.keep]:
            old = entry["checkpoint"]
            old_path = Path(old["path"])
            if old_path.is_symlink() or old_path.resolve().parent != self.output:
                raise ValueError("refuse recovery cleanup outside this publisher")
            with old_path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != old["sha256"]:
                raise ValueError("refuse recovery cleanup of changed checkpoint")
            entry.update(retirement_prepared=True, replacement=self.latest)
            atomic_json(self.output / "checkpoint-history.json", self.entries)
            old_path.unlink()
            entry["retired"] = True
        atomic_json(self.output / "checkpoint-history.json", self.entries)
        return ref

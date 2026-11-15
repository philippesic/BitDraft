import hashlib
from pathlib import Path

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, record, string
from w1a1_eagle.records import artifact

RSR_COMMIT = "99bcccded70416839f1ef1fc6d58726c507d1fb9"


def rsr_kernel(source: str) -> str:
    start, end = "namespace {", "}  // namespace"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("RSR device source boundaries differ")
    body = source[source.index(start) : source.index(end) + len(end)]
    if "rsr_gemv_v5_9_kernel" not in body:
        raise ValueError("RSR v5.9 kernel missing")
    return "namespace rsr_upstream {\n" + body + "\n}\n"


def install_rsr(raw: Record, root: Path, output: Path) -> Record:
    source = artifact(raw["source"], root)
    licence = artifact(raw["licence"], root)
    source.verify()
    licence.verify()
    if (
        source.sha256 != "9950d5885a8c8dc4f394c77f11213a8a7850c871f0ee095dfa20ac34787ae072"
        or licence.sha256 != "8969d1d2ea3796712833f8722ae51d69600c57851367ea830950d9d477ea486a"
    ):
        raise ValueError("RSR source or licence differs from reviewed commit")
    if string(raw["commit"]) != RSR_COMMIT:
        raise ValueError("unreviewed RSR commit")
    text = licence.path.read_text()
    if "MIT License" not in text or "Copyright (c) 2026 IndexLab" not in text:
        raise ValueError("RSR licence differs from reviewed notice")
    body = rsr_kernel(source.path.read_text())
    owner = Store(output)
    owner.atomic(output / "rsr-upstream.cuh", body.encode())
    owner.atomic(output / "RSR-LICENSE", licence.path.read_bytes())
    return {**raw, "device_sha256": hashlib.sha256(body.encode()).hexdigest()}


def integer_identity(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.startswith("integers,")]
    if len(lines) != 1 or lines[0] not in {"integers,0", "integers,1"}:
        raise ValueError("RSR integer identity record missing or malformed")
    return lines[0] == "integers,1"


def index_entry(manifest: Record, name: str, block: int) -> Record:
    return record(record(manifest["tensors"])[name + "." + str(block)])

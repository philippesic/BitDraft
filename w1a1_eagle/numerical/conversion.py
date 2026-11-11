import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import torch

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.captures import TAPS, FloatArray, HalfArray
from w1a1_eagle.records import artifact, canonical, read, reference


def convert(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) != {"schema", "corpus", "maximum_relative_l2", "capture_limit_bytes"}
        or raw["schema"] != 1
    ):
        raise ValueError("unsupported feature conversion specification")
    source = artifact(raw["corpus"], specification.resolve().parent)
    source.verify()
    threshold = raw["maximum_relative_l2"]
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("explicit positive FP16 rounding threshold required")
    corpus = read(source.path, 64 * 1024 * 1024)
    if corpus.get("schema") != 1 or output.exists():
        raise ValueError("explicit corpus and new conversion root required")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[2]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    rows: list[Record] = []
    measurements: list[Record] = []
    written = 0
    torch.set_num_threads(2)
    for value in sequence(corpus["records"]):
        row = dict(record(value))
        name = string(row["id"])
        metadata_ref = artifact(row["metadata"], source.path.parent)
        metadata_ref.verify()
        metadata = read(metadata_ref.path)
        if metadata.get("feature_dtype", "float32") != "float32" or metadata.get("taps") != list(
            TAPS
        ):
            raise ValueError("conversion requires original F32 eight-tap captures")
        count = len(sequence(metadata["input_tokens"])) * 2560
        cell = Store(store.root / hashlib.sha256(name.encode()).hexdigest())
        files = []
        by_name = {
            artifact(v, source.path.parent).path.name: artifact(v, source.path.parent)
            for v in sequence(row["files"])
        }
        for input_ref in by_name.values():
            input_ref.verify()
        for tap in TAPS:
            ref = by_name.get(f"layer-{tap}.f32")
            if ref is None or ref.size != count * 4:
                raise ValueError("F32 feature geometry mismatch")
            written += count * 2
            if written > integer(raw["capture_limit_bytes"]):
                raise ValueError("converted feature payload exceeds authorized bound")
            mapped = cast(FloatArray, np.memmap(ref.path, dtype="<f4", mode="r", shape=(count,)))
            error_squared = 0.0
            norm_squared = 0.0
            maximum = 0.0

            def write(sink: BinaryIO) -> None:
                nonlocal error_squared, norm_squared, maximum
                for start in range(0, count, 1024 * 1024):
                    array = cast(
                        FloatArray, np.array(mapped[start : start + 1024 * 1024], copy=True)
                    )
                    original = torch.from_numpy(array)
                    rounded = original.half()
                    if not bool(torch.isfinite(original).all()) or not bool(
                        torch.isfinite(rounded).all()
                    ):
                        raise ValueError("nonfinite input or FP16 overflow; preserve F32 source")
                    delta = rounded.float().double() - original.double()
                    error_squared += float(delta.square().sum())
                    norm_squared += float(original.double().square().sum())
                    maximum = max(maximum, float(delta.abs().max()))
                    encoded = cast(HalfArray, rounded.numpy())
                    sink.write(encoded.tobytes())

            converted = cell.stream(f"layer-{tap}.f16", write, count * 2)
            relative = math.sqrt(error_squared / norm_squared) if norm_squared else 0.0
            measurements.append(
                {
                    "id": name,
                    "tap": tap,
                    "relative_l2": relative,
                    "max_absolute": maximum,
                    "source": reference(ref),
                    "converted": reference(converted),
                }
            )
            if relative > threshold:
                raise ValueError("FP16 rounding exceeds the frozen numerical margin")
            files.append(reference(converted))
        for name, ref in by_name.items():
            if name in {f"layer-{tap}.f32" for tap in TAPS}:
                continue
            if ref.size > 16 * 1024 * 1024:
                raise ValueError("supporting file exceeds conversion copy bound")
            cell.atomic(cell.root / name, ref.path.read_bytes())
            copied = artifact(
                {"path": str(cell.root / name), "sha256": ref.sha256, "bytes": ref.size}, cell.root
            )
            copied.verify()
            files.append(reference(copied))
        metadata.update({"feature_dtype": "float16", "conversion_source": reference(metadata_ref)})
        cell.atomic(cell.root / "converted-metadata.json", canonical(metadata))
        path = cell.root / "converted-metadata.json"
        row["metadata"] = {
            "path": str(path),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        row["files"] = files
        rows.append(row)
        store.atomic(store.root / "measurements.json", canonical({"arrays": measurements}))
    converted_corpus = dict(corpus)
    converted_corpus["records"] = rows
    store.atomic(store.root / "corpus.json", canonical(converted_corpus))
    corpus_path = store.root / "corpus.json"
    result: Record = {
        "status": "converted",
        "source": reference(source),
        "feature_bytes": written,
        "arrays": len(measurements),
        "maximum_relative_l2": max(number(v["relative_l2"]) for v in measurements),
        "source_sha256": code,
        "corpus": {
            "path": str(corpus_path),
            "sha256": file_hash(corpus_path),
            "bytes": corpus_path.stat().st_size,
        },
        "roles_preserved": True,
        "source_deleted": False,
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Arguments(argparse.Namespace):
    specification: Path
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(namespace=Arguments())
    try:
        print(json.dumps(convert(args.specification, args.output), indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        refusal: Record = {"status": "refused", "error": str(error)}
        print(json.dumps(refusal))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

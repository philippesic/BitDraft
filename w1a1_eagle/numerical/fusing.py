import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.model import Geometry
from w1a1_eagle.numerical.captures import (
    FUSED_SEMANTICS,
    TAP_SEMANTICS,
    TAPS,
    FloatArray,
    HalfArray,
)
from w1a1_eagle.numerical.weights import fusion_identity, load_he4
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.storage import output_path, relocated

SPECIFICATION_FIELDS = {
    "schema",
    "corpus",
    "corpus_format",
    "source",
    "vocabulary",
    "frozen_format",
    "fused_dtype",
    "maximum_relative_l2",
    "capture_limit_bytes",
    "memory_bytes",
    "device",
    "rows",
}
OPTIONAL_FIELDS = {"frozen_baseline", "initialization", "geometry", "disk_reserve_bytes"}
SUPPORT_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class Fusion:
    operand: Tensor
    sha256: str
    description: Record
    dtype: Literal["float16", "float32"]
    rows: int
    threshold: float


@dataclass
class Totals:
    written: int = 0
    limit: int = 0
    maximum_relative_l2: float = 0.0
    arrays: int = 0


def specification(raw: Record) -> None:
    if set(raw) - OPTIONAL_FIELDS != SPECIFICATION_FIELDS or raw["schema"] != "fused-conversion-v1":
        raise ValueError("unsupported fused conversion specification")
    if raw["corpus_format"] not in {"corpus-v1", "corpus-mixture-v1"}:
        raise ValueError("fused conversion reads corpus-v1 or corpus-mixture-v1 inputs")
    if raw["frozen_format"] not in {"Q4_0", "baseline-k"}:
        raise ValueError("unsupported frozen table format")
    if (raw["frozen_format"] == "baseline-k") != ("frozen_baseline" in raw):
        raise ValueError("baseline-k requires an authenticated Baseline artifact")
    if raw["fused_dtype"] not in {"float16", "float32"}:
        raise ValueError("fused storage is float16 or float32")
    if raw["device"] not in {"cpu", "cuda"}:
        raise ValueError("supported devices are cpu/cuda")
    threshold = raw["maximum_relative_l2"]
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 < threshold < 1
    ):
        raise ValueError("explicit positive rounding threshold required")
    if not 1 <= integer(raw["rows"]) <= 8192:
        raise ValueError("row chunk must be between 1 and 8192")
    if integer(raw["capture_limit_bytes"]) < 1 or integer(raw["memory_bytes"]) < 1:
        raise ValueError("positive storage and memory bounds required")
    if integer(raw.get("disk_reserve_bytes", 8 * 1024**3)) < 0:
        raise ValueError("nonnegative disk reserve required")


def fusion_table(raw: Record, base: Path) -> Fusion:
    geometry = Geometry(
        **{key: integer(value) for key, value in record(raw.get("geometry", {})).items()}
    )
    frozen_format = cast(Literal["Q4_0", "baseline-k"], raw["frozen_format"])
    baseline = artifact(raw["frozen_baseline"], base) if "frozen_baseline" in raw else None
    initialization = artifact(raw["initialization"], base) if "initialization" in raw else None
    source = artifact(raw["source"], base)
    vocabulary = artifact(raw["vocabulary"], base)
    model = load_he4(
        source,
        vocabulary,
        integer(raw["memory_bytes"]),
        geometry,
        initialization,
        frozen_format,
        baseline,
    )
    operand = model.fusion.operand.detach().clone()
    del model
    sha256 = fusion_identity(operand)
    description: Record = {
        "sha256": sha256,
        "operand": "fusion.operand",
        "shape": list(operand.shape),
        "frozen_format": frozen_format,
        "frozen_baseline": None if baseline is None else reference(baseline),
        "source": reference(source),
        "vocabulary": reference(vocabulary),
        "initialization": None if initialization is None else reference(initialization),
    }
    return Fusion(
        operand.to(string(raw["device"])),
        sha256,
        description,
        cast(Literal["float16", "float32"], raw["fused_dtype"]),
        integer(raw["rows"]),
        number(raw["maximum_relative_l2"]),
    )


def tap_files(row: Record, root: Path, suffix: str) -> tuple[list[Artifact], list[Artifact]]:
    by_name: dict[str, Artifact] = {}
    for value in sequence(row["files"]):
        ref = artifact(value, root)
        if ref.path.name in by_name:
            raise ValueError("duplicate capture filenames")
        by_name[ref.path.name] = ref
    taps = []
    for tap in TAPS:
        name = f"layer-{tap}.{suffix}"
        if name not in by_name:
            raise ValueError("eight-tap source record lacks a feature file")
        taps.append(by_name.pop(name))
    if any(name.startswith("layer-") for name in by_name):
        raise ValueError("source record mixes feature storage formats")
    return taps, list(by_name.values())


def source_metadata(row: Record, root: Path) -> tuple[Artifact, Record]:
    ref = artifact(row["metadata"], root)
    ref.verify()
    metadata = read(ref.path)
    if metadata.get("taps") != list(TAPS) or metadata.get("feature_semantics") != TAP_SEMANTICS:
        raise ValueError("fused conversion requires eight-tap source captures")
    if metadata.get("feature_dtype", "float32") not in {"float16", "float32"}:
        raise ValueError("unsupported source feature dtype")
    return ref, metadata


def fuse_record(
    name: str,
    row: Record,
    root: Path,
    output: Store,
    fusion: Fusion,
    hidden: int,
    totals: Totals,
) -> tuple[Record, Record]:
    metadata_ref, metadata = source_metadata(row, root)
    source_half = metadata.get("feature_dtype", "float32") == "float16"
    taps, support = tap_files(row, root, "f16" if source_half else "f32")
    count = len(sequence(metadata["input_tokens"]))
    width = 2 if source_half else 4
    if any(ref.size != count * hidden * width for ref in taps):
        raise ValueError("eight-tap source geometry mismatch")
    out_half = fusion.dtype == "float16"
    out_width = 2 if out_half else 4
    totals.written += count * hidden * out_width
    if totals.written > totals.limit:
        raise ValueError("fused feature payload exceeds authorized bound")
    cell = Store(output.root / hashlib.sha256(name.encode()).hexdigest(), output.reserve_bytes)
    error_squared = 0.0
    norm_squared = 0.0
    maximum = 0.0
    device = fusion.operand.device

    def write(sink: BinaryIO) -> None:
        nonlocal error_squared, norm_squared, maximum
        digests = [hashlib.sha256() for _ in taps]
        handles = [ref.path.open("rb", buffering=0) for ref in taps]
        try:
            before = [os.fstat(handle.fileno()) for handle in handles]
            for start in range(0, count, fusion.rows):
                rows = min(fusion.rows, count - start)
                size = rows * hidden * width
                parts = []
                for handle, digest in zip(handles, digests, strict=True):
                    data = handle.read(size)
                    if len(data) != size:
                        raise ValueError("source feature file ended early")
                    digest.update(data)
                    parts.append(
                        torch.frombuffer(
                            bytearray(data), dtype=torch.float16 if source_half else torch.float32
                        ).view(rows, hidden)
                    )
                stacked = torch.stack(parts, dim=1).reshape(rows, -1)
                fused = torch.nn.functional.linear(stacked.to(device).float(), fusion.operand).cpu()
                stored = fused.half() if out_half else fused
                if not bool(torch.isfinite(fused).all()) or not bool(torch.isfinite(stored).all()):
                    raise ValueError("nonfinite fused value or F16 overflow; keep eight taps")
                delta = stored.double() - fused.double()
                error_squared += float(delta.square().sum())
                norm_squared += float(fused.double().square().sum())
                maximum = max(maximum, float(delta.abs().max()))
                encoded = cast("HalfArray | FloatArray", stored.numpy())
                sink.write(encoded.tobytes())
            for handle, previous in zip(handles, before, strict=True):
                if handle.read(1):
                    raise ValueError("source feature file exceeds its recorded size")
                current = os.fstat(handle.fileno())
                if (current.st_size, current.st_mtime_ns) != (
                    previous.st_size,
                    previous.st_mtime_ns,
                ):
                    raise ValueError("source feature file changed during conversion")
        finally:
            for handle in handles:
                handle.close()
        for digest, ref in zip(digests, taps, strict=True):
            if digest.hexdigest() != ref.sha256:
                raise ValueError("source feature content mismatch")

    stored_ref = cell.stream(
        "fused.f16" if out_half else "fused.f32", write, count * hidden * out_width
    )
    relative = math.sqrt(error_squared / norm_squared) if norm_squared else 0.0
    if relative > fusion.threshold:
        raise ValueError("fused rounding exceeds the frozen numerical margin")
    totals.maximum_relative_l2 = max(totals.maximum_relative_l2, relative)
    totals.arrays += 1
    files = [reference(stored_ref)]
    for ref in support:
        if ref.size > SUPPORT_LIMIT:
            raise ValueError("supporting file exceeds conversion copy bound")
        ref.verify()
        payload = ref.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != ref.sha256:
            raise ValueError("supporting file changed during conversion")
        cell.atomic(cell.root / ref.path.name, payload)
        files.append(
            reference(Artifact(cell.root / ref.path.name, ref.sha256, ref.size, cell.root))
        )
    fused_metadata = metadata | {
        "feature_semantics": FUSED_SEMANTICS,
        "feature_dtype": fusion.dtype,
        "feature_shape": [count, hidden],
        "fusion_sha256": fusion.sha256,
        "fused_source": {
            "metadata": reference(metadata_ref),
            "features": [reference(ref) for ref in taps],
            "feature_dtype": "float16" if source_half else "float32",
        },
    }
    cell.atomic(cell.root / "fused-metadata.json", canonical(fused_metadata))
    path = cell.root / "fused-metadata.json"
    converted = dict(row)
    converted["metadata"] = reference(Artifact.capture(path))
    converted["files"] = files
    measurement: Record = {
        "id": name,
        "tokens": count,
        "relative_l2": relative,
        "max_absolute": maximum,
        "converted": reference(stored_ref),
    }
    cell.atomic(cell.root / "fusion-measurement.json", canonical(measurement))
    return converted, measurement


def forecast(corpus: Record, root: Path, hidden: int, width: int) -> int:
    total = 0
    for value in sequence(corpus["records"]):
        row = record(value)
        if row["role"] != "train":
            continue
        _, metadata = source_metadata(row, root)
        total += len(sequence(metadata["input_tokens"])) * hidden * width
    return total


def fuse_corpus(
    source: Artifact,
    output: Store,
    fusion: Fusion,
    hidden: int,
    totals: Totals,
    measurements: list[Record],
) -> Artifact:
    source.verify()
    corpus = read(source.path, 64 * 1024 * 1024)
    if set(corpus) != {"schema", "target_sha256", "protected", "expected_train", "records"} or (
        corpus["schema"] != 1
    ):
        raise ValueError("fused conversion reads corpus-v1 manifests")
    rows: list[Record] = []
    for value in sequence(corpus["records"]):
        row = record(value)
        if row["role"] != "train":
            rows.append(row)
            continue
        converted, measurement = fuse_record(
            string(row["id"]), row, source.path.parent, output, fusion, hidden, totals
        )
        rows.append(converted)
        measurements.append(measurement)
    manifest: Record = {
        "schema": "fused-corpus-v1",
        "target_sha256": corpus["target_sha256"],
        "protected": corpus["protected"],
        "expected_train": corpus["expected_train"],
        "fusion": fusion.description,
        "source": reference(source),
        "records": rows,
    }
    output.atomic(output.root / "corpus.json", canonical(manifest))
    return Artifact.capture(output.root / "corpus.json")


def fuse(path: Path, output: Path) -> Record:
    raw = read(path)
    specification(raw)
    base = path.resolve().parent
    source = artifact(raw["corpus"], base)
    source.verify()
    if output.exists():
        raise ValueError("fused conversion writes a new root")
    geometry = Geometry(
        **{key: integer(value) for key, value in record(raw.get("geometry", {})).items()}
    )
    width = 2 if raw["fused_dtype"] == "float16" else 4
    components: list[Artifact] = []
    if raw["corpus_format"] == "corpus-mixture-v1":
        mixture = read(source.path)
        if (
            set(mixture) != {"schema", "target_sha256", "components"}
            or mixture["schema"] != "corpus-mixture-v1"
        ):
            raise ValueError("unsupported corpus mixture")
        for value in sequence(mixture["components"]):
            component = record(value)
            if set(component) != {"kind", "source"} or component["kind"] != "corpus-v1":
                raise ValueError("fused conversion reads corpus-v1 mixture components")
            ref = artifact(component["source"], source.path.parent)
            ref.verify()
            components.append(ref)
    else:
        components.append(source)
    expected = sum(
        forecast(read(ref.path, 64 * 1024 * 1024), ref.path.parent, geometry.hidden, width)
        for ref in components
    )
    reserve = integer(raw.get("disk_reserve_bytes", 8 * 1024**3))
    if expected > integer(raw["capture_limit_bytes"]):
        raise ValueError("fused feature forecast exceeds the declared bound")
    available = shutil.disk_usage(output.parent if output.parent.exists() else base).free
    if expected + reserve > available:
        raise ValueError("fused feature forecast exceeds free disk plus reserve")
    store = Store(output, reserve)
    code_root = Path(__file__).resolve().parents[2]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    torch.set_num_threads(2)
    device = string(raw["device"])
    with device_lease(device):
        fusion = fusion_table(raw, base)
        totals = Totals(limit=integer(raw["capture_limit_bytes"]))
        measurements: list[Record] = []
        produced: list[Artifact] = []
        for index, ref in enumerate(components):
            target = (
                store
                if raw["corpus_format"] == "corpus-v1"
                else Store(store.root / f"component-{index}", reserve)
            )
            produced.append(fuse_corpus(ref, target, fusion, geometry.hidden, totals, measurements))
    store.atomic(store.root / "measurements.json", canonical({"arrays": measurements}))
    result: Record = {
        "status": "converted",
        "source": reference(source),
        "corpus_format": "fused-corpus-v1"
        if raw["corpus_format"] == "corpus-v1"
        else "fused-mixture-v1",
        "fusion": fusion.description,
        "fused_dtype": fusion.dtype,
        "device": device,
        "rows": fusion.rows,
        "torch": torch.__version__,
        "feature_bytes": totals.written,
        "forecast_bytes": expected,
        "arrays": totals.arrays,
        "maximum_relative_l2": totals.maximum_relative_l2,
        "source_sha256": code,
        "roles_preserved": True,
        "source_deleted": False,
    }
    if raw["corpus_format"] == "corpus-mixture-v1":
        mixture_raw = read(source.path)
        manifest: Record = {
            "schema": "fused-mixture-v1",
            "target_sha256": mixture_raw["target_sha256"],
            "fusion": fusion.description,
            "source": reference(source),
            "components": [
                {"kind": "fused-corpus-v1", "source": reference(ref)} for ref in produced
            ],
        }
        store.atomic(store.root / "mixture.json", canonical(manifest))
        result["corpus"] = reference(Artifact.capture(store.root / "mixture.json"))
    else:
        result["corpus"] = reference(produced[0])
    store.atomic(store.root / "result.json", canonical(result))
    return result


class Arguments(argparse.Namespace):
    specification: Path
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args(namespace=Arguments())
    args.specification = relocated(args.specification).resolve()
    args.output = output_path(args.output, "datasets", args.specification.stem)
    try:
        print(json.dumps(fuse(args.specification, args.output), indent=2))
        return 0
    except (ValueError, OSError, KeyError) as error:
        refusal: Record = {"status": "refused", "error": str(error)}
        print(json.dumps(refusal))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

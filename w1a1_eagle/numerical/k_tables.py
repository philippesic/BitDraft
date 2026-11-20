import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeAlias, cast

import gguf
import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4
from w1a1_eagle.numerical.weights import GGUF
from w1a1_eagle.records import artifact, read

BLOCKS = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q6_K": 210}
Bytes: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]"
Floats: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
PARTS = {"ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}


def quantize(
    quantizer: Path,
    source_kind: str,
    target_kind: str,
    rows: int,
    columns: int,
    source: Path,
    output: Path,
    rows_per_call: int | None = None,
) -> Bytes:
    completed = subprocess.run(
        [
            str(quantizer),
            source_kind,
            target_kind,
            str(rows),
            str(columns),
            str(source),
            str(output),
            *([] if rows_per_call is None else [str(rows_per_call)]),
        ],
        capture_output=True,
        check=False,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise ValueError("table quantizer failed: " + completed.stderr.decode(errors="replace"))
    data = cast(Bytes, np.fromfile(output, dtype=np.uint8))
    if data.size != rows * columns // 256 * BLOCKS[target_kind]:
        raise ValueError("table quantizer output size differs")
    return cast(Bytes, data.reshape(rows, columns // 256 * BLOCKS[target_kind]))


def quantizer_binary(build: Artifact) -> Artifact:
    build.verify()
    binary = artifact(read(build.path)["binary"], build.path.parent)
    binary.verify()
    return binary


def parameter_name(tensor: str) -> str:
    _, index, part, _ = tensor.split(".")
    return f"blocks.{index}.{PARTS[part]}.operand"


def k_codes(quantizer: Path, value: Tensor, kind: str, scratch: Path, name: str) -> Bytes:
    if (
        kind not in ("Q4_K", "Q6_K")
        or value.dtype != torch.float32
        or value.ndim != 2
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("finite F32 table and a Q4_K or Q6_K type required: " + name)
    rows, columns = (int(extent) for extent in value.shape)
    source = scratch / (name + ".f32")
    cast(Floats, value.detach().cpu().contiguous().numpy()).tofile(source)
    try:
        return quantize(quantizer, "F32", kind, rows, columns, source, scratch / (name + ".bin"))
    finally:
        source.unlink()
        (scratch / (name + ".bin")).unlink(missing_ok=True)


def plain_codes(model: HE4, quantizer: Path, workers: int = 6) -> dict[str, tuple[str, Bytes]]:
    if not model.plain_feed_forward or len(model.feed_forward_types) != 3 * model.geometry.layers:
        raise ValueError("plain feed-forward codes need the plain model and its Baseline types")
    with tempfile.TemporaryDirectory(prefix="plain-tables-") as directory:
        scratch = Path(directory)

        def encode(entry: tuple[str, str]) -> tuple[str, tuple[str, Bytes]]:
            name, kind = entry
            operand = model.get_parameter(parameter_name(name))
            return name, (kind, k_codes(quantizer, operand, kind, scratch, name))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(encode, model.feed_forward_types))


def dequantized_codes(codes: Bytes, kind: str, shape: tuple[int, ...]) -> Tensor:
    api = cast(GGUF, gguf)
    kinds = cast(object, api.GGMLQuantizationType)
    native = cast(object, getattr(kinds, kind))
    values = api.dequantize(codes, native).astype("float32").reshape(shape)
    return torch.from_numpy(cast(Floats, values).copy())


def plain_tables(model: HE4, quantizer: Path) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, (kind, codes) in plain_codes(model, quantizer).items():
        operand = model.get_parameter(parameter_name(name))
        result[parameter_name(name)] = dequantized_codes(codes, kind, tuple(operand.shape)).to(
            operand.device
        )
    return result

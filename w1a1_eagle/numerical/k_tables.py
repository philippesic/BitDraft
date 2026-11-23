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
from w1a1_eagle.model import HE4, Table, q4_k_parts, q6_k_parts
from w1a1_eagle.numerical.weights import GGUF
from w1a1_eagle.records import artifact, read

BLOCKS = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q6_K": 210}
Bytes: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]"
Floats: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
PARTS = {
    "ffn_gate": "gate",
    "ffn_up": "up",
    "ffn_down": "down",
    "attn_q": "query",
    "attn_k": "key",
    "attn_v": "value",
    "attn_output": "output",
}


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


def table_codes(
    model: HE4, types: tuple[tuple[str, str], ...], quantizer: Path, workers: int = 6
) -> dict[str, tuple[str, Bytes]]:
    with tempfile.TemporaryDirectory(prefix="trained-tables-") as directory:
        scratch = Path(directory)

        def encode(entry: tuple[str, str]) -> tuple[str, tuple[str, Bytes]]:
            name, kind = entry
            operand = model.get_parameter(parameter_name(name))
            return name, (kind, k_codes(quantizer, operand, kind, scratch, name))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(encode, types))


def plain_codes(model: HE4, quantizer: Path) -> dict[str, tuple[str, Bytes]]:
    if not model.plain_feed_forward or len(model.feed_forward_types) != 3 * model.geometry.layers:
        raise ValueError("plain feed-forward codes need the plain model and its Baseline types")
    return table_codes(model, model.feed_forward_types, quantizer)


def attention_codes(model: HE4, quantizer: Path) -> dict[str, tuple[str, Bytes]]:
    if not model.trained_attention or len(model.attention_types) != 4 * model.geometry.layers:
        raise ValueError("attention codes need trained attention tables and their Baseline types")
    return table_codes(model, model.attention_types, quantizer)


def trained_rows(model: HE4, quantizer: Path) -> Bytes:
    if not model.head_rows:
        raise ValueError("the model has no trained vocabulary rows")
    with tempfile.TemporaryDirectory(prefix="trained-rows-") as directory:
        return k_codes(quantizer, model.head_values, "Q4_K", Path(directory), "rows")


def exported_rows(model: HE4, quantizer: Path) -> dict[str, Tensor]:
    restored = dequantized_codes(
        trained_rows(model, quantizer), "Q4_K", tuple(model.head_values.shape)
    )
    return {"head_values": restored.to(model.head_values.device)}


def half_bytes(value: Tensor) -> Tensor:
    return value.reshape(-1, 1).half().cpu().contiguous().view(torch.uint8)


def as_bytes(packed: Tensor, rows: int) -> Bytes:
    return cast(Bytes, packed.contiguous().numpy()).reshape(rows, -1)


def q4_k_codes(value: Tensor) -> Bytes:
    d, dmin, scale_codes, min_codes, levels = q4_k_parts(value.detach().float())
    blocks = levels.shape[0]
    ls = scale_codes.reshape(blocks, 8).cpu().to(torch.uint8)
    lm = min_codes.reshape(blocks, 8).cpu().to(torch.uint8)
    scales = torch.cat(
        [
            ls[:, 0:4] | ((ls[:, 4:8] >> 4) << 6),
            lm[:, 0:4] | ((lm[:, 4:8] >> 4) << 6),
            (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4),
        ],
        1,
    )
    quants = levels.reshape(blocks, 4, 2, 32).cpu().to(torch.uint8)
    qs = (quants[:, :, 0, :] | (quants[:, :, 1, :] << 4)).reshape(blocks, 128)
    packed = torch.cat([half_bytes(d), half_bytes(dmin), scales, qs], 1)
    return as_bytes(packed, int(value.shape[0]))


def q6_k_codes(value: Tensor) -> Bytes:
    d, codes, levels, tiny = q6_k_parts(value.detach().float())
    blocks = levels.shape[0]
    stored = (levels + 32).reshape(blocks, 2, 4, 32).cpu().to(torch.uint8)
    low = stored & 0xF
    high = stored >> 4
    ql = torch.cat(
        [low[:, :, 0] | (low[:, :, 2] << 4), low[:, :, 1] | (low[:, :, 3] << 4)], 2
    ).reshape(blocks, 128)
    qh = (
        high[:, :, 0] | (high[:, :, 1] << 2) | (high[:, :, 2] << 4) | (high[:, :, 3] << 6)
    ).reshape(blocks, 64)
    scales = codes.reshape(blocks, 16).cpu().to(torch.int8).view(torch.uint8)
    packed = torch.cat([ql, qh, scales, half_bytes(d)], 1)
    packed[tiny.reshape(blocks).cpu()] = 0
    return as_bytes(packed, int(value.shape[0]))


def rounded_codes(model: HE4) -> dict[str, tuple[str, Bytes]]:
    if not model.trained_attention or len(model.attention_types) != 4 * model.geometry.layers:
        raise ValueError("rounded codes need trained attention tables and their Baseline types")
    result: dict[str, tuple[str, Bytes]] = {}
    for name, kind in model.attention_types:
        operand = model.get_parameter(parameter_name(name))
        result[name] = (kind, q4_k_codes(operand) if kind == "Q4_K" else q6_k_codes(operand))
    return result


def dequantized_codes(codes: Bytes, kind: str, shape: tuple[int, ...]) -> Tensor:
    api = cast(GGUF, gguf)
    kinds = cast(object, api.GGMLQuantizationType)
    native = cast(object, getattr(kinds, kind))
    values = api.dequantize(codes, native).astype("float32").reshape(shape)
    return torch.from_numpy(cast(Floats, values).copy())


def rounded_attention(model: HE4) -> bool:
    return any(
        cast(Table, getattr(module, part)).rounding is not None
        for module in model.blocks
        for part in ("query", "key", "value", "output")
    )


def plain_tables(model: HE4, quantizer: Path) -> dict[str, Tensor]:
    coded = (
        rounded_codes(model)
        if rounded_attention(model)
        else attention_codes(model, quantizer)
        if model.trained_attention
        else plain_codes(model, quantizer)
    )
    result: dict[str, Tensor] = {}
    for name, (kind, codes) in coded.items():
        operand = model.get_parameter(parameter_name(name))
        result[parameter_name(name)] = dequantized_codes(codes, kind, tuple(operand.shape)).to(
            operand.device
        )
    return result

import math
from pathlib import Path
from typing import Protocol, cast

import gguf
import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Geometry, Layer, Table
from w1a1_eagle.numerical.weights import GGUF, TensorRecord, shapes

TABLE_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"}
FEED_FORWARD = ("gate", "up", "down")
CHUNK_ROWS = 8192
MODULES = {
    "attn_q": "query",
    "attn_k": "key",
    "attn_v": "value",
    "attn_output": "output",
    "ffn_gate": "gate",
    "ffn_up": "up",
    "ffn_down": "down",
}


class Rows(Protocol):
    def __getitem__(self, key: "slice[int, int, None]") -> object: ...


def dequantized(item: TensorRecord, shape: tuple[int, ...], device: str = "cpu") -> Tensor:
    api = cast(GGUF, gguf)
    data = cast(Rows, item.data)
    result = torch.empty(shape, dtype=torch.float32, device=device)
    rows = shape[0] if len(shape) == 2 else 1
    for start in range(0, rows, CHUNK_ROWS):
        part = data[start : start + CHUNK_ROWS] if len(shape) == 2 else data
        decoded = api.dequantize(part, item.tensor_type).astype("float32")
        block = torch.from_numpy(cast("np.ndarray[tuple[int, ...], np.dtype[np.float32]]", decoded))
        if not bool(torch.isfinite(block).all()):
            raise ValueError("nonfinite Baseline tensor: " + item.name)
        if len(shape) == 2:
            result[start : start + block.shape[0]].copy_(block.reshape(-1, shape[1]))
        else:
            result.copy_(block.reshape(shape))
    return result


def table_tensor(path: Path, names: tuple[str, ...]) -> tuple[str, int]:
    api = cast(GGUF, gguf)
    lookup = {item.name: item for item in api.GGUFReader(str(path)).tensors}
    for name in names:
        if name in lookup:
            return name, int(lookup[name].shape[-1])
    raise ValueError("vocabulary table unavailable: " + str(path))


def table_rows(path: Path, names: tuple[str, ...]) -> int:
    return table_tensor(path, names)[1]


def module_name(tensor: str) -> str:
    if tensor == "fc.weight":
        return "fusion"
    _, index, part, _ = tensor.split(".")
    return f"blocks.{index}.{MODULES[part]}"


def load_baseline(
    drafter: Artifact,
    target: Artifact,
    memory_budget_bytes: int,
    geometry: Geometry = Geometry(),
    device: str = "cpu",
) -> HE4:
    api = cast(GGUF, gguf)
    reader = api.GGUFReader(str(drafter.path))
    expected = shapes(geometry)
    records = {item.name: item for item in reader.tensors}
    if len(records) != len(reader.tensors) or set(records) != set(expected):
        raise ValueError("Baseline tensor coverage differs from the HE4 graph")
    for name, shape in expected.items():
        item = records[name]
        if tuple(int(v) for v in reversed(item.shape)) != shape:
            raise ValueError("Baseline tensor shape mismatch: " + name)
        allowed = {"F32"} if len(shape) == 1 else TABLE_TYPES
        if item.tensor_type.name not in allowed:
            raise ValueError("unsupported Baseline tensor type: " + name)
    if reader.fields["dflash.target_layers"].contents() != [2, 6, 10, 14, 18, 22, 26, 30]:
        raise ValueError("target tap semantics mismatch")
    mask = reader.fields["tokenizer.ggml.mask_token_id"].contents()
    if type(mask) is not int:
        raise ValueError("invalid mask identity")
    target_reader = api.GGUFReader(str(target.path))
    lookup = {item.name: item for item in target_reader.tensors}
    if "output.weight" in lookup:
        raise ValueError(
            "Baseline evaluation supports targets whose output table is the token table"
        )
    table = lookup.get("token_embd.weight")
    if table is None or table.tensor_type.name not in TABLE_TYPES:
        raise ValueError("target token table unavailable")
    table_shape = tuple(int(v) for v in reversed(table.shape))
    if len(table_shape) != 2 or table_shape[1] != geometry.hidden or not 0 <= mask < table_shape[0]:
        raise ValueError("target token table geometry mismatch")
    resident = 4 * (sum(math.prod(shape) for shape in expected.values()) + math.prod(table_shape))
    estimated = CHUNK_ROWS * table_shape[1] * 4 * 8 + (resident if device == "cpu" else 0)
    if type(memory_budget_bytes) is not int or memory_budget_bytes < estimated:
        raise ValueError("Baseline materialization exceeds memory budget")
    drafter.verify()
    target.verify()
    placeholder = torch.zeros((1, 1))
    tensors = {
        name: dequantized(records[name], shape) if len(shape) == 1 else placeholder
        for name, shape in expected.items()
    }
    model = HE4(tensors, torch.zeros((1, geometry.hidden)), 0, geometry)
    model.mask = mask
    for module in model.blocks:
        for name in FEED_FORWARD:
            setattr(cast(Layer, module), name, Table(placeholder))
    model.requires_grad_(False)
    model = model.to(device)
    for name, shape in expected.items():
        if len(shape) == 2:
            operand = dequantized(records[name], shape, device)
            cast(Table, model.get_submodule(module_name(name))).operand = operand
            del operand
    vocabulary = dequantized(table, table_shape, device)
    model.vocabulary = vocabulary
    del vocabulary
    model.origin = (drafter.sha256, target.sha256)
    model.frozen_format = "baseline-evaluation"
    return model.eval()

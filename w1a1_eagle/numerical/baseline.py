import math
from typing import cast

import gguf
import torch
from torch import Tensor

from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Geometry, Layer, Table
from w1a1_eagle.numerical.weights import GGUF, TensorRecord, shapes

TABLE_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"}
FEED_FORWARD = ("gate", "up", "down")


def dequantized(item: TensorRecord, shape: tuple[int, ...]) -> Tensor:
    api = cast(GGUF, gguf)
    value = torch.tensor(
        api.dequantize(item.data, item.tensor_type).reshape(shape).astype("float32"),
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite Baseline tensor: " + item.name)
    return value


def load_baseline(
    drafter: Artifact,
    target: Artifact,
    memory_budget_bytes: int,
    geometry: Geometry = Geometry(),
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
    estimated = 12 * (sum(math.prod(shape) for shape in expected.values()) + math.prod(table_shape))
    if type(memory_budget_bytes) is not int or memory_budget_bytes < estimated:
        raise ValueError("Baseline materialization exceeds memory budget")
    drafter.verify()
    target.verify()
    tensors = {name: dequantized(records[name], shape) for name, shape in expected.items()}
    model = HE4(tensors, dequantized(table, table_shape), mask, geometry)
    for index, module in enumerate(model.blocks):
        block = cast(Layer, module)
        for name in FEED_FORWARD:
            setattr(block, name, Table(tensors[f"blk.{index}.ffn_{name}.weight"]))
    model.origin = (drafter.sha256, target.sha256)
    model.frozen_format = "baseline-evaluation"
    model.requires_grad_(False)
    return model.eval()

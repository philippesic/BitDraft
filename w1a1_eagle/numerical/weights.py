import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, Sequence, cast

import gguf
import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Geometry
from w1a1_eagle.records import canonical, reference


class Array(Protocol):
    def reshape(self, shape: tuple[int, ...]) -> "Array": ...
    def astype(self, dtype: str) -> "Array": ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def tobytes(self) -> bytes: ...


class Kind(Protocol):
    @property
    def name(self) -> str: ...


class TensorRecord(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> Sequence[int]: ...
    @property
    def data(self) -> object: ...
    @property
    def tensor_type(self) -> Kind: ...


class Field(Protocol):
    @property
    def types(self) -> Sequence[object]: ...
    def contents(self) -> object: ...


class Reader(Protocol):
    @property
    def tensors(self) -> list[TensorRecord]: ...
    @property
    def fields(self) -> dict[str, Field]: ...


class Types(Protocol):
    @property
    def Q4_0(self) -> object: ...


class GGUF(Protocol):
    @property
    def GGMLQuantizationType(self) -> Types: ...
    def GGUFReader(self, path: str) -> Reader: ...
    def dequantize(self, data: object, kind: object) -> Array: ...
    def quantize(self, data: object, kind: object) -> Array: ...


def shapes(geometry: Geometry) -> dict[str, tuple[int, ...]]:
    cfg = geometry
    result = {
        "fc.weight": (cfg.hidden, cfg.hidden * cfg.taps),
        "enc.output_norm.weight": (cfg.hidden,),
        "output_norm.weight": (cfg.hidden,),
    }
    layer_shapes = {
        "attn_norm": (cfg.hidden,),
        "attn_q_norm": (cfg.head_dim,),
        "attn_k_norm": (cfg.head_dim,),
        "ffn_norm": (cfg.hidden,),
        "attn_q": (cfg.heads * cfg.head_dim, cfg.hidden),
        "attn_k": (cfg.kv_heads * cfg.head_dim, cfg.hidden),
        "attn_v": (cfg.kv_heads * cfg.head_dim, cfg.hidden),
        "attn_output": (cfg.hidden, cfg.heads * cfg.head_dim),
        "ffn_gate": (cfg.intermediate, cfg.hidden),
        "ffn_up": (cfg.intermediate, cfg.hidden),
        "ffn_down": (cfg.hidden, cfg.intermediate),
    }
    for index in range(cfg.layers):
        result.update({f"blk.{index}.{name}.weight": shape for name, shape in layer_shapes.items()})
    return result


def decode(
    source: Artifact,
    vocabulary: Artifact,
    memory_budget_bytes: int,
    geometry: Geometry = Geometry(),
) -> tuple[dict[str, Tensor], Tensor, int, tuple[tuple[str, tuple[int, ...], bytes], ...]]:
    api = cast(GGUF, gguf)
    reader = api.GGUFReader(str(source.path))
    expected = shapes(geometry)
    records = {t.name: t for t in reader.tensors}
    if len(records) != len(reader.tensors) or set(records) != set(expected):
        raise ValueError("source tensor coverage mismatch")
    for name, shape in expected.items():
        if tuple(int(v) for v in reversed(records[name].shape)) != shape:
            raise ValueError("source tensor shape mismatch: " + name)
        if (
            ".ffn_" in name
            and not name.endswith("norm.weight")
            and records[name].tensor_type.name != "BF16"
        ):
            raise ValueError("FFN initialization requires original BF16 tensors")
    if reader.fields["dflash.target_layers"].contents() != [2, 6, 10, 14, 18, 22, 26, 30]:
        raise ValueError("target tap semantics mismatch")
    mask = reader.fields["tokenizer.ggml.mask_token_id"].contents()
    if type(mask) is not int:
        raise ValueError("invalid mask identity")
    vocabulary_reader = api.GGUFReader(str(vocabulary.path))
    lookup = {t.name: t for t in vocabulary_reader.tensors}
    selected = lookup.get("output.weight", lookup.get("token_embd.weight"))
    if selected is None or selected.tensor_type.name != "Q4_K":
        raise ValueError("explicit Q4_K vocabulary basis required")
    vocabulary_shape = tuple(int(v) for v in reversed(selected.shape))
    if (
        len(vocabulary_shape) != 2
        or vocabulary_shape[1] != geometry.hidden
        or not 0 <= mask < vocabulary_shape[0]
    ):
        raise ValueError("vocabulary geometry mismatch")
    estimated = 12 * (
        sum(math.prod(shape) for shape in expected.values()) + math.prod(vocabulary_shape)
    )
    if type(memory_budget_bytes) is not int or memory_budget_bytes < estimated:
        raise ValueError("model materialization exceeds memory budget")
    source.verify()
    vocabulary.verify()
    tensors: dict[str, Tensor] = {}
    codes: list[tuple[str, tuple[int, ...], bytes]] = []
    for name, record in records.items():
        decoded = (
            api.dequantize(record.data, record.tensor_type)
            .reshape(expected[name])
            .astype("float32")
        )
        if (
            name.endswith(
                tuple(f".{part}.weight" for part in ("attn_q", "attn_k", "attn_v", "attn_output"))
            )
            or name == "fc.weight"
        ):
            coded = api.quantize(decoded, api.GGMLQuantizationType.Q4_0)
            codes.append((name, coded.shape, coded.tobytes()))
            decoded = api.dequantize(coded, api.GGMLQuantizationType.Q4_0).reshape(expected[name])
        tensors[name] = torch.tensor(decoded, dtype=torch.float32)
        if not bool(torch.isfinite(tensors[name]).all()):
            raise ValueError("nonfinite source tensor")
    head = torch.tensor(
        api.dequantize(selected.data, selected.tensor_type).reshape(vocabulary_shape),
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(head).all()):
        raise ValueError("nonfinite vocabulary")
    coded_head = cast(Array, selected.data)
    codes.append(("output.weight", coded_head.shape, coded_head.tobytes()))
    return tensors, head, mask, tuple(codes)


def load_he4(
    source: Artifact,
    vocabulary: Artifact,
    memory_budget_bytes: int,
    geometry: Geometry = Geometry(),
    initialization: Artifact | None = None,
) -> HE4:
    if initialization is None:
        tensors, head, mask, codes = decode(source, vocabulary, memory_budget_bytes, geometry)
    else:
        initialization.verify()
        source.verify()
        vocabulary.verify()
        with zipfile.ZipFile(initialization.path) as archive:
            if sum(item.file_size for item in archive.infolist()) * 2 > memory_budget_bytes:
                raise ValueError("materialized initialization exceeds memory budget")
        raw = record(
            cast(
                object,
                torch.load(initialization.path, map_location="cpu", weights_only=True, mmap=True),
            )
        )
        if (
            raw.get("schema") != "he4-initialization-v1"
            or raw.get("origin") != [source.sha256, vocabulary.sha256]
            or raw.get("geometry") != cast(Record, asdict(geometry))
        ):
            raise ValueError("materialized initialization origin/geometry differs")
        tensors = {}
        expected = shapes(geometry)
        saved = record(raw["tensors"])
        if set(saved) != set(expected):
            raise ValueError("materialized tensor coverage differs")
        for name, value in saved.items():
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.float32
                or tuple(value.shape) != expected[name]
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("invalid materialized source tensor")
            tensors[name] = value
        saved_head = raw["head"]
        if (
            not torch.is_tensor(saved_head)
            or saved_head.dtype != torch.float32
            or not bool(torch.isfinite(saved_head).all())
        ):
            raise ValueError("invalid materialized vocabulary")
        head = saved_head
        mask = integer(raw["mask"])
        entries = []
        for value in sequence(raw["codes"]):
            row = record(value)
            data = row["data"]
            if not isinstance(data, bytes):
                raise ValueError("native initialization codes must be immutable bytes")
            entries.append(
                (string(row["name"]), tuple(integer(v) for v in sequence(row["shape"])), data)
            )
        codes = tuple(entries)
    model = HE4(tensors, head, mask, geometry)
    model.origin = (source.sha256, vocabulary.sha256)
    model.native_codes = codes
    return model


def prepare_initialization(
    source: Artifact, vocabulary: Artifact, memory_bytes: int, geometry: Geometry, output: Path
) -> Record:
    if output.exists():
        raise ValueError("initialization output must be new")
    tensors, head, mask, codes = decode(source, vocabulary, memory_bytes, geometry)
    payload: dict[str, object] = {
        "schema": "he4-initialization-v1",
        "origin": [source.sha256, vocabulary.sha256],
        "geometry": cast(Record, asdict(geometry)),
        "tensors": tensors,
        "head": head,
        "mask": mask,
        "codes": [
            {"name": name, "shape": list(shape), "data": data} for name, shape, data in codes
        ],
    }
    store = Store(output)
    materialized = store.stream(
        "initialization.bin", lambda sink: torch.save(payload, sink), memory_bytes
    )
    result: Record = {
        "schema": "he4-initialization-v1",
        "source": reference(source),
        "vocabulary": reference(vocabulary),
        "materialized": reference(materialized),
        "geometry": cast(Record, asdict(geometry)),
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

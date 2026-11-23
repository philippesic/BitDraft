import hashlib
import math
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol, Sequence, cast

import gguf
import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Geometry, Table
from w1a1_eagle.numerical.captures import FloatArray
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


TABLE_LAYERS = (
    ("query", "attn_q"),
    ("key", "attn_k"),
    ("value", "attn_v"),
    ("output", "attn_output"),
)


def is_table(name: str) -> bool:
    return name == "fc.weight" or name.endswith(
        tuple(f".{part}.weight" for _, part in TABLE_LAYERS)
    )


def table_names(geometry: Geometry) -> dict[str, str]:
    return {
        "fc.weight": "fusion.operand",
        **{
            f"blk.{index}.{part}.weight": f"blocks.{index}.{module}.operand"
            for index in range(geometry.layers)
            for module, part in TABLE_LAYERS
        },
    }


def frozen_fingerprint(tensors: dict[str, Tensor], head: Tensor, geometry: Geometry) -> str:
    digest = hashlib.sha256()
    operands = [("vocabulary", head)] + [
        (module, tensors[name]) for name, module in table_names(geometry).items()
    ]
    for name, value in operands:
        operand = value.detach().float().cpu().contiguous()
        digest.update(canonical([name, str(operand.dtype), list(operand.shape)]))
        digest.update(cast(Array, operand.numpy()).tobytes())
    return digest.hexdigest()


def fusion_identity(operand: Tensor) -> str:
    value = operand.detach().to("cpu", torch.float32).contiguous()
    digest = hashlib.sha256(canonical(["fusion.operand", "float32", list(value.shape)]))
    data = cast(FloatArray, value.numpy()).data.cast("B")
    for offset in range(0, len(data), 16 * 1024 * 1024):
        digest.update(data[offset : offset + 16 * 1024 * 1024])
    return digest.hexdigest()


def full_precision_tables(source: Artifact, geometry: Geometry) -> dict[str, Tensor]:
    api = cast(GGUF, gguf)
    expected = shapes(geometry)
    result: dict[str, Tensor] = {}
    for item in api.GGUFReader(str(source.path)).tensors:
        if is_table(item.name):
            decoded = (
                api.dequantize(item.data, item.tensor_type)
                .reshape(expected[item.name])
                .astype("float32")
            )
            result[item.name] = torch.tensor(decoded, dtype=torch.float32)
            if not bool(torch.isfinite(result[item.name]).all()):
                raise ValueError("nonfinite source table")
    if set(result) != set(table_names(geometry)):
        raise ValueError("source table coverage mismatch")
    return result


def q4_0_codes(value: Tensor) -> Array:
    if value.dtype != torch.float32 or value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("finite F32 table required for Q4_0 conversion")
    api = cast(GGUF, gguf)
    operand = cast(Array, value.detach().cpu().contiguous().numpy())
    return api.quantize(operand, api.GGMLQuantizationType.Q4_0)


def exported_tables(model: HE4) -> dict[str, Tensor]:
    api = cast(GGUF, gguf)
    result: dict[str, Tensor] = {}
    for name in table_names(model.geometry).values():
        operand = model.get_parameter(name)
        restored = api.dequantize(q4_0_codes(operand), api.GGMLQuantizationType.Q4_0)
        result[name] = torch.tensor(
            restored.reshape(tuple(operand.shape)), dtype=torch.float32, device=operand.device
        )
    return result


def decode(
    source: Artifact,
    vocabulary: Artifact,
    memory_budget_bytes: int,
    geometry: Geometry = Geometry(),
    vocabulary_kinds: frozenset[str] = frozenset({"Q4_K"}),
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
    if selected is None or selected.tensor_type.name not in vocabulary_kinds:
        raise ValueError("explicit " + "/".join(sorted(vocabulary_kinds)) + " vocabulary required")
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
        if is_table(name):
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
    frozen_format: Literal["Q4_0", "baseline-k"] = "Q4_0",
    frozen_baseline: Artifact | None = None,
    trained_tables: bool = False,
    plain_feed_forward: bool = False,
    trained_attention: bool = False,
    head_rows: tuple[int, ...] = (),
    attention_rounding: bool = False,
) -> HE4:
    if trained_attention and (frozen_baseline is None or trained_tables or plain_feed_forward):
        raise ValueError(
            "trained attention tables start from Baseline's tables with one-bit feed-forward"
        )
    if plain_feed_forward and (
        frozen_baseline is None or trained_tables or initialization is not None
    ):
        raise ValueError(
            "plain feed-forward tables need frozen_format baseline-k, frozen attention tables"
            " and no materialized initialization"
        )
    if frozen_format not in {"Q4_0", "baseline-k"}:
        raise ValueError("unsupported frozen table format")
    if (frozen_format == "baseline-k") != (frozen_baseline is not None):
        raise ValueError("baseline-k requires an authenticated Baseline artifact")
    if trained_tables and frozen_format != "Q4_0":
        raise ValueError(
            "trained tables cannot use frozen_format baseline-k: they are not copied from"
            " Baseline and export as Q4_0"
        )
    if initialization is None:
        tensors, head, mask, codes = decode(
            source,
            vocabulary,
            memory_budget_bytes,
            geometry,
            frozenset({"Q4_K", "Q6_K"}) if plain_feed_forward else frozenset({"Q4_K"}),
        )
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
    native_types: tuple[tuple[str, str], ...] = ()
    attention_types: tuple[tuple[str, str], ...] = ()
    feed_forward_types: list[tuple[str, str]] = []
    parent_frozen_sha256 = None
    if trained_tables:
        parent_frozen_sha256 = frozen_fingerprint(tensors, head, geometry)
        tensors = {**tensors, **full_precision_tables(source, geometry)}
        codes = tuple(entry for entry in codes if entry[0] == "output.weight")
    if frozen_baseline is not None:
        parent_frozen_sha256 = frozen_fingerprint(tensors, head, geometry)
        frozen_baseline.verify()
        api = cast(GGUF, gguf)
        baseline = api.GGUFReader(str(frozen_baseline.path))
        records = {item.name: item for item in baseline.tensors}
        if len(records) != len(baseline.tensors):
            raise ValueError("duplicate Baseline tensors")
        replacements = []
        types = []
        for name, shape, data in codes:
            if name == "output.weight":
                replacements.append((name, shape, data))
                continue
            item = records.get(name)
            if item is None or item.tensor_type.name not in {"Q4_K", "Q6_K"}:
                raise ValueError("Baseline frozen table must be Q4_K or Q6_K: " + name)
            if tuple(int(v) for v in reversed(item.shape)) != tuple(tensors[name].shape):
                raise ValueError("Baseline frozen table geometry differs: " + name)
            decoded = api.dequantize(item.data, item.tensor_type).reshape(
                tuple(tensors[name].shape)
            )
            operand = torch.tensor(decoded, dtype=torch.float32)
            if not bool(torch.isfinite(operand).all()):
                raise ValueError("nonfinite Baseline frozen operand")
            tensors[name] = operand
            coded = cast(Array, item.data)
            replacements.append((name, coded.shape, coded.tobytes()))
            types.append((name, item.tensor_type.name))
        if trained_attention:
            attention_types = tuple(entry for entry in types if ".attn_" in entry[0])
            replacements = [entry for entry in replacements if ".attn_" not in entry[0]]
            types = [entry for entry in types if ".attn_" not in entry[0]]
        codes = tuple(replacements)
        native_types = tuple(types)
        if plain_feed_forward:
            for name in sorted(tensors):
                if ".ffn_" not in name or name.endswith("norm.weight"):
                    continue
                item = records.get(name)
                if item is None or item.tensor_type.name not in {"Q4_K", "Q6_K"}:
                    raise ValueError("Baseline feed-forward table must be Q4_K or Q6_K: " + name)
                if tuple(int(v) for v in reversed(item.shape)) != tuple(tensors[name].shape):
                    raise ValueError("Baseline feed-forward geometry differs: " + name)
                feed_forward_types.append((name, item.tensor_type.name))
    model = HE4(
        tensors,
        head,
        mask,
        geometry,
        trained_tables,
        plain_feed_forward,
        trained_attention,
        head_rows,
    )
    if plain_feed_forward:
        model.feed_forward_types = tuple(feed_forward_types)
    model.attention_types = attention_types
    if attention_rounding:
        if not trained_attention:
            raise ValueError("attention rounding applies to trained attention tables")
        kinds = dict(attention_types)
        for index, module in enumerate(model.blocks):
            for part, name in (
                ("query", "attn_q"),
                ("key", "attn_k"),
                ("value", "attn_v"),
                ("output", "attn_output"),
            ):
                table = cast(Table, getattr(module, part))
                table.rounding = kinds[f"blk.{index}.{name}.weight"]
    model.parent_frozen_sha256 = parent_frozen_sha256
    model.frozen_format = frozen_format
    model.frozen_baseline = None if frozen_baseline is None else frozen_baseline.sha256
    model.native_types = native_types
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

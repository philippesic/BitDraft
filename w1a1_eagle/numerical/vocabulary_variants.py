import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, Sequence, TypeAlias, cast

import gguf
import numpy as np
import torch

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.identity import source_identity
from w1a1_eagle.numerical.export import ExportAPI, PayloadBytes, write_gguf
from w1a1_eagle.numerical.gguf_readback import Expected
from w1a1_eagle.numerical.gguf_readback import verify as verify_gguf
from w1a1_eagle.numerical.labels import label_frequencies
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.storage import project_root

SHORTLIST_KEY = "dflash.vocab_shortlist"
SHORTLIST_RULE = (
    "the N most frequent target tokens at distinct supervised positions of a TRAIN mixture, "
    "ties by token id, stored in ascending token order; the runtime adds every token of the "
    "request so far"
)
TIED_KEY = "dflash.component.tied"
LAYOUTS = ("shared", "scoring")
BLOCKS = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q6_K": 210}
PACKED = {"Q4_0": (32, 18), **{name: (256, width) for name, width in BLOCKS.items()}}
VARIANT_TYPES = ("Q2_K", "Q3_K")
ROW_CHUNK = 16384

Bytes: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]"
Floats: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"


class TensorView(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def data(self) -> object: ...
    @property
    def tensor_type(self) -> object: ...
    @property
    def shape(self) -> Sequence[int]: ...


class FieldView(Protocol):
    @property
    def types(self) -> Sequence[object]: ...
    def contents(self) -> object: ...


class ReaderView(Protocol):
    @property
    def tensors(self) -> list[TensorView]: ...
    @property
    def fields(self) -> dict[str, FieldView]: ...


class TypeName(Protocol):
    @property
    def name(self) -> str: ...


class TableKinds(Protocol):
    @property
    def Q2_K(self) -> object: ...
    @property
    def Q3_K(self) -> object: ...
    @property
    def Q4_K(self) -> object: ...
    @property
    def Q6_K(self) -> object: ...


def table_kind(name: str) -> object:
    kinds = cast(TableKinds, gguf_api().GGMLQuantizationType)
    return {"Q2_K": kinds.Q2_K, "Q3_K": kinds.Q3_K, "Q4_K": kinds.Q4_K, "Q6_K": kinds.Q6_K}[name]


@dataclass(frozen=True)
class Shortlist:
    tokens: tuple[int, ...]
    size: int | None
    mixture: str
    labels: int
    distinct: int
    coverage: float | None

    def payload(self) -> bytes:
        return b"".join(token.to_bytes(4, "little", signed=True) for token in self.tokens)

    def record(self) -> Record:
        return {
            "schema": "vocabulary-shortlist-v1",
            "rule": SHORTLIST_RULE,
            "size": self.size,
            "rows": len(self.tokens),
            "train_mixture_sha256": self.mixture,
            "train_labels": self.labels,
            "distinct_train_labels": self.distinct,
            "label_coverage": self.coverage,
            "tokens_sha256": hashlib.sha256(self.payload()).hexdigest(),
        }


@dataclass(frozen=True)
class Variant:
    kind: str
    layout: str

    @property
    def name(self) -> str:
        return "table-" + self.kind.lower().replace("_", "") + "-" + self.layout


@dataclass(frozen=True)
class TableSettings:
    source: Artifact
    tensor: str
    quantizer: Artifact
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class Settings:
    export: Artifact
    mixture: Artifact | None
    size: int | None
    tables: TableSettings | None


def shortlist(mixture: Artifact, size: int | None) -> Shortlist:
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("shortlist size must be a positive integer or null for every TRAIN label")
    frequencies = label_frequencies(mixture)
    chosen = frequencies.ranked if size is None else frequencies.ranked[:size]
    if not chosen:
        raise ValueError("the TRAIN mixture has no labels for a shortlist")
    return Shortlist(
        tuple(sorted(chosen)),
        size,
        frequencies.mixture,
        frequencies.labels,
        len(frequencies.ranked),
        frequencies.coverage(len(chosen)),
    )


def settings(path: Path) -> Settings:
    raw = read(path)
    root = path.resolve().parent
    if set(raw) - {"shortlist", "tables"} != {"schema", "export"} or raw["schema"] != (
        "vocabulary-variants-v1"
    ):
        raise ValueError("unsupported vocabulary variants specification")
    if "shortlist" not in raw and "tables" not in raw:
        raise ValueError("vocabulary variants need a shortlist or table variants")
    mixture: Artifact | None = None
    size: int | None = None
    if "shortlist" in raw:
        section = record(raw["shortlist"])
        if set(section) != {"mixture", "size"}:
            raise ValueError("a shortlist names its TRAIN mixture and size")
        mixture = artifact(section["mixture"], root)
        size = None if section["size"] is None else integer(section["size"])
    tables: TableSettings | None = None
    if "tables" in raw:
        section = record(raw["tables"])
        if set(section) != {"source", "tensor", "quantizer", "variants"}:
            raise ValueError("table variants need source, tensor, quantizer and variants")
        variants = []
        for value in sequence(section["variants"]):
            row = record(value)
            if set(row) != {"type", "layout"}:
                raise ValueError("a table variant names its type and layout")
            variant = Variant(string(row["type"]), string(row["layout"]))
            if variant.kind not in VARIANT_TYPES or variant.layout not in LAYOUTS:
                raise ValueError("table variants support Q2_K or Q3_K, shared or scoring")
            variants.append(variant)
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("table variants must be distinct and at least one")
        tables = TableSettings(
            artifact(section["source"], root),
            string(section["tensor"]),
            artifact(section["quantizer"], root),
            tuple(variants),
        )
    return Settings(artifact(raw["export"], root), mixture, size, tables)


def gguf_api() -> ExportAPI:
    return cast(ExportAPI, gguf)


def reader(path: Path) -> ReaderView:
    return cast(ReaderView, gguf_api().GGUFReader(str(path)))


def kind_name(tensor: TensorView) -> str:
    return cast(TypeName, tensor.tensor_type).name


def raw_bytes(tensor: TensorView) -> Bytes:
    return cast(Bytes, np.ascontiguousarray(cast(Bytes, tensor.data)).view(np.uint8))


def element_shape(tensor: TensorView) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def table_tensor(source: ReaderView, name: str) -> TensorView:
    found = [tensor for tensor in source.tensors if tensor.name == name]
    if len(found) != 1:
        raise ValueError("table tensor missing or repeated: " + name)
    return found[0]


def table_rows(tensor: TensorView) -> tuple[int, int]:
    shape = element_shape(tensor)
    if len(shape) != 2 or shape[0] % 256 or kind_name(tensor) not in BLOCKS:
        raise ValueError("a vocabulary table is a two-dimensional K-quantized matrix")
    return shape[1], shape[0]


def decoded(data: Bytes, kind: str, columns: int) -> Floats:
    values = gguf_api().dequantize(data, table_kind(kind))
    return cast(Floats, values.astype("float32").reshape((-1, columns)))


def quantize(
    quantizer: Path,
    source_kind: str,
    target_kind: str,
    rows: int,
    columns: int,
    source: Path,
    output: Path,
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


def relative_error(
    reference_rows: Bytes, reference_kind: str, rows: Bytes, kind: str, columns: int
) -> float:
    squared = 0.0
    total = 0.0
    for start in range(0, int(rows.shape[0]), ROW_CHUNK):
        expected = torch.from_numpy(
            decoded(reference_rows[start : start + ROW_CHUNK], reference_kind, columns)
        ).double()
        actual = torch.from_numpy(decoded(rows[start : start + ROW_CHUNK], kind, columns)).double()
        squared += float((actual - expected).square().sum())
        total += float(expected.square().sum())
    if total <= 0:
        raise ValueError("source table has no energy")
    return float(np.sqrt(squared / total))


@dataclass(frozen=True)
class Output:
    name: str
    data: object
    kind: object
    kind_name: str
    payload: bytes


def write_variant(
    store: Store,
    filename: str,
    source: ReaderView,
    config: str,
    added: list[tuple[str, object]],
    replaced: dict[str, tuple[Bytes, str]],
    inserted: dict[str, tuple[str, Bytes, str]],
    budget: int,
) -> Artifact:
    writer = gguf_api().GGUFWriter(None, "dflash")
    overridden = {key for key, _ in added}
    for key, field in source.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key in overridden:
            continue
        if not field.types:
            raise ValueError("source metadata has no declared type")
        writer.add_key_value(
            key,
            field.contents(),
            field.types[0],
            field.types[-1] if len(field.types) > 1 else None,
        )
    for key, value in added:
        if isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, list):
            writer.add_array(key, cast(list[object], value))
        else:
            raise ValueError("unsupported variant metadata value")
    outputs: list[Output] = []
    for tensor in source.tensors:
        if tensor.name in inserted:
            name, data, kind = inserted[tensor.name]
            outputs.append(Output(name, data, table_kind(kind), kind, data.tobytes()))
        if tensor.name in replaced:
            data, kind = replaced[tensor.name]
            outputs.append(Output(tensor.name, data, table_kind(kind), kind, data.tobytes()))
            continue
        payload = raw_bytes(tensor).tobytes()
        outputs.append(
            Output(tensor.name, tensor.data, tensor.tensor_type, kind_name(tensor), payload)
        )
    expected: Expected = {}
    for item in outputs:
        array = cast(Bytes, item.data)
        shape = tuple(int(value) for value in array.shape)
        if item.kind_name in PACKED:
            block, width = PACKED[item.kind_name]
            shape = (*shape[:-1], shape[-1] // width * block)
        expected[item.name] = (
            item.kind_name,
            tuple(reversed(shape)),
            len(item.payload),
            hashlib.sha256(item.payload).hexdigest(),
        )
        writer.add_tensor(item.name, cast(Bytes, item.data), raw_dtype=item.kind)

    def write(output: BinaryIO) -> None:
        write_gguf(
            writer,
            output,
            [cast(PayloadBytes, np.frombuffer(item.payload, dtype=np.uint8)) for item in outputs],
        )

    produced = store.stream(filename, write, budget)
    verify_gguf(produced.path, expected, config)
    return produced


def write_variants(specification: Path, output: Path) -> Record:
    chosen = settings(specification)
    if output.exists():
        raise ValueError("vocabulary variant output must be new")
    store = Store(output)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    chosen.export.verify()
    source = reader(chosen.export.path)
    fields = {key: field for key, field in source.fields.items()}
    config_field = fields.get("dflash.w1ax.config_hash")
    if config_field is None or SHORTLIST_KEY in fields:
        raise ValueError("the export must be an HE4 export without a shortlist")
    config = string(config_field.contents())
    head = table_tensor(source, "output.weight")
    if kind_name(head) != "Q4_K" or string_contents(fields, TIED_KEY) is not True:
        raise ValueError("the export must carry one tied Q4_K vocabulary table")
    rows, columns = table_rows(head)
    budget = chosen.export.size + 2 * rows * columns
    result: Record = {
        "schema": "vocabulary-variants-v1",
        "export": reference(chosen.export),
        "config_hash": config,
        "code_sha256": source_identity(project_root()),
        "outputs": {},
    }
    outputs = cast(dict[str, object], result["outputs"])
    if chosen.mixture is not None:
        selected = shortlist(chosen.mixture, chosen.size)
        if max(selected.tokens) >= rows:
            raise ValueError("shortlist tokens exceed the vocabulary")
        store.atomic(store.root / "shortlist.i32", selected.payload())
        description = selected.record()
        description["train_mixture"] = reference(chosen.mixture)
        store.atomic(store.root / "shortlist.json", canonical(description))
        drafter = write_variant(
            store,
            "drafter-shortlist.gguf",
            source,
            config,
            [
                (SHORTLIST_KEY, list(selected.tokens)),
                (SHORTLIST_KEY + ".size", selected.size or 0),
                (SHORTLIST_KEY + ".mixture", selected.mixture),
                (SHORTLIST_KEY + ".rule", SHORTLIST_RULE),
            ],
            {},
            {},
            budget,
        )
        outputs["shortlist"] = {
            "sidecar": reference(Artifact.capture(store.root / "shortlist.i32")),
            "record": reference(Artifact.capture(store.root / "shortlist.json")),
            "drafter": reference(drafter),
            "rows": len(selected.tokens),
            "distinct_train_labels": selected.distinct,
            "label_coverage": selected.coverage,
        }
    if chosen.tables is not None:
        outputs["tables"] = table_variants(
            chosen.tables, store, source, config, head, rows, columns, budget
        )
    store.atomic(store.root / "result.json", canonical(result))
    return result


def string_contents(fields: dict[str, FieldView], key: str) -> object:
    field = fields.get(key)
    return None if field is None else field.contents()


def table_variants(
    tables: TableSettings,
    store: Store,
    source: ReaderView,
    config: str,
    head: TensorView,
    rows: int,
    columns: int,
    budget: int,
) -> Record:
    tables.source.verify()
    tables.quantizer.verify()
    built = read(tables.quantizer.path)
    binary = artifact(built["binary"], tables.quantizer.path.parent)
    binary.verify()
    original = table_tensor(reader(tables.source.path), tables.tensor)
    source_kind = kind_name(original)
    if table_rows(original) != (rows, columns):
        raise ValueError("source table geometry differs from the export's vocabulary")
    scratch = Store(store.root / "scratch")
    source_path = scratch.root / "source.bin"
    source_rows = cast(Bytes, raw_bytes(original).reshape(rows, -1))
    source_rows.tofile(source_path)
    head_rows = cast(Bytes, raw_bytes(head).reshape(rows, -1))
    try:
        reproduced = quantize(
            binary.path,
            source_kind,
            "Q4_K",
            rows,
            columns,
            source_path,
            scratch.root / "Q4_K.bin",
        )
        if reproduced.tobytes() != head_rows.tobytes():
            raise ValueError(
                "the quantizer does not reproduce the export's Q4_K table from this source"
            )
        report: Record = {
            "source": reference(tables.source),
            "tensor": tables.tensor,
            "source_type": source_kind,
            "quantizer": reference(tables.quantizer),
            "quantizer_binary": reference(binary),
            "importance_matrix": None,
            "reproduced_export_table": True,
            "relative_error": {
                "Q4_K": relative_error(source_rows, source_kind, head_rows, "Q4_K", columns)
            },
            "variants": {},
        }
        errors = cast(dict[str, object], report["relative_error"])
        variants = cast(dict[str, object], report["variants"])
        for variant in tables.variants:
            coded = quantize(
                binary.path,
                source_kind,
                variant.kind,
                rows,
                columns,
                source_path,
                scratch.root / (variant.kind + ".bin"),
            )
            if variant.kind not in errors:
                errors[variant.kind] = relative_error(
                    source_rows, source_kind, coded, variant.kind, columns
                )
            provenance: list[tuple[str, object]] = [
                ("dflash.vocabulary.type", variant.kind),
                ("dflash.vocabulary.layout", variant.layout),
                (
                    "dflash.vocabulary.source",
                    tables.tensor + " " + tables.source.sha256,
                ),
                ("dflash.vocabulary.quantizer", binary.sha256),
            ]
            if variant.layout == "shared":
                produced = write_variant(
                    store,
                    variant.name + ".gguf",
                    source,
                    config,
                    provenance,
                    {"output.weight": (coded, variant.kind)},
                    {},
                    budget,
                )
            else:
                produced = write_variant(
                    store,
                    variant.name + ".gguf",
                    source,
                    config,
                    [(TIED_KEY, False), *provenance],
                    {"output.weight": (coded, variant.kind)},
                    {"output.weight": ("token_embd.weight", head_rows, "Q4_K")},
                    budget,
                )
            variants[variant.name] = {
                "type": variant.kind,
                "layout": variant.layout,
                "drafter": reference(produced),
                "scoring_table_bytes": int(coded.size),
            }
        return report
    finally:
        shutil.rmtree(scratch.root, ignore_errors=True)

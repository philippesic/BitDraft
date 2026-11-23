import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, Sequence, TypeAlias, cast

import gguf
import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Layer, Projection
from w1a1_eagle.numerical.gguf_readback import Expected, metadata
from w1a1_eagle.numerical.gguf_readback import verify as verify_gguf
from w1a1_eagle.numerical.k_tables import (
    attention_codes,
    plain_codes,
    quantizer_binary,
    rounded_attention,
    rounded_codes,
    trained_rows,
)
from w1a1_eagle.numerical.weights import GGUF, q4_0_codes
from w1a1_eagle.numerical.worker import TensorWorker

ANCHOR_CLIP_KEY = ".w1ax.clip_anchor"
TRAINED_ROWS_KEY = "dflash.vocab_trained_rows"
NativeArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8 | np.int32 | np.float32]]"
PayloadBytes: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]"


class Writer(Protocol):
    fout: list[BinaryIO] | None
    state: object

    def add_key_value(
        self, key: str, value: object, kind: object, subtype: object | None = None
    ) -> None: ...
    def add_uint32(self, key: str, value: int) -> None: ...
    def add_float32(self, key: str, value: float) -> None: ...
    def add_string(self, key: str, value: str) -> None: ...
    def add_bool(self, key: str, value: bool) -> None: ...
    def add_array(self, key: str, value: Sequence[object]) -> None: ...
    def add_tensor(self, key: str, value: NativeArray, raw_dtype: object) -> None: ...
    def write_header_to_file(self) -> None: ...
    def write_kv_data_to_file(self) -> None: ...
    def write_ti_data_to_file(self) -> None: ...
    def write_padding(self, output: BinaryIO, size: int) -> None: ...


class ExportTypes(Protocol):
    @property
    def F32(self) -> object: ...
    @property
    def I32(self) -> object: ...
    @property
    def Q4_0(self) -> object: ...
    @property
    def Q4_K(self) -> object: ...
    @property
    def Q6_K(self) -> object: ...


class WriterStates(Protocol):
    @property
    def EMPTY(self) -> object: ...


class ExportAPI(GGUF, Protocol):
    @property
    def GGMLQuantizationType(self) -> ExportTypes: ...
    @property
    def WriterState(self) -> WriterStates: ...
    def GGUFWriter(self, path: None, architecture: str) -> Writer: ...
    def GGUFValueType(self, value: int) -> object: ...


@dataclass(frozen=True)
class ExportTensor:
    name: str
    data: NativeArray
    kind: object
    kind_name: str


def write_gguf(writer: Writer, output: BinaryIO, payloads: Sequence[PayloadBytes]) -> None:
    writer.fout = [output]
    writer.state = cast(ExportAPI, gguf).WriterState.EMPTY
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_padding(output, output.tell())
    for view in payloads:
        for offset in range(0, view.nbytes, 1024 * 1024):
            output.write(view[offset : offset + 1024 * 1024].tobytes())
        writer.write_padding(output, view.nbytes)


def f32(value: Tensor) -> NativeArray:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite export operand")
    return cast(NativeArray, value.detach().float().cpu().contiguous().numpy())


def packed_signs(value: Tensor) -> NativeArray:
    if value.dtype != torch.float32 or value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("finite F32 sign matrix required")
    signs = value.detach().cpu() >= 0
    signs = torch.nn.functional.pad(signs, (0, (-value.shape[1]) % 32))
    array = cast("np.ndarray[tuple[int, ...], np.dtype[np.bool_]]", signs.numpy())
    packed = cast(NativeArray, np.packbits(array, axis=-1, bitorder="little"))
    return cast(NativeArray, packed.view(np.int32))


def prepare(
    worker: TensorWorker, source: Artifact, vocabulary: Artifact, quantizer: Path | None = None
) -> tuple[Writer, list[ExportTensor]]:
    worker.verify_frozen()
    worker.verify_finite()
    model = worker.model
    worker.progress.validate()
    if (
        worker.progress.needs_recovery
        or worker.updates != worker.progress.updates
        or worker.cursor != worker.progress.cursor
    ):
        raise ValueError("export requires consistent complete budget accounting")
    if not worker.ready or model.origin != (source.sha256, vocabulary.sha256):
        raise ValueError(
            "export requires a complete update and matching admitted source/vocabulary"
        )
    source.verify()
    vocabulary.verify()
    api = cast(ExportAPI, gguf)
    kinds = api.GGMLQuantizationType
    reader = api.GGUFReader(str(source.path)) if not model.native_codes else None
    writer = api.GGUFWriter(None, "dflash")
    if reader is None:
        for key, value, kind, subtype in metadata(source.path):
            if (
                not key.startswith(("GGUF.", "dflash.w1ax.", "dflash.component."))
                and key != "general.architecture"
            ):
                writer.add_key_value(
                    key,
                    value,
                    api.GGUFValueType(kind),
                    api.GGUFValueType(subtype) if subtype is not None else None,
                )
    else:
        for key, field in reader.fields.items():
            if (
                not key.startswith(("GGUF.", "dflash.w1ax.", "dflash.component."))
                and key != "general.architecture"
            ):
                if not field.types:
                    raise ValueError("source metadata has no declared type")
                writer.add_key_value(
                    key,
                    field.contents(),
                    field.types[0],
                    field.types[-1] if len(field.types) > 1 else None,
                )
    if model.frozen_format != "Q4_0":
        writer.add_string("dflash.w1ax.frozen_format", model.frozen_format)
        if model.frozen_baseline is None:
            raise ValueError("Baseline frozen tables lack authenticated identity")
        writer.add_string("dflash.w1ax.frozen_baseline", model.frozen_baseline)
    writer.add_uint32("dflash.w1ax.version", 2)
    writer.add_uint32("dflash.w1ax.activation_bits", 8)
    for key, config_value in {
        "profile": "ffn" + str(model.geometry.layers * 3),
        "bit_order": "little",
        "sign_rule": "nonnegative_is_one",
        "scale_rule": "f32_learned_nonnegative",
        "arithmetic": "affine_same_quantized_stream_v1",
        "a1": "signed_pm_bound_threshold0_nonnegative_positive_zero_row_scale0",
        "execution": "packed_signs_int8_codes_dp4a",
        "config_hash": worker.identity,
    }.items():
        writer.add_string("dflash.w1ax." + key, config_value)
    writer.add_bool("dflash.w1ax.private_head", True)
    writer.add_bool("dflash.component.tied", True)
    selected = [
        f"blk.{index}.ffn_{name}.weight"
        for index in range(model.geometry.layers)
        for name in ("gate", "up", "down")
    ]
    writer.add_array("dflash.w1ax.tensors", sorted(selected))
    tensors: list[ExportTensor] = []
    tables = {"fc.weight": model.fusion.operand}

    def add(name: str, value: Tensor) -> None:
        tensors.append(ExportTensor(name, f32(value), kinds.F32, "F32"))

    add("enc.output_norm.weight", model.encoder_norm)
    add("output_norm.weight", model.output_norm)
    for index, module in enumerate(model.blocks):
        block = cast(Layer, module)
        for name, operand in (
            ("attn_q", block.query.operand),
            ("attn_k", block.key.operand),
            ("attn_v", block.value.operand),
            ("attn_output", block.output.operand),
        ):
            tables[f"blk.{index}.{name}.weight"] = operand
        for name, value in (
            ("attn_norm", block.attn_norm),
            ("attn_q_norm", block.q_norm),
            ("attn_k_norm", block.k_norm),
            ("ffn_norm", block.ffn_norm),
        ):
            add(f"blk.{index}.{name}.weight", value)
        for name, projection in (("gate", block.gate), ("up", block.up), ("down", block.down)):
            if not isinstance(projection, Projection):
                raise ValueError("W1Ax export needs one-bit feed-forward projections")
            base = f"blk.{index}.ffn_{name}"
            bound = projection.log_clip.detach().float().cpu().exp()
            if not bool(torch.isfinite(bound)) or float(bound.detach()) <= 0:
                raise ValueError("invalid export clipping bound")
            writer.add_uint32(base + ".w1ax.bits", projection.bits)
            writer.add_uint32(base + ".w1ax.mode", 2)
            writer.add_float32(base + ".w1ax.clip", float(bound.detach()))
            if projection.log_anchor_clip is not None:
                anchor = projection.log_anchor_clip.detach().float().cpu().exp()
                if not bool(torch.isfinite(anchor)) or float(anchor) <= 0:
                    raise ValueError("invalid export anchor clipping bound")
                writer.add_float32(base + ANCHOR_CLIP_KEY, float(anchor))
            tensors.append(
                ExportTensor(
                    base + ".w1a1_packed", packed_signs(projection.latent_sign), kinds.I32, "I32"
                )
            )
            add(base + ".w1a1_scale", projection.scale.clamp_min(0))
            add(base + ".w1ax_midpoint", projection.midpoint)
            if name != "up":
                add(base + ".w1ax_gain", projection.log_gain.detach().float().cpu().exp())
    trained = {name: value for name, value in tables.items() if value.requires_grad}
    expected = (
        len(tables)
        if model.trained_tables
        else 4 * model.geometry.layers
        if model.trained_attention
        else 0
    )
    if len(trained) != expected:
        raise ValueError("trained table coverage differs from the model setting")
    frozen = {name: value for name, value in tables.items() if name not in trained}
    if model.trained_attention:
        if quantizer is None:
            raise ValueError("trained attention tables export through the table quantizer")
        codes = (
            rounded_codes(model) if rounded_attention(model) else attention_codes(model, quantizer)
        )
        if set(codes) != set(trained):
            raise ValueError("trained attention code coverage differs")
        for name, (kind_name, payload) in sorted(codes.items()):
            native_kind = {"Q4_K": kinds.Q4_K, "Q6_K": kinds.Q6_K}[kind_name]
            tensors.append(ExportTensor(name, cast(NativeArray, payload), native_kind, kind_name))
    else:
        for name, value in trained.items():
            coded = cast(NativeArray, q4_0_codes(value))
            tensors.append(ExportTensor(name, coded, kinds.Q4_0, "Q4_0"))
    if model.native_codes:
        cached = {name: (shape, data) for name, shape, data in model.native_codes}
        if len(cached) != len(model.native_codes) or set(cached) != {*frozen, "output.weight"}:
            raise ValueError("native frozen code coverage differs")
        for name, (shape, data) in cached.items():
            array = cast(NativeArray, np.frombuffer(data, dtype=np.uint8).reshape(shape))
            head_code = name == "output.weight"
            kind_name = "Q4_K" if head_code else dict(model.native_types).get(name, "Q4_0")
            native_kind = {"Q4_0": kinds.Q4_0, "Q4_K": kinds.Q4_K, "Q6_K": kinds.Q6_K}[kind_name]
            operand = model.vocabulary if head_code else frozen[name]
            restored = torch.tensor(
                api.dequantize(array, native_kind).reshape(tuple(operand.shape)),
                dtype=torch.float32,
            )
            if not torch.equal(restored, operand.detach().cpu()):
                raise ValueError("frozen code execution/export operands disagree")
            tensors.append(
                ExportTensor(
                    name,
                    array,
                    native_kind,
                    kind_name,
                )
            )
    else:
        if reader is None:
            raise ValueError("source reader unavailable")
        for tensor in reader.tensors:
            if tensor.name in selected or tensor.name in trained or "norm.weight" in tensor.name:
                continue
            decoded = (
                api.dequantize(tensor.data, tensor.tensor_type)
                .reshape(tuple(int(v) for v in reversed(tensor.shape)))
                .astype("float32")
            )
            coded = cast(NativeArray, api.quantize(decoded, kinds.Q4_0))
            restored = torch.tensor(
                api.dequantize(coded, kinds.Q4_0).reshape(
                    tuple(int(v) for v in reversed(tensor.shape))
                ),
                dtype=torch.float32,
            )
            if tensor.name not in frozen or not torch.equal(
                restored, frozen[tensor.name].detach().cpu()
            ):
                raise ValueError("frozen source execution/export operands disagree")
            tensors.append(ExportTensor(tensor.name, coded, kinds.Q4_0, "Q4_0"))
        head_reader = api.GGUFReader(str(vocabulary.path))
        lookup = {tensor.name: tensor for tensor in head_reader.tensors}
        head = lookup.get("output.weight", lookup.get("token_embd.weight"))
        if head is None or head.tensor_type.name != "Q4_K":
            raise ValueError("explicit original Q4_K head required")
        decoded_head = torch.tensor(
            api.dequantize(head.data, head.tensor_type).reshape(
                tuple(int(v) for v in reversed(head.shape))
            ),
            dtype=torch.float32,
        )
        if not torch.equal(decoded_head, model.vocabulary.detach().cpu()):
            raise ValueError("frozen vocabulary execution/export operands disagree")
        tensors.append(
            ExportTensor("output.weight", cast(NativeArray, head.data), kinds.Q4_K, "Q4_K")
        )
    if model.head_rows:
        if quantizer is None:
            raise ValueError("trained vocabulary rows export through the table quantizer")
        found = [i for i, tensor in enumerate(tensors) if tensor.name == "output.weight"]
        if len(found) != 1 or tensors[found[0]].kind_name != "Q4_K":
            raise ValueError("trained vocabulary rows need the Q4_K output table")
        table = tensors[found[0]]
        coded = np.array(table.data, dtype=np.uint8, copy=True)
        rows = trained_rows(model, quantizer)
        if rows.shape[1] != coded.shape[1]:
            raise ValueError("trained vocabulary row width differs from the output table")
        coded[list(model.head_rows)] = rows
        tensors[found[0]] = ExportTensor(table.name, coded, table.kind, "Q4_K")
        writer.add_array(TRAINED_ROWS_KEY, list(model.head_rows))
    if len({tensor.name for tensor in tensors}) != len(tensors):
        raise ValueError("duplicate export tensor")
    return writer, tensors


def export_he4(
    worker: TensorWorker,
    source: Artifact,
    vocabulary: Artifact,
    store: Store,
    max_bytes: int,
    scratch_bytes: int,
    quantizer: Path | None = None,
) -> StateRef:
    with worker.lock:
        if (
            type(max_bytes) is not int
            or max_bytes < 1
            or type(scratch_bytes) is not int
            or scratch_bytes < 1
        ):
            raise ValueError("positive export byte and scratch budgets required")
        parameters = list(worker.model.parameters())
        operands = parameters + list(worker.model.buffers())
        estimate = (
            max(value.numel() for value in parameters) * 32
            + sum(value.numel() for value in operands) * 2
            + worker.model.vocabulary.numel() * 9
        )
        if estimate > scratch_bytes:
            raise ValueError("export working memory exceeds scratch budget")
        writer, tensors = prepare(worker, source, vocabulary, quantizer)
        return publish(worker, writer, tensors, store, max_bytes, scratch_bytes, worker.identity)


def publish(
    worker: TensorWorker,
    writer: Writer,
    tensors: list[ExportTensor],
    store: Store,
    max_bytes: int,
    scratch_bytes: int,
    recipe: str | None,
) -> StateRef:
    with worker.lock:
        required = sum(tensor.data.nbytes for tensor in tensors) + 2 * max(
            tensor.data.nbytes for tensor in tensors
        )
        if type(scratch_bytes) is not int or required > scratch_bytes:
            raise ValueError("export scratch budget exceeded")
        expected: Expected = {}
        for tensor in tensors:
            shape = tuple(tensor.data.shape)
            if tensor.kind_name in {"Q4_0", "Q4_K", "Q6_K"}:
                block, width = {"Q4_0": (32, 18), "Q4_K": (256, 144), "Q6_K": (256, 210)}[
                    tensor.kind_name
                ]
                shape = (*shape[:-1], shape[-1] // width * block)
            expected[tensor.name] = (
                tensor.kind_name,
                tuple(reversed(shape)),
                tensor.data.nbytes,
                hashlib.sha256(tensor.data.tobytes()).hexdigest(),
            )
        for tensor in tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.kind)

        def write(output: BinaryIO) -> None:
            write_gguf(
                writer,
                output,
                [cast(PayloadBytes, tensor.data.view(np.uint8).reshape(-1)) for tensor in tensors],
            )

        def verify(artifact: Artifact) -> None:
            verify_gguf(artifact.path, expected, recipe)

        return store.commit_with(write, worker.identity, worker.updates, max_bytes, verify)


def plain_norms(model: HE4) -> dict[str, Tensor]:
    norms: dict[str, Tensor] = {
        "enc.output_norm.weight": model.encoder_norm,
        "output_norm.weight": model.output_norm,
    }
    for index, module in enumerate(model.blocks):
        block = cast(Layer, module)
        for name, value in (
            ("attn_norm", block.attn_norm),
            ("attn_q_norm", block.q_norm),
            ("attn_k_norm", block.k_norm),
            ("ffn_norm", block.ffn_norm),
        ):
            norms[f"blk.{index}.{name}.weight"] = value
    return norms


def export_plain(
    worker: TensorWorker,
    baseline: Artifact,
    quantizer: Artifact,
    store: Store,
    max_bytes: int,
    scratch_bytes: int,
) -> StateRef:
    with worker.lock:
        model = worker.model
        if type(max_bytes) is not int or max_bytes < 1 or type(scratch_bytes) is not int:
            raise ValueError("positive export byte and scratch budgets required")
        worker.verify_frozen()
        worker.verify_finite()
        worker.progress.validate()
        if (
            not model.plain_feed_forward
            or model.frozen_baseline != baseline.sha256
            or worker.progress.needs_recovery
            or worker.updates != worker.progress.updates
            or worker.cursor != worker.progress.cursor
            or not worker.ready
        ):
            raise ValueError(
                "plain export requires a complete plain update and its authenticated Baseline"
            )
        baseline.verify()
        binary = quantizer_binary(quantizer)
        api = cast(ExportAPI, gguf)
        reader = api.GGUFReader(str(baseline.path))
        writer = api.GGUFWriter(None, "dflash")
        for key, value, kind, subtype in metadata(baseline.path):
            if not key.startswith("GGUF.") and key != "general.architecture":
                writer.add_key_value(
                    key,
                    value,
                    api.GGUFValueType(kind),
                    api.GGUFValueType(subtype) if subtype is not None else None,
                )
        codes = plain_codes(model, binary.path)
        norms = plain_norms(model)
        frozen = {name: data for name, _, data in model.native_codes if name != "output.weight"}
        kinds = api.GGMLQuantizationType
        native = {"F32": kinds.F32, "Q4_K": kinds.Q4_K, "Q6_K": kinds.Q6_K}
        tensors: list[ExportTensor] = []
        for item in reader.tensors:
            kind_name = item.tensor_type.name
            raw = cast(NativeArray, np.ascontiguousarray(cast(NativeArray, item.data)))
            if item.name in codes:
                coded_kind, coded = codes.pop(item.name)
                if coded_kind != kind_name or coded.shape != raw.view(np.uint8).shape:
                    raise ValueError("plain table type or geometry differs: " + item.name)
                tensors.append(ExportTensor(item.name, coded, native[kind_name], kind_name))
            elif item.name in norms:
                if kind_name != "F32":
                    raise ValueError("Baseline normalization weight is not F32: " + item.name)
                value = norms.pop(item.name)
                tensors.append(ExportTensor(item.name, f32(value), kinds.F32, "F32"))
            elif item.name in frozen:
                if raw.view(np.uint8).tobytes() != frozen.pop(item.name):
                    raise ValueError("frozen Baseline table differs from execution: " + item.name)
                tensors.append(ExportTensor(item.name, raw, native[kind_name], kind_name))
            else:
                raise ValueError("unexpected Baseline tensor: " + item.name)
        if codes or norms or frozen:
            raise ValueError("plain export coverage differs from the Baseline file")
        return publish(worker, writer, tensors, store, max_bytes, scratch_bytes, None)

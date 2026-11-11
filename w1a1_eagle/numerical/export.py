import hashlib
from dataclasses import dataclass
from typing import BinaryIO, Protocol, Sequence, TypeAlias, cast

import gguf
import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import Layer
from w1a1_eagle.numerical.weights import GGUF
from w1a1_eagle.numerical.worker import TensorWorker

NativeArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.uint8 | np.int32 | np.float32]]"


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


class WriterStates(Protocol):
    @property
    def EMPTY(self) -> object: ...


class ExportAPI(GGUF, Protocol):
    @property
    def GGMLQuantizationType(self) -> ExportTypes: ...
    @property
    def WriterState(self) -> WriterStates: ...
    def GGUFWriter(self, path: None, architecture: str) -> Writer: ...


@dataclass(frozen=True)
class ExportTensor:
    name: str
    data: NativeArray
    kind: object
    kind_name: str


def f32(value: Tensor) -> NativeArray:
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite export operand")
    return cast(NativeArray, value.detach().float().cpu().contiguous().numpy())


def packed_signs(value: Tensor) -> NativeArray:
    if value.dtype != torch.float32 or value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("finite F32 sign matrix required")
    signs = (value.detach().cpu() >= 0).to(torch.int64)
    signs = torch.nn.functional.pad(signs, (0, (-value.shape[1]) % 32))
    powers = torch.bitwise_left_shift(torch.ones(32, dtype=torch.int64), torch.arange(32))
    result = (signs.reshape(value.shape[0], -1, 32) * powers).sum(-1).to(torch.int32)
    return cast(NativeArray, result.contiguous().numpy())


def prepare(
    worker: TensorWorker, source: Artifact, vocabulary: Artifact
) -> tuple[Writer, list[ExportTensor]]:
    worker.verify_frozen()
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
    reader = api.GGUFReader(str(source.path))
    writer = api.GGUFWriter(None, "dflash")
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
    frozen = {"fc.weight": model.fusion.operand}

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
            frozen[f"blk.{index}.{name}.weight"] = operand
        for name, value in (
            ("attn_norm", block.attn_norm),
            ("attn_q_norm", block.q_norm),
            ("attn_k_norm", block.k_norm),
            ("ffn_norm", block.ffn_norm),
        ):
            add(f"blk.{index}.{name}.weight", value)
        for name, projection in (("gate", block.gate), ("up", block.up), ("down", block.down)):
            base = f"blk.{index}.ffn_{name}"
            bound = projection.log_clip.exp()
            if not bool(torch.isfinite(bound)) or float(bound.detach()) <= 0:
                raise ValueError("invalid export clipping bound")
            writer.add_uint32(base + ".w1ax.bits", projection.bits)
            writer.add_uint32(base + ".w1ax.mode", 2)
            writer.add_float32(base + ".w1ax.clip", float(bound.detach()))
            tensors.append(
                ExportTensor(
                    base + ".w1a1_packed", packed_signs(projection.latent_sign), kinds.I32, "I32"
                )
            )
            add(base + ".w1a1_scale", projection.scale.clamp_min(0))
            add(base + ".w1ax_midpoint", projection.midpoint)
            if name != "up":
                add(base + ".w1ax_gain", projection.log_gain.exp())
    for tensor in reader.tensors:
        if tensor.name in selected or "norm.weight" in tensor.name:
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
    tensors.append(ExportTensor("output.weight", cast(NativeArray, head.data), kinds.Q4_K, "Q4_K"))
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
        writer, tensors = prepare(worker, source, vocabulary)
        required = sum(tensor.data.nbytes for tensor in tensors) + 2 * max(
            tensor.data.nbytes for tensor in tensors
        )
        if type(scratch_bytes) is not int or required > scratch_bytes:
            raise ValueError("export scratch budget exceeded")
        expected = {
            tensor.name: (tensor.kind_name, hashlib.sha256(tensor.data.tobytes()).hexdigest())
            for tensor in tensors
        }
        for tensor in tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.kind)

        def write(output: BinaryIO) -> None:
            writer.fout = [output]
            writer.state = cast(ExportAPI, gguf).WriterState.EMPTY
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            writer.write_padding(output, output.tell())
            for tensor in tensors:
                view = cast(
                    "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]",
                    tensor.data.view(np.uint8).reshape(-1),
                )
                for offset in range(0, view.nbytes, 1024 * 1024):
                    output.write(view[offset : offset + 1024 * 1024].tobytes())
                writer.write_padding(output, view.nbytes)

        def verify(artifact: Artifact) -> None:
            reader = cast(ExportAPI, gguf).GGUFReader(str(artifact.path))
            actual = {
                tensor.name: (
                    tensor.tensor_type.name,
                    hashlib.sha256(cast(NativeArray, tensor.data).tobytes()).hexdigest(),
                )
                for tensor in reader.tensors
            }
            if expected != actual or len(reader.tensors) != len(expected):
                raise ValueError("GGUF producer/reader tensor disagreement")
            if reader.fields["dflash.w1ax.config_hash"].contents() != worker.identity:
                raise ValueError("GGUF recipe identity disagreement")

        return store.commit_with(write, worker.identity, worker.updates, max_bytes, verify)

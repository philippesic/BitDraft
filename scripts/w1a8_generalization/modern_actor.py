"""Modern 4B W1A8 actor from a packed initializer, with opt-in compact scale training.

No HF model or target embedding table is loaded. The caller authenticates borrowed
native F32 rows through admission and supplies ExternalNativeF32Embedding. Compact
state covers model scales only: the outer recipe owns optimizer, RNG and cursor.
CPU A8 arithmetic is unchanged and still requires bounded native trajectory checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from w1a1_eagle import native_step, recurrent_binary, recurrent_qat
from w1a1_eagle.native_step import (
    NATIVE_NORM_PATHS,
    ExternalNativeF32Embedding,
    NativeStepAdapter,
    bind_frozen_norms,
)
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, unpack_signs
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

INITIALIZER_SHA256 = "6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048"
CONFIG_SHA256 = "212d1922aafb9b28b0d855e7e04c825c898d5ada6c87a941bcc42b92472bd4e5"
SCALE_STATE_SCHEMA = "modern4b_w1a8_scale_offsets_v1"
MODERN_DIMENSIONS = {
    "hidden_size": 2560,
    "intermediate_size": 7680,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 160,
    "num_hidden_layers": 1,
    "draft_vocab_size": 50000,
    "vocab_size": 248320,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_pin(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def _tensor_hash(value: torch.Tensor) -> str:
    return _sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def projection_shapes(config: dict) -> dict[str, tuple[int, int]]:
    h, f = config["hidden_size"], config["intermediate_size"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    return {
        "fc": (h, 3 * h),
        "output": (config["draft_vocab_size"], h),
        "blk.0.attn_q": (q, 2 * h),
        "blk.0.attn_k": (kv, 2 * h),
        "blk.0.attn_v": (kv, 2 * h),
        "blk.0.attn_output": (h, q),
        "blk.0.ffn_gate": (f, h),
        "blk.0.ffn_up": (f, h),
        "blk.0.ffn_down": (h, f),
    }


def qk_row_indices(heads: int, head_dim: int, rotary_dim: int) -> np.ndarray:
    """Native row i takes source row order[i]; unpermute with argsort(order)."""
    _require(
        all(type(x) is int and x > 0 for x in (heads, head_dim, rotary_dim))
        and rotary_dim <= head_dim
        and rotary_dim % 2 == 0,
        "invalid partial Q/K geometry",
    )
    order = np.arange(head_dim)
    order[:rotary_dim] = np.arange(rotary_dim).reshape(2, rotary_dim // 2).T.reshape(-1)
    return (np.arange(heads)[:, None] * head_dim + order).reshape(-1)


def validate_metadata(reader: Any, config: dict) -> dict[str, dict[str, int]]:
    """Fail closed on missing/contradictory modern metadata, without legacy fallbacks."""
    _require(
        all(config.get(k) == v and type(config[k]) is int for k, v in MODERN_DIMENSIONS.items()),
        "not the modern 4B actor geometry",
    )
    rope = {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}
    _require(
        config.get("rope_parameters") == rope and config.get("partial_rotary_factor") == 0.25,
        "unsupported config RoPE",
    )
    _require(
        config.get("pretraining_tp") == 1
        and config.get("hidden_act") == "silu"
        and config.get("rms_norm_eps") == 1e-6
        and not config.get("attention_bias")
        and not config.get("mlp_bias")
        and not config.get("tie_word_embeddings"),
        "unsupported actor config semantics",
    )
    expected = {
        "general.architecture": "eagle3",
        "eagle3.block_count": 1,
        "eagle3.embedding_length": config["hidden_size"],
        "eagle3.target_hidden_size": config["target_hidden_size"],
        "eagle3.feed_forward_length": config["intermediate_size"],
        "eagle3.context_length": config["max_position_embeddings"],
        "eagle3.attention.head_count": config["num_attention_heads"],
        "eagle3.attention.head_count_kv": config["num_key_value_heads"],
        "eagle3.attention.key_length": config["head_dim"],
        "eagle3.attention.value_length": config["head_dim"],
        "eagle3.rope.dimension_count": int(config["head_dim"] * rope["partial_rotary_factor"]),
        "eagle3.rope.freq_base": rope["rope_theta"],
        "eagle3.attention.layer_norm_rms_epsilon": float(np.float32(config["rms_norm_eps"])),
        "eagle3.vocab_size": config["vocab_size"],
        "eagle3.norm_before_fc": False,
        "eagle3.norm_before_residual": False,
        "eagle3.target_layers": [
            x + 1 for x in config["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
        ],
        "eagle3.w1a1.version": 2,
        "eagle3.w1a1.scale_group_size": 0,
        "eagle3.w1a1.activation_bits": 8,
        "eagle3.w1a1.bit_order": "little",
        "eagle3.w1a1.sign_rule": "nonnegative_is_one",
        "eagle3.w1a1.scale_rule": "f32_learned_nonnegative",
        "eagle3.w1a1.arithmetic": "f32",
    }
    for key, value in expected.items():
        _require(
            key in reader.fields and reader.fields[key].contents() == value,
            "missing or contradictory metadata: " + key,
        )
    for base, (_, logical_k) in projection_shapes(config).items():
        prefix = "eagle3.w1a1.tensor." + (base + ".weight").replace(".", "_")
        for suffix, value in (
            ("logical_k", logical_k),
            ("packed", base + ".w1a1_packed"),
            ("scale", base + ".w1a1_scale"),
        ):
            key = prefix + "." + suffix
            _require(
                key in reader.fields and reader.fields[key].contents() == value,
                "missing or contradictory projection metadata: " + key,
            )
    return {
        base: {
            "heads": config[key],
            "head_dim": config["head_dim"],
            "rotary_dim": expected["eagle3.rope.dimension_count"],
        }
        for base, key in (
            ("blk.0.attn_q", "num_attention_heads"),
            ("blk.0.attn_k", "num_key_value_heads"),
        )
    }


@dataclass
class ModernActor:
    drafter: nn.Module
    adapter: NativeStepAdapter
    linears: dict[str, RowBinaryLinear]  # Existing path-keyed training API.
    initializer: dict
    config: dict
    contract: dict
    qk_layout: dict
    initial_sign_hashes: dict[str, str]
    initial_scale_hashes: dict[str, str]
    scale_only: bool = False

    def scale_parameters(self) -> tuple[nn.Parameter, ...]:
        """The only nine optimizer tensors after explicit scale-only opt-in."""
        _check_scale_only(self)
        return tuple(self.linears[path].scale_offset for path in CANDIDATE_D_BASE_TO_PATH.values())


def _ancestry(actor: ModernActor) -> dict:
    return {
        "initializer": {k: actor.initializer[k] for k in ("sha256", "bytes")},
        "config": {k: actor.config[k] for k in ("sha256", "bytes")},
        "contract": json.loads(json.dumps(actor.contract)),
    }


def _check_scale_only(actor: ModernActor, *, frozen_integrity: bool = False) -> None:
    _require(
        actor.scale_only and set(actor.linears) == set(CANDIDATE_D_BASE_TO_PATH.values()),
        "explicit nine-projection scale-only mode is required",
    )
    expected = {id(m.scale_offset) for m in actor.linears.values()}
    actual = {id(p) for p in actor.drafter.parameters() if p.requires_grad}
    _require(
        actual == expected and len(expected) == 9,
        "optimizer scope is not exactly nine scale offsets",
    )
    for path, module in actor.linears.items():
        _require(
            module.latent_sign.requires_grad is False
            and module.scale_offset.requires_grad
            and module.scale_offset.dtype == torch.float32
            and module.contract == W1AxContract(8)
            and module.a1_computation == "reference",
            "scale-only arithmetic/parameter contract changed",
        )
        if frozen_integrity:
            _require(
                _tensor_hash(module.latent_sign) == actor.initial_sign_hashes[path],
                "frozen signs changed: " + path,
            )
            _require(
                _tensor_hash(module.initial_scale) == actor.initial_scale_hashes[path],
                "initializer scales changed: " + path,
            )


def enable_scale_only(actor: ModernActor) -> tuple[nn.Parameter, ...]:
    """Explicitly freeze signs/norms; leave all existing forward arithmetic intact."""
    for parameter in actor.drafter.parameters():
        parameter.requires_grad_(False)
    for module in actor.linears.values():
        module.scale_offset.requires_grad_(True)
    actor.scale_only = True
    return actor.scale_parameters()


def build_modern_actor(
    initializer: str | Path,
    config: str | Path,
    *,
    embedding_lookup: ExternalNativeF32Embedding,
    device: str | torch.device = "cpu",
    scale_only: bool = False,
) -> ModernActor:
    """Construct the frozen modern initializer; a new initializer requires a new pin."""
    from gguf import GGMLQuantizationType as Type
    from gguf import GGUFReader

    _require(
        isinstance(embedding_lookup, ExternalNativeF32Embedding),
        "external native F32 lookup required",
    )
    init_pin, config_pin = file_pin(initializer), file_pin(config)
    _require(init_pin["sha256"] == INITIALIZER_SHA256, "initializer ancestry mismatch")
    _require(config_pin["sha256"] == CONFIG_SHA256, "config ancestry mismatch")
    cfg = json.loads(Path(config).read_text())
    _require(
        (embedding_lookup.hidden_size, embedding_lookup.vocab_size)
        == (cfg["hidden_size"], cfg["vocab_size"]),
        "embedding dimensions differ",
    )
    reader = GGUFReader(str(initializer), mode="r")
    layout = validate_metadata(reader, cfg)
    tensors = {t.name: t for t in reader.tensors}
    expected = (
        {
            base + suffix
            for base in CANDIDATE_D_BASE_TO_PATH
            for suffix in (".w1a1_packed", ".w1a1_scale")
        }
        | set(NATIVE_NORM_PATHS)
        | {"d2t"}
    )
    _require(
        len(tensors) == len(reader.tensors) and set(tensors) == expected,
        "expected exactly nine packed projections, four norms and d2t; no embedding/dense shadows",
    )
    drafter = nn.Module()
    drafter.config = SimpleNamespace(**cfg)
    drafter.early_stop_method = drafter.tree_mask = None
    drafter.midlayer = nn.Module()
    drafter.midlayer.self_attn, drafter.midlayer.mlp = nn.Module(), nn.Module()
    for path in NATIVE_NORM_PATHS.values():
        parent, name = path.rsplit(".", 1) if "." in path else ("", path)
        norm = nn.Module()
        norm.weight = nn.Parameter(torch.ones(cfg["hidden_size"]), requires_grad=False)
        norm.variance_epsilon = cfg["rms_norm_eps"]
        setattr(drafter.get_submodule(parent) if parent else drafter, name, norm)
    norms = {}
    for name in NATIVE_NORM_PATHS:
        tensor = tensors[name]
        _require(
            tensor.tensor_type == Type.F32 and tensor.data.shape == (cfg["hidden_size"],),
            "invalid native norm: " + name,
        )
        norms[name] = torch.from_numpy(np.array(tensor.data, dtype=np.float32, copy=True))
    bind_frozen_norms(drafter, norms)
    mapping = tensors["d2t"]
    absolute = np.array(mapping.data, dtype=np.int64, copy=True)
    _require(
        mapping.tensor_type == Type.I64
        and absolute.shape == (cfg["draft_vocab_size"],)
        and len(np.unique(absolute)) == len(absolute)
        and absolute.min() >= 0
        and absolute.max() < cfg["vocab_size"],
        "invalid native absolute d2t map",
    )
    membership = torch.zeros(cfg["vocab_size"], dtype=torch.bool)
    membership[torch.from_numpy(absolute)] = True
    drafter.register_buffer(
        "d2t", torch.from_numpy(absolute - np.arange(len(absolute), dtype=np.int64))
    )
    drafter.register_buffer("t2d", membership)
    linears, sign_hashes, scale_hashes = {}, {}, {}
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        rows, logical_k = projection_shapes(cfg)[base]
        packed, scales = tensors[base + ".w1a1_packed"], tensors[base + ".w1a1_scale"]
        _require(
            packed.tensor_type == Type.I32
            and packed.data.shape == (rows, math.ceil(logical_k / 32))
            and scales.tensor_type == Type.F32
            and scales.data.shape == (rows,),
            "projection shape/type: " + base,
        )
        signs = unpack_signs(packed.data, logical_k) * np.float32(0.5)
        values = np.array(scales.data, dtype=np.float32, copy=True)
        if base in layout:
            inverse_order = np.argsort(qk_row_indices(**layout[base]))
            signs, values = signs[inverse_order].copy(), values[inverse_order].copy()
        module = RowBinaryLinear(torch.from_numpy(signs), torch.from_numpy(values), W1AxContract(8))
        parent, name = path.rsplit(".", 1) if "." in path else ("", path)
        setattr(drafter.get_submodule(parent) if parent else drafter, name, module)
        linears[path] = module
        sign_hashes[path], scale_hashes[path] = (
            _tensor_hash(module.latent_sign),
            _tensor_hash(module.initial_scale),
        )
        del signs, values
    source_pins = {
        Path(m.__file__).name: file_pin(m.__file__)["sha256"]
        for m in (native_step, recurrent_binary, recurrent_qat)
    }
    source_pins["modern_actor.py"] = file_pin(__file__)["sha256"]
    semantics = {
        "schema": SCALE_STATE_SCHEMA,
        "activation_bits": 8,
        "scale_layout": "row",
        "arithmetic": "unchanged_RowBinaryLinear_reference_A8",
        "qk_layout": layout,
        "signs": "packed_little_endian_to_plus_minus_half_inverse_partial_qk",
        "mapping": "native_absolute_d2t_to_source_offsets_and_bool_membership",
        "shape": {k: list(v) for k, v in projection_shapes(cfg).items()},
        "norms": {k: _tensor_hash(v) for k, v in norms.items()},
        "absolute_d2t_sha256": _sha_bytes(absolute.astype("<i8").tobytes()),
        "sources": source_pins,
    }
    contract = {
        "sha256": _sha_bytes(json.dumps(semantics, sort_keys=True).encode()),
        "semantics": semantics,
    }
    drafter.to(device=device)
    adapter = NativeStepAdapter(
        drafter, embedding_lookup=embedding_lookup, embedding_precision="external_native_f32"
    )
    actor = ModernActor(
        drafter, adapter, linears, init_pin, config_pin, contract, layout, sign_hashes, scale_hashes
    )
    if scale_only:
        enable_scale_only(actor)
    return actor


def collect_scale_state(actor: ModernActor) -> dict:
    """CPU F32 offset copies; deliberately excludes full signs and all outer recipe state."""
    _check_scale_only(actor, frozen_integrity=True)
    offsets = {
        base: actor.linears[path].scale_offset.detach().cpu().clone()
        for base, path in CANDIDATE_D_BASE_TO_PATH.items()
    }
    _require(
        all(torch.isfinite(x).all().item() for x in offsets.values()), "nonfinite scale offset"
    )
    return {"schema": SCALE_STATE_SCHEMA, "ancestry": _ancestry(actor), "scale_offsets": offsets}


def load_scale_state(actor: ModernActor, state: dict) -> None:
    """Validate the complete compact state before changing any model parameter."""
    _check_scale_only(actor, frozen_integrity=True)
    _require(
        isinstance(state, dict)
        and set(state) == {"schema", "ancestry", "scale_offsets"}
        and state["schema"] == SCALE_STATE_SCHEMA
        and state["ancestry"] == _ancestry(actor),
        "scale-state schema or initializer/config/contract ancestry mismatch",
    )
    values = state["scale_offsets"]
    _require(
        isinstance(values, dict) and set(values) == set(CANDIDATE_D_BASE_TO_PATH),
        "expected exactly nine known scale offsets",
    )
    pending = []
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        value, destination = values[base], actor.linears[path].scale_offset
        _require(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            and value.device.type == "cpu"
            and value.shape == destination.shape
            and not value.requires_grad
            and bool(torch.isfinite(value).all()),
            "invalid compact scale shape/dtype/finite: " + base,
        )
        pending.append((destination, value.to(destination.device)))
    with torch.no_grad():
        for destination, value in pending:
            destination.copy_(value)


def export_scale_only_gguf(actor: ModernActor, destination: str | Path) -> dict:
    """Copy frozen GGUF, patch only nine scale payloads, publish atomically without overwrite.

    Output is a native candidate requiring actual replay/serving validation. Negative
    raw scales retain their exact offsets in compact state; export uses the existing
    effective_scales() nonnegative projection without mutating training parameters.
    """
    from gguf import GGMLQuantizationType as Type
    from gguf import GGUFReader

    _check_scale_only(actor, frozen_integrity=True)
    destination = Path(destination).resolve()
    source = Path(actor.initializer["path"])
    _require(
        destination != source.resolve() and not destination.exists(),
        "export cannot overwrite any existing file",
    )
    _require(file_pin(source) == actor.initializer, "initializer changed before export")
    reader = GGUFReader(str(source), mode="r")
    tensors = {t.name: t for t in reader.tensors}
    pending = []
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        module = actor.linears[path]
        _require(bool(torch.isfinite(module.scale_offset).all()), "nonfinite scale offset: " + base)
        values = module.effective_scales().detach().cpu().numpy().astype("<f4", copy=True)
        _require(
            np.isfinite(values).all() and (values >= 0).all(), "invalid effective scales: " + base
        )
        if base in actor.qk_layout:
            values = values[qk_row_indices(**actor.qk_layout[base])].copy()
        tensor = tensors[base + ".w1a1_scale"]
        _require(
            tensor.tensor_type == Type.F32
            and tuple(tensor.data.shape) == values.shape
            and tensor.n_bytes == values.nbytes,
            "native scale payload differs: " + base,
        )
        pending.append((base, tensor.data_offset, values.tobytes()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix="." + destination.name + ".", dir=destination.parent
    )
    temporary = Path(temporary)
    published = False
    try:
        with os.fdopen(handle, "w+b") as stream, source.open("rb") as original:
            shutil.copyfileobj(original, stream, length=1024 * 1024)
            for _, offset, payload in pending:
                stream.seek(offset)
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Byte-check unchanged intervals and exact replacements before publication.
        ranges = sorted(pending, key=lambda x: x[1])
        cursor = 0
        with source.open("rb") as original, temporary.open("rb") as candidate:
            for _, offset, payload in ranges:
                remaining = offset - cursor
                while remaining:
                    length = min(1024 * 1024, remaining)
                    _require(
                        original.read(length) == candidate.read(length),
                        "non-scale export byte changed",
                    )
                    remaining -= length
                original.seek(len(payload), 1)
                _require(candidate.read(len(payload)) == payload, "export scale payload differs")
                cursor = offset + len(payload)
            while chunk := original.read(1024 * 1024):
                _require(chunk == candidate.read(len(chunk)), "non-scale export tail changed")
            _require(candidate.read(1) == b"", "export grew unexpectedly")
        # Hard-link publication is atomic and fails if another writer claimed destination.
        os.link(temporary, destination)
        published = True
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    _require(published, "export was not published")
    return {
        "schema": "modern4b_native_scale_export_v1",
        "initializer": actor.initializer,
        "config": actor.config,
        "contract": actor.contract,
        "output": file_pin(destination),
        "replaced_payloads": [
            {"base": base, "offset": offset, "bytes": len(payload), "sha256": _sha_bytes(payload)}
            for base, offset, payload in pending
        ],
        "all_other_bytes_unchanged": True,
        "native_qualified": False,
    }

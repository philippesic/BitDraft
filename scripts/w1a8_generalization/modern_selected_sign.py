"""Opt-in attention-output sign QAT state/export, separate from scale-only APIs.

Build a ModernActor with scale_only=False, explicitly enable this mask, and let
the outer recipe own Adam/RNG/cursor. Only the O-projection latent signs and all
nine raw scale offsets are trainable. No trainer or native admission is provided.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from modern_actor import ModernActor, _ancestry, file_pin, qk_row_indices

from w1a1_eagle.native_step import NATIVE_NORM_PATHS
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, pack_signs, unpack_signs
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

STATE_SCHEMA = "modern4b_selected_O_sign_and_nine_scale_offsets_v1"
EXPORT_SCHEMA = "modern4b_selected_O_native_export_v1"
SELECTED_BASES = ("blk.0.attn_output",)
SELECTED_PATH = CANDIDATE_D_BASE_TO_PATH[SELECTED_BASES[0]]
_CHUNK_ELEMENTS = 262144


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor_hash(value):
    """Bound host transfers to 1MiB; do not clone all nine latent matrices."""
    _require(value.is_contiguous(), "tensor must be contiguous")
    digest = hashlib.sha256()
    flat = value.detach().view(-1)
    for chunk in flat.split(_CHUNK_ELEMENTS):
        digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()


def _finite(value):
    return all(
        bool(torch.isfinite(chunk).all())
        for chunk in value.detach().view(-1).split(_CHUNK_ELEMENTS)
    )


def _check_structure(actor):
    _require(
        isinstance(actor, ModernActor) and not actor.scale_only,
        "build explicit non-scale-only ModernActor",
    )
    mapping = CANDIDATE_D_BASE_TO_PATH
    _require(set(actor.linears) == set(mapping.values()), "exactly nine known projections required")
    _require(len({id(m) for m in actor.linears.values()}) == 9, "projection modules must not alias")
    semantics = actor.contract.get("semantics", {})
    shapes = semantics.get("shape", {})
    _require(set(shapes) == set(mapping), "complete source-order projection geometry required")
    _require(
        set(actor.initial_sign_hashes) == set(mapping.values())
        and set(actor.initial_scale_hashes) == set(mapping.values()),
        "complete initial tensor identities required",
    )
    parameters = set()
    for base, path in mapping.items():
        module = actor.linears[path]
        shape = tuple(shapes[base])
        _require(
            isinstance(module, RowBinaryLinear)
            and actor.drafter.get_submodule(path) is module
            and len(shape) == 2
            and all(type(n) is int and n > 0 for n in shape)
            and tuple(module.latent_sign.shape) == shape
            and tuple(module.initial_scale.shape) == tuple(module.scale_offset.shape) == (shape[0],)
            and module.latent_sign.dtype
            == module.initial_scale.dtype
            == module.scale_offset.dtype
            == torch.float32
            and module.latent_sign.is_contiguous()
            and module.contract == W1AxContract(8)
            and module.a1_computation == "reference"
            and module.frozen_bias is None
            and getattr(module, "activation_quantizer", None) is None
            and getattr(module, "affine_binary", None) is None,
            "selected-sign projection/arithmetic contract differs: " + base,
        )
        parameters.update((id(module.latent_sign), id(module.scale_offset)))
    _require(len(parameters) == 18, "nine sign and nine scale tensors must be distinct")
    _require(
        set(actor.qk_layout) == {"blk.0.attn_q", "blk.0.attn_k"}
        and semantics.get("qk_layout") == actor.qk_layout,
        "original partial Q/K layout required; O cannot be permuted",
    )
    for base, layout in actor.qk_layout.items():
        order = qk_row_indices(**layout)
        _require(len(order) == shapes[base][0], "Q/K row geometry differs")
    _require(
        set(semantics.get("norms", {})) == set(NATIVE_NORM_PATHS),
        "complete frozen norm identities required",
    )
    for path in NATIVE_NORM_PATHS.values():
        weight = actor.drafter.get_submodule(path).weight
        _require(weight.dtype == torch.float32, "frozen norm dtype differs")
        parameters.add(id(weight))
    _require(
        {id(p) for p in actor.drafter.parameters()} == parameters,
        "unexpected actor parameters outside declared native linears/norms",
    )


def selected_sign_contract(actor, selected_bases=SELECTED_BASES):
    """Immutable source-order selection and initializer/config/source ancestry."""
    _require(
        tuple(selected_bases) == SELECTED_BASES,
        "only the explicit attention-output sign mask is supported",
    )
    _check_structure(actor)
    return {
        "schema": STATE_SCHEMA,
        "helper": file_pin(__file__),
        "selected_bases": list(SELECTED_BASES),
        "selected_paths": [SELECTED_PATH],
        "selected_shapes": {
            base: list(actor.linears[CANDIDATE_D_BASE_TO_PATH[base]].latent_sign.shape)
            for base in SELECTED_BASES
        },
        "source_order_scales": list(CANDIDATE_D_BASE_TO_PATH),
        "packing": (
            "nonnegative including signed zero is one; little-endian I32; unused tail bits zero"
        ),
        "native_selected_mapping": {
            "blk.0.attn_output.w1a1_packed": SELECTED_PATH + ".latent_sign"
        },
        "qk_scale_permutation": copy.deepcopy(actor.qk_layout),
        "frozen_initial_sign_hashes": copy.deepcopy(actor.initial_sign_hashes),
        "frozen_initial_scale_hashes": copy.deepcopy(actor.initial_scale_hashes),
    }


def _check_frozen(actor, *, include_selected_initial=False):
    """Verify all non-exported model values before state/export boundaries."""
    for path, module in actor.linears.items():
        _require(
            _tensor_hash(module.initial_scale) == actor.initial_scale_hashes[path],
            "initializer scales changed: " + path,
        )
        if path != SELECTED_PATH or include_selected_initial:
            _require(
                _tensor_hash(module.latent_sign) == actor.initial_sign_hashes[path],
                "frozen signs changed: " + path,
            )
    semantics = actor.contract["semantics"]
    for name, path in NATIVE_NORM_PATHS.items():
        _require(
            _tensor_hash(actor.drafter.get_submodule(path).weight) == semantics["norms"][name],
            "frozen norm changed: " + name,
        )
    offsets = actor.drafter.d2t.detach().cpu()
    membership = actor.drafter.t2d.detach().cpu()
    _require(offsets.ndim == 1 and offsets.dtype == torch.int64, "d2t offsets differ")
    absolute = offsets + torch.arange(len(offsets), dtype=torch.int64)
    _require(
        membership.ndim == 1
        and membership.dtype == torch.bool
        and len(torch.unique(absolute)) == len(absolute)
        and bool((absolute >= 0).all())
        and bool((absolute < len(membership)).all()),
        "native vocabulary map differs",
    )
    expected_membership = torch.zeros_like(membership)
    expected_membership[absolute] = True
    _require(
        torch.equal(membership, expected_membership)
        and hashlib.sha256(absolute.numpy().astype("<i8").tobytes()).hexdigest()
        == semantics.get("absolute_d2t_sha256"),
        "frozen native vocabulary map changed",
    )


def _check_selected(actor, *, integrity=False, bounded=True):
    _check_structure(actor)
    expected_contract = selected_sign_contract(actor)
    _require(
        getattr(actor, "_selected_sign_contract", None) == expected_contract,
        "explicit selected-sign opt-in/contract required",
    )
    expected = {id(actor.linears[SELECTED_PATH].latent_sign)} | {
        id(m.scale_offset) for m in actor.linears.values()
    }
    _require(
        {id(p) for p in actor.drafter.parameters() if p.requires_grad} == expected,
        "optimizer scope is not selected O signs plus nine raw scale offsets",
    )
    latent = actor.linears[SELECTED_PATH].latent_sign
    _require(_finite(latent), "nonfinite selected latent signs")
    if bounded:
        _require(
            bool((latent.detach().abs() <= 1).all()),
            "selected latent signs must be clamped to [-1,1]",
        )
    for path, module in actor.linears.items():
        _require(_finite(module.scale_offset), "nonfinite raw scale offset: " + path)
    if integrity:
        _check_frozen(actor)
    return expected_contract


def enable_selected_sign(actor, selected_bases=SELECTED_BASES):
    """Validate before changing flags; freeze all except selected O signs/nine scales."""
    contract = selected_sign_contract(actor, selected_bases)
    if hasattr(actor, "_selected_sign_contract"):
        _check_selected(actor, integrity=True)
        return copy.deepcopy(contract)
    _check_frozen(actor, include_selected_initial=True)
    for module in actor.linears.values():
        _require(
            _finite(module.latent_sign) and _finite(module.scale_offset),
            "nonfinite initial trainable values",
        )
    for parameter in actor.drafter.parameters():
        parameter.requires_grad_(False)
    actor.linears[SELECTED_PATH].latent_sign.requires_grad_(True)
    for module in actor.linears.values():
        module.scale_offset.requires_grad_(True)
    actor._selected_sign_contract = copy.deepcopy(contract)
    return contract


def selected_parameter_groups(actor, *, sign_lr=None, scale_lr=None):
    """Return separate sign/scale optimizer groups without constructing an optimizer."""
    _check_selected(actor)
    groups = [
        {"name": "selected_O_latent_sign", "params": (actor.linears[SELECTED_PATH].latent_sign,)},
        {
            "name": "nine_raw_scale_offsets",
            "params": tuple(
                actor.linears[path].scale_offset for path in CANDIDATE_D_BASE_TO_PATH.values()
            ),
        },
    ]
    for group, lr in zip(groups, (sign_lr, scale_lr), strict=True):
        if lr is not None:
            _require(
                type(lr) in (int, float) and math.isfinite(lr) and lr > 0,
                "positive finite learning rate required",
            )
            group["lr"] = float(lr)
    return groups


def clamp_selected_latents_(actor):
    """Project only selected finite latent signs; scale projection stays recipe-owned."""
    _check_selected(actor, bounded=False)
    with torch.no_grad():
        actor.linears[SELECTED_PATH].latent_sign.clamp_(-1, 1)


def collect_selected_state(actor):
    """Detached CPU F32 source-order O latent and all nine raw offsets, no Adam/RNG."""
    contract = _check_selected(actor, integrity=True)
    return {
        "schema": STATE_SCHEMA,
        "ancestry": _ancestry(actor),
        "selection": contract,
        "latent_signs": {
            base: actor.linears[CANDIDATE_D_BASE_TO_PATH[base]].latent_sign.detach().cpu().clone()
            for base in SELECTED_BASES
        },
        "scale_offsets": {
            base: actor.linears[path].scale_offset.detach().cpu().clone()
            for base, path in CANDIDATE_D_BASE_TO_PATH.items()
        },
    }


def _validate_state(actor, state):
    contract = _check_selected(actor, integrity=True)
    _require(
        isinstance(state, dict)
        and set(state) == {"schema", "ancestry", "selection", "latent_signs", "scale_offsets"}
        and state["schema"] == STATE_SCHEMA
        and state["ancestry"] == _ancestry(actor)
        and state["selection"] == contract,
        "selected-state schema/selection/initializer/config/source ancestry differs",
    )
    pending = []
    for key, mapping in (
        ("latent_signs", {base: CANDIDATE_D_BASE_TO_PATH[base] for base in SELECTED_BASES}),
        ("scale_offsets", CANDIDATE_D_BASE_TO_PATH),
    ):
        values = state[key]
        _require(
            isinstance(values, dict) and set(values) == set(mapping),
            "selected-state tensor mask differs: " + key,
        )
        for base, path in mapping.items():
            value = values[base]
            destination = getattr(
                actor.linears[path], "latent_sign" if key == "latent_signs" else "scale_offset"
            )
            _require(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and value.device.type == "cpu"
                and value.shape == destination.shape
                and value.is_contiguous()
                and not value.requires_grad
                and _finite(value),
                "invalid selected-state tensor shape/dtype/finite: " + key + "/" + base,
            )
            if key == "latent_signs":
                _require(bool((value.abs() <= 1).all()), "state latent signs exceed [-1,1]")
            pending.append((destination, value))
    return pending


def load_selected_state(actor, state):
    """Validate every tensor and prepare transfers before mutating any parameter."""
    pending = _validate_state(actor, state)
    # Snapshot all sources even when a caller supplied CPU views aliasing live
    # parameters; no earlier copy may alter a later validated payload.
    prepared = [
        (destination, value.to(destination.device).clone()) for destination, value in pending
    ]
    with torch.no_grad():
        for destination, value in prepared:
            destination.copy_(value)


def selected_state_budget(actor):
    """Arithmetic payload estimates, not measured memory or serialization sizes."""
    _check_selected(actor)
    signs = actor.linears[SELECTED_PATH].latent_sign.numel()
    scales = sum(m.scale_offset.numel() for m in actor.linears.values())
    raw = 4 * (signs + scales)
    return {
        "selected_sign_parameters": signs,
        "scale_parameters": scales,
        "F32_model_state_payload_bytes": raw,
        "F32_model_plus_Adam_two_moments_payload_bytes": 3 * raw,
        "additional_F32_gradient_payload_bytes": raw,
        "estimated_not_measured": True,
        "excludes": (
            "serialization metadata, Adam scalar steps, RNG, activations, "
            "frozen model, allocator and temporary buffers"
        ),
    }


def _initializer_reader(actor):
    from gguf import GGUFReader

    source = Path(actor.initializer["path"])
    _require(file_pin(source) == actor.initializer, "initializer changed")
    reader = GGUFReader(str(source), mode="r")
    tensors = {t.name: t for t in reader.tensors}
    _require(len(tensors) == len(reader.tensors), "duplicate initializer tensor")
    return source, reader, tensors


def selected_sign_metrics(actor, reference_state=None):
    """Report actual latent displacement and hard-sign flips in selected source order."""
    _check_selected(actor, integrity=True)
    if reference_state is not None:
        _validate_state(actor, reference_state)
        initial = reference_state["latent_signs"][SELECTED_BASES[0]]
        reference_kind = "explicit same-ancestry selected compact state"
    else:
        _, _reader, tensors = _initializer_reader(actor)
        module = actor.linears[SELECTED_PATH]
        tensor = tensors[SELECTED_BASES[0] + ".w1a1_packed"]
        initial = torch.from_numpy(unpack_signs(tensor.data, module.in_features) * np.float32(0.5))
        reference_kind = "pinned initializer plus/minus half latents; O has no Q/K permutation"
    current = actor.linears[SELECTED_PATH].latent_sign.detach().view(-1)
    original = initial.view(-1)
    _require(current.shape == original.shape, "metric reference shape differs")
    changed = flips = saturated = 0
    total = maximum = 0.0
    minimum, maximum_latent = math.inf, -math.inf
    for start in range(0, current.numel(), _CHUNK_ELEMENTS):
        a = current[start : start + _CHUNK_ELEMENTS].cpu()
        b = original[start : start + _CHUNK_ELEMENTS]
        delta = (a - b).abs()
        changed += int(a.ne(b).sum())
        flips += int(a.ge(0).ne(b.ge(0)).sum())
        saturated += int(a.abs().eq(1).sum())
        total += float(delta.double().sum())
        maximum = max(maximum, float(delta.max()))
        minimum, maximum_latent = min(minimum, float(a.min())), max(maximum_latent, float(a.max()))
    return {
        "base": SELECTED_BASES[0],
        "reference": reference_kind,
        "parameters": current.numel(),
        "changed_latents": changed,
        "hard_sign_flips": flips,
        "mean_absolute_displacement": total / current.numel(),
        "max_absolute_displacement": maximum,
        "saturated_latents": saturated,
        "latent_min": minimum,
        "latent_max": maximum_latent,
        "measured_from_current_tensors": True,
    }


def export_selected_sign_gguf(actor, destination):
    """Patch selected O packed signs/nine scales; audit every other byte unchanged."""
    from gguf import GGMLQuantizationType as Type

    contract = _check_selected(actor, integrity=True)
    source, _reader, tensors = _initializer_reader(actor)
    destination = Path(destination).resolve()
    _require(
        destination != source.resolve() and not destination.exists(),
        "export cannot overwrite an existing file",
    )
    pending = []
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        module = actor.linears[path]
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
        pending.append((base + ".w1a1_scale", int(tensor.data_offset), values.tobytes()))
    base = SELECTED_BASES[0]
    module = actor.linears[SELECTED_PATH]
    original = tensors[base + ".w1a1_packed"]
    _require(
        original.tensor_type == Type.I32
        and tuple(original.data.shape) == (module.out_features, math.ceil(module.in_features / 32)),
        "native selected packed geometry/type differs",
    )
    # This checks the initializer's unused padding before repacking selected bits.
    unpack_signs(original.data, module.in_features)
    latent = module.latent_sign.detach().cpu().numpy()
    signs = np.where(latent < 0, np.float32(-1), np.float32(1))
    packed = pack_signs(signs)
    _require(packed.nbytes == original.n_bytes, "selected packed payload size differs")
    pending.append((base + ".w1a1_packed", int(original.data_offset), packed.tobytes()))
    ranges = sorted(pending, key=lambda x: x[1])
    cursor = 0
    for _, offset, payload in ranges:
        _require(
            cursor <= offset and offset + len(payload) <= actor.initializer["bytes"],
            "invalid/overlapping native patch ranges",
        )
        cursor = offset + len(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as candidate, source.open("rb") as original_stream:
            shutil.copyfileobj(original_stream, candidate, length=1024 * 1024)
            for _, offset, payload in pending:
                candidate.seek(offset)
                candidate.write(payload)
            candidate.flush()
            os.fsync(candidate.fileno())
        cursor = 0
        with source.open("rb") as original_stream, temporary.open("rb") as candidate:
            for _, offset, payload in ranges:
                remaining = offset - cursor
                while remaining:
                    length = min(1024 * 1024, remaining)
                    _require(
                        original_stream.read(length) == candidate.read(length),
                        "undeclared export bytes changed",
                    )
                    remaining -= length
                original_stream.seek(len(payload), 1)
                _require(candidate.read(len(payload)) == payload, "declared native patch differs")
                cursor = offset + len(payload)
            while chunk := original_stream.read(1024 * 1024):
                _require(chunk == candidate.read(len(chunk)), "undeclared export tail changed")
            _require(candidate.read(1) == b"", "native export size changed")
        _require(file_pin(source) == actor.initializer, "initializer changed during export")
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": EXPORT_SCHEMA,
        "ancestry": _ancestry(actor),
        "selection": contract,
        "output": file_pin(destination),
        "replaced_payloads": [
            {
                "tensor": name,
                "offset": offset,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, offset, payload in ranges
        ],
        "all_other_bytes_unchanged": True,
        "native_qualified": False,
    }

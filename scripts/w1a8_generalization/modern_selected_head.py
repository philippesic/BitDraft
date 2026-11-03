"""Sparse FIT-selected head sign QAT; no model construction or native admission.

The full frozen head still supplies the original full-width A8 GEMM and input
VJP. Only R source-order rows have attached sign masters/gradients/Adam state.
Use round() around a whole forward/backward round; it excludes the head from
ordinary shared signs. A graph must be consumed inside its round. Standalone
forwards build their own signs and support one backward. Higher-order gradients
and retain_graph reuse are deliberately unsupported. The outer recipe owns Adam,
RNG, cursor, schedule, checkpoint publication and storage admission.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import modern_actor as actor_source
import numpy as np
import torch
from modern_actor import _ancestry, file_pin, qk_row_indices
from modern_selected_sign import _check_frozen, _check_structure, _finite, _tensor_hash
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from w1a1_eagle import native_step, recurrent_binary, recurrent_qat
from w1a1_eagle.recurrent_binary import (
    CANDIDATE_D_BASE_TO_PATH,
    hard_sign_ste,
    pack_signs,
    unpack_signs,
)
from w1a1_eagle.recurrent_qat import hard_activation, shared_round_hard_signs

SCHEMA = "modern4b_selected_head_rows_and_nine_scales_v1"
SELECTOR_SCHEMA = "modern4b_fit_supported_head_rows_v1"
HEAD_PATH = CANDIDATE_D_BASE_TO_PATH["output"]


def require(value, message):
    if not value:
        raise ValueError(message)


def checked_pin(ref):
    require(
        isinstance(ref, dict)
        and set(ref) == {"path", "sha256", "bytes"}
        and file_pin(ref["path"]) == ref,
        "source/base pin changed or malformed",
    )
    return copy.deepcopy(ref)


def absolute_map(actor):
    offsets = actor.drafter.d2t.detach()
    return offsets + torch.arange(len(offsets), device=offsets.device, dtype=torch.int64)


def select_head_rows(fit_records, absolute_d2t, *, source_pins, count=512):
    """Select once by unique FIT groups, then label frequency, then target ID.

    Input contains FIT windows only. Unsupported labels are reported, never
    mapped to an arbitrary class. Manifest pins authenticate the caller's input
    ancestry; their semantic FIT role remains the caller's data-audit obligation.
    """
    require(
        isinstance(absolute_d2t, torch.Tensor)
        and absolute_d2t.dtype == torch.int64
        and absolute_d2t.ndim == 1
        and absolute_d2t.device.type == "cpu"
        and absolute_d2t.numel() > 0
        and bool((absolute_d2t >= 0).all())
        and len(absolute_d2t.unique()) == len(absolute_d2t),
        "invalid absolute d2t map",
    )
    require(type(count) is int and 1 <= count <= len(absolute_d2t), "invalid selected row count")
    require(isinstance(source_pins, list) and source_pins, "FIT source manifest pins required")
    pins = [checked_pin(ref) for ref in source_pins]
    inverse = {int(token): row for row, token in enumerate(absolute_d2t.tolist())}
    groups, frequencies, unsupported = defaultdict(set), Counter(), Counter()
    records = []
    for record in fit_records:
        require(
            isinstance(record, dict)
            and record.get("role") == "fit"
            and isinstance(record.get("group_id"), str)
            and record["group_id"]
            and record.get("source_split", "TRAIN") == "TRAIN"
            and not record["group_id"].lower().startswith("fresh")
            and record.get("source_ancestry", {}).get("source_split", "TRAIN") == "TRAIN"
            and isinstance(record.get("labels"), list)
            and record["labels"]
            and all(type(t) is int and t >= 0 for t in record["labels"]),
            "selector accepts FIT TRAIN labels only; DEV/fresh roles forbidden",
        )
        records.append({"group_id": record["group_id"], "labels": list(record["labels"])})
        for token in record["labels"]:
            if token in inverse:
                groups[token].add(record["group_id"])
                frequencies[token] += 1
            else:
                unsupported[token] += 1
    require(
        len(groups) >= count, "not enough observed supported FIT labels; do not pad or consult DEV"
    )
    chosen = sorted(groups, key=lambda token: (-len(groups[token]), -frequencies[token], token))[
        :count
    ]
    total = sum(frequencies.values())
    return {
        "schema": SELECTOR_SCHEMA,
        "helper": file_pin(__file__),
        "source_pins": pins,
        "selection_rule": (
            "unique FIT group coverage descending; frequency descending; target token ascending"
        ),
        "count": count,
        "row_ids": [inverse[token] for token in chosen],
        "target_ids": chosen,
        "map_sha256": _tensor_hash(absolute_d2t),
        "records_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "FIT_only": True,
        "supported_label_count": total,
        "unsupported_label_count": sum(unsupported.values()),
        "selected_label_count": sum(frequencies[token] for token in chosen),
        "supported_label_coverage": sum(frequencies[token] for token in chosen) / total,
        "unique_FIT_groups": len({r["group_id"] for r in records}),
        "rows": [
            {
                "row_id": inverse[t],
                "target_id": t,
                "unique_groups": len(groups[t]),
                "frequency": frequencies[t],
            }
            for t in chosen
        ],
    }


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quantized, full_signs, attached_selected, rows, token):
        ctx.save_for_backward(quantized, full_signs, rows)
        ctx.token = token
        return F.linear(quantized, full_signs)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        token = ctx.token
        require(
            token["valid"] and token["versions"] == token["owner"]._versions(),
            "stale selected-head graph/update or abandoned round",
        )
        quantized, signs, rows = ctx.saved_tensors
        input_grad = gradient.matmul(signs) if ctx.needs_input_grad[0] else None
        selected_grad = None
        if ctx.needs_input_grad[2]:
            selected_output = gradient.index_select(-1, rows).reshape(-1, len(rows))
            selected_grad = selected_output.transpose(0, 1).matmul(
                quantized.reshape(-1, signs.shape[1])
            )
        token["pending"] -= 1
        require(token["pending"] >= 0, "selected-head graph backward reused")
        return input_grad, None, selected_grad, None, None


class SelectedHead:
    """External selected-row Parameter; instance-local forward on the existing head.

    Construct only after a non-scale-only actor and declared scale-only starting
    state have been built/loaded by the caller. Other sign latents must match the
    pinned initializer: a learned-O ancestry requires a separate future contract.
    """

    def __init__(self, actor, selection, *, base_state_pin, learn_signs=True):
        require(type(learn_signs) is bool, "explicit Boolean sign learning required")
        _check_structure(actor)
        require(
            not hasattr(actor, "_selected_sign_contract")
            and not hasattr(actor, "_selected_head_owner")
            and "forward" not in actor.linears[HEAD_PATH].__dict__,
            "actor already has a sign/forward owner",
        )
        _check_frozen(actor, include_selected_initial=True)
        self.actor = actor
        self.head = actor.linears[HEAD_PATH]
        self.selection = copy.deepcopy(selection)
        self.base_state_pin = checked_pin(base_state_pin)
        require(
            file_pin(actor.initializer["path"]) == actor.initializer
            and file_pin(actor.config["path"]) == actor.config,
            "initializer/config pin changed",
        )
        self.learn_signs = learn_signs
        self._validate_selection()
        self.rows = torch.tensor(
            selection["row_ids"], dtype=torch.int64, device=self.head.latent_sign.device
        )
        self.latent = nn.Parameter(
            self.head.latent_sign.detach().index_select(0, self.rows).clone(),
            requires_grad=learn_signs,
        )
        self._cache = None
        self._module_identities = self._identities()
        self._original_forward = self.head.forward
        self._source_contract = {
            "schema": SCHEMA,
            "helper": file_pin(__file__),
            "validation_helper": file_pin(Path(__file__).with_name("modern_selected_sign.py")),
            "ancestry": _ancestry(actor),
            "base_state": self.base_state_pin,
            "math_sources": [
                file_pin(module.__file__)
                for module in (actor_source, native_step, recurrent_binary, recurrent_qat)
            ],
            "starting_scale_offset_hashes": {
                p: _tensor_hash(m.scale_offset) for p, m in actor.linears.items()
            },
            "selection": self.selection,
            "head_shape": list(self.head.latent_sign.shape),
            "source_order_scales": list(CANDIDATE_D_BASE_TO_PATH),
            "qk_scale_layout": copy.deepcopy(actor.qk_layout),
            "frozen_sign_hashes": copy.deepcopy(actor.initial_sign_hashes),
            "frozen_scale_hashes": copy.deepcopy(actor.initial_scale_hashes),
            "learn_signs": learn_signs,
            "packing": "source-order rows; zero-positive little-endian I32; no head permutation",
        }
        # Validate all values before any actor flag or forward mutation.
        require(
            _finite(self.latent)
            and bool((self.latent.detach().abs() <= 1).all())
            and all(_finite(m.scale_offset) for m in actor.linears.values()),
            "invalid starting masters/scales",
        )
        for parameter in actor.drafter.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for module in actor.linears.values():
            module.scale_offset.requires_grad_(True)
        self.head.forward = MethodType(lambda module, value: self.forward(value), self.head)
        actor._selected_head_owner = self
        self._installed_forward = self.head.forward

    def _validate_selection(self):
        s = self.selection
        require(
            isinstance(s, dict)
            and s.get("schema") == SELECTOR_SCHEMA
            and s.get("FIT_only") is True
            and s.get("helper") == file_pin(__file__),
            "selector source/schema/FIT contract differs",
        )
        require(
            type(s.get("count")) is int
            and isinstance(s.get("row_ids"), list)
            and len(s["row_ids"]) == s["count"] > 0
            and all(type(r) is int and 0 <= r < self.head.out_features for r in s["row_ids"])
            and len(set(s["row_ids"])) == len(s["row_ids"]),
            "invalid ordered selected head row IDs",
        )
        require(
            tuple(self.head.latent_sign.shape) != (50000, 2560) or s["count"] == 512,
            "modern production head requires exactly512 selected rows",
        )
        absolute = absolute_map(self.actor).cpu()
        require(
            s.get("map_sha256") == _tensor_hash(absolute)
            and s.get("target_ids") == absolute[s["row_ids"]].tolist(),
            "selected source-row/target mapping differs",
        )
        require(
            s.get("rows")
            and [r["row_id"] for r in s["rows"]] == s["row_ids"]
            and [r["target_id"] for r in s["rows"]] == s["target_ids"]
            and all(
                type(r["frequency"]) is int
                and r["frequency"] > 0
                and type(r["unique_groups"]) is int
                and 0 < r["unique_groups"] <= r["frequency"]
                for r in s["rows"]
            ),
            "selector row evidence differs",
        )
        require(s.get("source_pins"), "selector input ancestry required")
        for ref in s["source_pins"]:
            checked_pin(ref)

    def _identities(self):
        return tuple(
            (
                path,
                id(module),
                id(module.latent_sign),
                id(module.scale_offset),
                id(module.initial_scale),
            )
            for path, module in self.actor.linears.items()
        )

    def _versions(self):
        return (
            self.latent._version,
            self.rows._version,
            self.head.latent_sign._version,
            *(m.scale_offset._version for m in self.actor.linears.values()),
        )

    def check(self, *, integrity=False, bounded=True):
        _check_structure(self.actor)
        require(
            self._identities() == self._module_identities, "projection/parameter identities changed"
        )
        require(
            getattr(self.actor, "_selected_head_owner", None) is self
            and self.head.forward is self._installed_forward
            and not self.actor.scale_only
            and self.latent.requires_grad is self.learn_signs
            and self.latent.dtype == torch.float32
            and self.latent.is_contiguous()
            and tuple(self.latent.shape) == (len(self.selection["row_ids"]), self.head.in_features)
            and self.latent.device == self.head.latent_sign.device
            and self.rows.dtype == torch.int64
            and self.rows.device == self.latent.device
            and self.rows.tolist() == self.selection["row_ids"],
            "selected-head ownership/geometry changed",
        )
        expected = {id(m.scale_offset) for m in self.actor.linears.values()}
        require(
            {id(p) for p in self.actor.drafter.parameters() if p.requires_grad} == expected
            and id(self.latent) not in {id(p) for p in self.actor.drafter.parameters()},
            "optimizer actor scope differs; selected masters must stay external",
        )
        require(
            _finite(self.latent)
            and all(_finite(m.scale_offset) for m in self.actor.linears.values()),
            "nonfinite selected master/scale",
        )
        if bounded:
            require(bool((self.latent.detach().abs() <= 1).all()), "selected masters exceed [-1,1]")
        if integrity:
            require(self._cache is None, "state/export forbidden during active round")
            self._validate_selection()
            checked_pin(self.base_state_pin)
            require(
                file_pin(self.actor.initializer["path"]) == self.actor.initializer
                and file_pin(self.actor.config["path"]) == self.actor.config,
                "initializer/config pin changed",
            )
            for ref in self._source_contract["math_sources"]:
                checked_pin(ref)
            _check_frozen(self.actor, include_selected_initial=True)
            require(
                self._source_contract["helper"] == file_pin(__file__)
                and self._source_contract["validation_helper"]
                == file_pin(Path(__file__).with_name("modern_selected_sign.py"))
                and self._source_contract["ancestry"] == _ancestry(self.actor),
                "helper/base source ancestry changed",
            )

    def parameter_groups(self, *, sign_lr=None, scale_lr=None):
        self.check()
        groups = []
        if self.learn_signs:
            groups.append({"name": "selected_head_rows", "params": (self.latent,)})
        groups.append(
            {
                "name": "nine_raw_scale_offsets",
                "params": tuple(
                    self.actor.linears[p].scale_offset for p in CANDIDATE_D_BASE_TO_PATH.values()
                ),
            }
        )
        for group in groups:
            lr = sign_lr if group["name"] == "selected_head_rows" else scale_lr
            if lr is not None:
                require(
                    type(lr) in (int, float) and math.isfinite(lr) and lr > 0,
                    "positive finite LR required",
                )
                group["lr"] = float(lr)
        return groups

    def _new_cache(self):
        require(
            self.head._round_hard_signs is None, "head must be excluded from ordinary shared signs"
        )
        selected = hard_sign_ste(self.latent)
        with torch.no_grad():
            signs = hard_sign_ste(self.head.latent_sign)
            signs.index_copy_(0, self.rows, selected.detach())
        token = {"owner": self, "versions": self._versions(), "valid": True, "pending": 0}
        return signs, selected, token

    @contextmanager
    def round(self):
        self.check()
        require(self._cache is None, "nested or stale selected-head round")
        other = {p: m for p, m in self.actor.linears.items() if p != HEAD_PATH}
        require(
            all(m._round_hard_signs is None for m in self.actor.linears.values()),
            "nested ordinary shared signs",
        )
        cache = self._new_cache()
        self._cache = cache
        clean = False
        try:
            with shared_round_hard_signs(other):
                yield
            clean = True
        finally:
            self._cache = None
            cache[2]["valid"] = False
            if clean:
                require(cache[2]["pending"] == 0, "consume backward inside selected-head round")

    def forward(self, value):
        require(
            self.actor.linears[HEAD_PATH] is self.head
            and not self.head.latent_sign.requires_grad
            and self.latent.requires_grad is self.learn_signs,
            "selected-head forward ownership changed",
        )
        require(
            value.shape[-1] == self.head.in_features and value.device == self.latent.device,
            "input/head geometry or device differs",
        )
        require(self.head._round_hard_signs is None, "head cannot use ordinary shared signs")
        signs, selected, token = self._cache if self._cache is not None else self._new_cache()
        require(
            not (torch.is_grad_enabled() and self.learn_signs) or selected.requires_grad,
            "selected-head cache built without gradient tracking",
        )
        require(
            token["valid"] and token["versions"] == self._versions(),
            "update within cached selected-head round",
        )
        quantized, _, saturated = hard_activation(value, 8)
        self.head.last_saturation_fraction = saturated.float().mean().detach()
        tracked = torch.is_grad_enabled() and (quantized.requires_grad or selected.requires_grad)
        if tracked:
            token["pending"] += 1
        projected = _Projection.apply(quantized, signs, selected, self.rows, token)
        return projected * self.head.effective_scales() + 0

    def clamp_(self):
        self.check(bounded=False)
        require(self._cache is None, "cannot clamp inside active round")
        with torch.no_grad():
            self.latent.clamp_(-1, 1)

    def state(self):
        """Compact model state only; outer recipe must checkpoint Adam/RNG/cursor."""
        self.check(integrity=True)
        return {
            "schema": SCHEMA,
            "contract": copy.deepcopy(self._source_contract),
            "selected_latent": self.latent.detach().cpu().clone(),
            "scale_offsets": {
                base: self.actor.linears[path].scale_offset.detach().cpu().clone()
                for base, path in CANDIDATE_D_BASE_TO_PATH.items()
            },
        }

    def load_state(self, state):
        """Validate all payloads/ancestry and transfer all sources before mutation."""
        self.check(integrity=True)
        require(
            isinstance(state, dict)
            and set(state) == {"schema", "contract", "selected_latent", "scale_offsets"}
            and state["schema"] == SCHEMA
            and state["contract"] == self._source_contract,
            "sparse state schema/source/selection/base ancestry differs",
        )
        require(
            isinstance(state["scale_offsets"], dict)
            and set(state["scale_offsets"]) == set(CANDIDATE_D_BASE_TO_PATH),
            "sparse scale mask differs",
        )
        pending = [(self.latent, state["selected_latent"])] + [
            (self.actor.linears[path].scale_offset, state["scale_offsets"][base])
            for base, path in CANDIDATE_D_BASE_TO_PATH.items()
        ]
        for destination, value in pending:
            require(
                isinstance(value, torch.Tensor)
                and value.device.type == "cpu"
                and value.dtype == torch.float32
                and value.shape == destination.shape
                and value.is_contiguous()
                and not value.requires_grad
                and _finite(value),
                "invalid sparse tensor shape/dtype/finite/layout",
            )
        require(bool((state["selected_latent"].abs() <= 1).all()), "sparse masters exceed [-1,1]")
        prepared = [
            (destination, value.to(destination.device).clone()) for destination, value in pending
        ]
        with torch.no_grad():
            for destination, value in prepared:
                destination.copy_(value)

    def budget(self):
        self.check()
        signs = self.latent.numel()
        scales = sum(m.scale_offset.numel() for m in self.actor.linears.values())
        return {
            "selected_sign_parameters": signs,
            "scale_parameters": scales,
            "model_F32_bytes": 4 * (signs + scales),
            "model_plus_Adam_F32_bytes": 12 * (signs + scales),
            "additional_gradient_bytes": 4 * (signs + scales),
            "one_full_detached_sign_cache_bytes": 4 * self.head.latent_sign.numel(),
            "selected_packed_patch_bytes": len(self.rows)
            * math.ceil(self.head.in_features / 32)
            * 4,
            "estimated_not_measured": True,
        }

    def export(self, destination):
        """Patch selected native head rows and nine scales; audit all other bytes."""
        from gguf import GGMLQuantizationType as Type
        from gguf import GGUFReader

        self.check(integrity=True)
        actor = self.actor
        source = Path(actor.initializer["path"])
        require(file_pin(source) == actor.initializer, "initializer changed")
        reader = GGUFReader(str(source), mode="r")
        tensors = {t.name: t for t in reader.tensors}
        require(len(tensors) == len(reader.tensors), "duplicate native tensor names")
        pending = []
        for base, path in CANDIDATE_D_BASE_TO_PATH.items():
            values = (
                actor.linears[path]
                .effective_scales()
                .detach()
                .cpu()
                .numpy()
                .astype("<f4", copy=True)
            )
            require(np.isfinite(values).all() and (values >= 0).all(), "invalid effective scales")
            if base in actor.qk_layout:
                values = values[qk_row_indices(**actor.qk_layout[base])].copy()
            tensor = tensors[base + ".w1a1_scale"]
            require(
                tensor.tensor_type == Type.F32
                and tuple(tensor.data.shape) == values.shape
                and tensor.n_bytes == values.nbytes,
                "native scale shape/type differs",
            )
            pending.append((base + ".w1a1_scale", int(tensor.data_offset), values.tobytes()))
        packed = tensors["output.w1a1_packed"]
        words = math.ceil(self.head.in_features / 32)
        require(
            packed.tensor_type == Type.I32
            and tuple(packed.data.shape) == (self.head.out_features, words)
            and packed.n_bytes == self.head.out_features * words * 4,
            "native head packed shape/type differs",
        )
        # Bounded rows, including tail-bit validation; never unpack the whole head.
        masters = self.latent.detach().cpu().numpy()
        for index, row in enumerate(self.selection["row_ids"]):
            unpack_signs(packed.data[row : row + 1], self.head.in_features)
            bits = pack_signs(
                np.where(masters[index : index + 1] < 0, np.float32(-1), np.float32(1))
            )
            pending.append(
                (
                    f"output.w1a1_packed[{row}]",
                    int(packed.data_offset) + row * words * 4,
                    bits.tobytes(),
                )
            )
        return _publish_patches(
            source, actor.initializer, destination, pending, self._source_contract
        )


def _publish_patches(source, initializer, destination, pending, contract):
    """Validate declared nonoverlapping ranges, publish exclusively, audit gaps."""
    destination = Path(destination).resolve()
    source = Path(source)
    require(
        destination != source.resolve() and not destination.exists(),
        "export cannot overwrite existing file",
    )
    ranges = sorted(pending, key=lambda p: p[1])
    cursor = 0
    for _, offset, payload in ranges:
        require(
            type(offset) is int
            and isinstance(payload, bytes)
            and payload
            and cursor <= offset
            and offset + len(payload) <= initializer["bytes"],
            "invalid/overlapping native patch ranges",
        )
        cursor = offset + len(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + destination.name + ".", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w+b") as output, source.open("rb") as original:
            shutil.copyfileobj(original, output, length=1024**2)
            for _, offset, payload in ranges:
                output.seek(offset)
                output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        cursor = 0
        with source.open("rb") as original, temporary.open("rb") as output:
            for _, offset, payload in ranges:
                remaining = offset - cursor
                while remaining:
                    length = min(1024**2, remaining)
                    require(
                        original.read(length) == output.read(length),
                        "undeclared native export bytes changed",
                    )
                    remaining -= length
                original.seek(len(payload), 1)
                require(output.read(len(payload)) == payload, "native row/scale patch differs")
                cursor = offset + len(payload)
            while chunk := original.read(1024**2):
                require(chunk == output.read(len(chunk)), "undeclared native export tail changed")
            require(output.read(1) == b"", "native export size changed")
        require(file_pin(source) == initializer, "initializer changed during export")
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": SCHEMA,
        "contract": copy.deepcopy(contract),
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

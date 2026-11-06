#!/usr/bin/env python3
"""Portable checkpoint/operator helpers; run_parent.py owns the authentic pilot CLI."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os

import numpy as np
import torch
from shortlist_pilots.binary.adapters import DoubleBinary, MixtureScales, TernaryA8, install
from w1a1_eagle.block_training import _block_rng_state
from w1a1_eagle.continuous_qat import atomic_json, sha256
from w1a1_eagle.dspark_aligned13 import digest

SCHEMA = "shortlist_binary_fullmodel_v1"


def validate_limits(seconds, updates):
    if not 0 < seconds <= 840 or type(updates) is not int or updates < 1:
        raise ValueError("require 0 < train seconds <= 840 and positive updates")


def save_checkpoint(model, optimizer, cursor, source, intervention, path):
    """Portable research checkpoint; deliberately not a block_qat exact-resume receipt."""
    path = Path(path)
    binary_ids = {id(p) for m in model.binary_linears().values() for p in m.parameters()}
    # Live detached tensor references stream one tensor at a time through
    # torch.save; avoid an extra four-GB CPU copy of full48/Adam state.
    def detached(value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {k: detached(v) for k, v in value.items()}
        if isinstance(value, list):
            return [detached(v) for v in value]
        return value
    payload = {"schema": SCHEMA, "source": source, "intervention": intervention,
               "config": asdict(model.config), "cursor": asdict(cursor),
               "linears": {n: detached(m.state_dict()) for n, m in model.binary_linears().items()},
               "nonlinear_parameters": {n: p.detach() for n, p in model.named_parameters()
                                        if p.requires_grad and id(p) not in binary_ids},
               "optimizer": detached(optimizer.state_dict()),
               "rng": _block_rng_state(model.token_embd.device)}
    temporary = path.with_suffix(".tmp")
    if path.exists():
        raise ValueError("refuse checkpoint overwrite")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)
    receipt = {"schema": SCHEMA, "path": str(path.resolve()), "sha256": sha256(path),
               "source_sha256": digest(source), "cursor": asdict(cursor),
               "resume_contract": "portable state, not automatic exact resume"}
    atomic_json(path.with_suffix(".receipt.json"), receipt)
    return receipt


def export_operators(model, path):
    """Pack true signs; preserve F32 metadata. Payload is NOT a GGUF/native codec."""
    arrays, descriptions = {}, {}
    for name, module in model.binary_linears().items():
        key = name.replace(".", "_")
        latent = module.latent_sign.detach().cpu().numpy()
        arrays[key + "__signs"] = np.packbits(latent >= 0, axis=-1, bitorder="little")
        arrays[key + "__scales"] = module.effective_scales().detach().cpu().numpy()
        description = {"input": module.in_features, "output": module.out_features,
                       "sign_shape": list(latent.shape), "bitorder": "little",
                       "positive_zero": True, "type": "native-row-binary"}
        if isinstance(module, MixtureScales):
            for field in ("router", "input_experts", "output_expert_offsets"):
                arrays[key + "__" + field] = getattr(module, field).detach().cpu().numpy()
            description.update(type="mixture-scales-static" if module.static else "mixture-scales",
                               cost=module.cost())
        elif isinstance(module, DoubleBinary):
            out = module.factor_out.detach().cpu().numpy()
            arrays[key + "__factor_out"] = np.packbits(out >= 0, axis=-1, bitorder="little")
            arrays[key + "__input_scale"] = module.input_scale.detach().cpu().numpy()
            description.update(type="double-binary", factor_out_shape=list(out.shape), cost=module.cost())
        elif isinstance(module, TernaryA8):
            codes = module.codes().detach().cpu().numpy().astype(np.int8)
            # Four 2-bit values per byte: 00=0, 01=+1, 10=-1; 11 invalid.
            encoded = np.where(codes < 0, 2, codes).astype(np.uint8)
            width = encoded.shape[-1]
            padded = np.pad(encoded, ((0, 0), (0, (-width) % 4)))
            packed = sum(padded[:, offset::4] << (2*offset) for offset in range(4))
            del arrays[key + "__signs"]
            arrays[key + "__ternary2"] = packed
            description.update(type="ternary-a8", cost=module.cost(),
                               encoding="00=0,01=+1,10=-1,11=invalid; low pair first")
        descriptions[name] = description
    path = Path(path)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    receipt = {"schema": "shortlist_binary_operator_payload_v1", "sha256": sha256(path),
               "native_gguf": False, "operators": descriptions,
               "frozen_exceptions": "source model/checkpoint dependencies retained; no target changes"}
    atomic_json(path.with_suffix(".json"), receipt)
    return receipt


def restore_portable(model, path, expected_sha256):
    """Apply a research payload to an independently restored exact source parent.

    Returns the envelope for an explicit optimizer/RNG continuation adapter.
    This function restores model values only and does not claim exact resume.
    """
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError("research checkpoint digest differs")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("config") != asdict(model.config):
        raise ValueError("research checkpoint schema/model config differs")
    intervention = payload["intervention"]
    install(model, intervention["technique"], names=intervention.get("projections"),
            rank=intervention.get("rank", 32), sweeps=0)
    if set(payload["linears"]) != set(model.binary_linears()):
        raise ValueError("research checkpoint binary inventory differs")
    for name, module in model.binary_linears().items():
        saved, live = payload["linears"][name], module.state_dict()
        if set(saved) != set(live) or any(saved[n].shape != live[n].shape
                or saved[n].dtype != live[n].dtype or not bool(torch.isfinite(saved[n]).all())
                for n in live):
            raise ValueError("research checkpoint operator schema differs: " + name)
    named = dict(model.named_parameters())
    for name, value in payload["nonlinear_parameters"].items():
        if (name not in named or value.shape != named[name].shape
                or value.dtype != named[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError("research checkpoint norm schema differs")
    for name, module in model.binary_linears().items():
        module.load_state_dict(payload["linears"][name], strict=True)
    with torch.no_grad():
        for name, value in payload["nonlinear_parameters"].items():
            named[name].copy_(value)
    return payload


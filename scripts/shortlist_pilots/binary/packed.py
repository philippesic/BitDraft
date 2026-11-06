"""Executable reference consumer for exported binary operator payloads.

This consumes real packed bits and F32 metadata. It is a PyTorch reference,
not an installed llama.cpp operator or a CUDA performance implementation.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from w1a1_eagle.recurrent_qat import fixed_activation_codes
from w1a1_eagle.continuous_qat import sha256


class PackedOperator(nn.Module):
    def __init__(self, description, arrays):
        super().__init__()
        self.description = description
        for name, value in arrays.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach().float().clone()
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("finite packed-operator metadata required")
                self.register_buffer(name, tensor)
                continue
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError("finite packed-operator metadata required")
            self.register_buffer(name, torch.from_numpy(array.copy()))

    def forward(self, inputs):
        x = inputs.float()
        if x.shape[-1] != self.description["input"]:
            raise ValueError("packed operator input width differs")
        kind = self.description["type"]
        outscale = self.scales
        if kind.startswith("mixture-scales"):
            gate = (self.router.mean(0).expand(*x.shape[:-1], 2)
                    if kind == "mixture-scales-static" else x @ self.router).softmax(-1)
            x = x * (gate @ self.input_experts)
            outscale = outscale * (1+gate @ self.output_expert_offsets)
        if kind == "double-binary":
            x = x*self.input_scale
        codes, beta, _ = fixed_activation_codes(x, 8)
        if self.description.get("activation_byte_roundtrip", False):
            codes = codes.to(torch.int8).float()
        if kind == "double-binary":
            intermediate = F.linear(codes, self.signs)*beta
            return F.linear(intermediate, self.factor_out)*outscale
        return (F.linear(codes, self.signs)*outscale)*beta


def load_operator(path, projection, expected_sha256):
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (sha256(path) != expected_sha256 or receipt.get("sha256") != expected_sha256
            or receipt.get("schema") != "shortlist_binary_operator_payload_v1"
            or receipt.get("native_gguf") is not False):
        raise ValueError("authenticated portable operator payload required")
    description = receipt["operators"][projection]
    kind, key = description["type"], projection.replace(".", "_") + "__"
    if kind not in {"native-row-binary", "mixture-scales", "mixture-scales-static", "double-binary", "ternary-a8"}:
        raise ValueError("unknown packed operator type")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name[len(key):]: archive[name] for name in archive.files if name.startswith(key)}
    if kind == "ternary-a8":
        packed = arrays.pop("ternary2")
        encoded = np.stack([(packed >> (2*i)) & 3 for i in range(4)], axis=-1)
        encoded = encoded.reshape(description["output"], -1)[:, :description["input"]].astype(np.int8)
        if (encoded == 3).any():
            raise ValueError("invalid ternary code")
        arrays["signs"] = np.where(encoded == 2, -1, encoded)
    else:
        shape = description["sign_shape"]
        arrays["signs"] = np.unpackbits(arrays["signs"], axis=-1, bitorder="little")[:, :shape[1]].astype(np.int8)*2-1
    if kind == "double-binary":
        shape = description["factor_out_shape"]
        arrays["factor_out"] = np.unpackbits(arrays["factor_out"], axis=-1, bitorder="little")[:, :shape[1]].astype(np.int8)*2-1
    return PackedOperator(description, arrays)

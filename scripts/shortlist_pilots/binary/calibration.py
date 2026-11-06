"""Native-A8-aware projection reconstruction, preserving the single-plane codec.

Only the input metric is transplanted from ARB-X; no second plane/midpoint is
added. Real TRAIN inputs come from the exact-parent full drafter. These are the
training actor's integer codes, not a claim of serving-feature equivalence.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from w1a1_eagle.recurrent_qat import fixed_activation_codes
from w1a1_eagle.recurrent_binary import hard_sign_ste


def reconstruction_loss(module, inputs, reference_weight, metric):
    """Equal normalized SSE units; teacher uses the same immutable A8 codes."""
    if reference_weight.shape != module.latent_sign.shape or metric not in {"a8", "weight-mse"}:
        raise ValueError("matching dense projection reference and metric required")
    if metric == "weight-mse":
        candidate = hard_sign_ste(module.latent_sign) * module.effective_scales()[:, None]
        target = reference_weight.detach()
    else:
        # Detach activation values to fit only this projection. Upstream tensors
        # and the entire teacher are frozen; no teacher gradient leaks through.
        inputs = inputs.detach().float()
        with torch.no_grad():
            codes, beta, _ = fixed_activation_codes(inputs, 8)
            target = F.linear(codes, reference_weight.detach()) * beta
        binary = hard_sign_ste(module.latent_sign)
        candidate = (F.linear(codes, binary) * module.effective_scales()) * beta
    energy = target.square().sum().detach()
    if not bool(torch.isfinite(energy)) or float(energy) <= 0:
        raise ValueError("finite positive target reconstruction energy required")
    loss = (candidate-target).square().sum() / energy
    if not bool(torch.isfinite(loss)):
        raise ValueError("finite reconstruction loss required")
    return loss


def load_reference(path, expected_sha256, projection, shape):
    """Read one authenticated original floating source tensor, no model-wide copy."""
    from shortlist_pilots.objectives.runtime import sha
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
    path = Path(path)
    if sha(path) != expected_sha256:
        raise ValueError("reference source digest differs")
    reader = GGUFReader(path)
    candidates = [t for t in reader.tensors if t.name == projection + ".weight"]
    if len(candidates) != 1:
        raise ValueError("exactly one source projection required")
    tensor = candidates[0]
    if tensor.tensor_type not in {GGMLQuantizationType.BF16, GGMLQuantizationType.F16,
                                  GGMLQuantizationType.F32}:
        raise ValueError("calibration reference must retain original floating source")
    result = torch.from_numpy(np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32).copy())
    if tuple(result.shape) != tuple(shape) or not bool(torch.isfinite(result).all()):
        raise ValueError("reference projection geometry/values differ")
    return result


def capture_inputs(model, batches, projection):
    """Authentic actor inputs at one projection; bounded to the current pair."""
    module = model.binary_linears()[projection]
    captured = []
    handle = module.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            for batch in batches:
                model(batch)
    finally:
        handle.remove()
    if len(captured) != len(batches):
        raise ValueError("one selected-projection input per block required")
    return captured

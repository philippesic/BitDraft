"""Actual-shape CUDA operator timer; reference arithmetic, not a native backend.

Resident decoded F32 signs exclude weight-unpack/storage-kernel costs. Timed
forwards include input scaling, router/mixtures, whole-row A8 packing, factor
intermediate, matmuls and output scaling. CPU fixtures test contracts only.
"""
import statistics
import time
import torch
from shortlist_pilots.binary.adapters import DoubleBinary, MixtureScales, TernaryA8
from shortlist_pilots.binary.packed import PackedOperator


@torch.no_grad()
def inference_reference(module):
    description = {"input": module.in_features, "output": module.out_features, "type": "native-row-binary",
                   "activation_byte_roundtrip": True}
    arrays = {"signs": torch.where(module.latent_sign < 0, -1., 1.),
              "scales": module.effective_scales()}
    if isinstance(module, MixtureScales):
        description["type"] = "mixture-scales-static" if module.static else "mixture-scales"
        for name in ("router", "input_experts", "output_expert_offsets"):
            arrays[name] = getattr(module, name)
    elif isinstance(module, DoubleBinary):
        description["type"] = "double-binary"
        arrays["factor_out"] = torch.where(module.factor_out < 0, -1., 1.)
        arrays["input_scale"] = module.input_scale
    elif isinstance(module, TernaryA8):
        description["type"] = "ternary-a8"
        arrays["signs"] = module.codes()
    return PackedOperator(description, arrays)


def timing_summary(milliseconds, wall_milliseconds):
    return {"cuda_ms_median": statistics.median(milliseconds), "cuda_ms_min": min(milliseconds),
            "cuda_ms_max": max(milliseconds), "wall_ms_median": statistics.median(wall_milliseconds)}


@torch.no_grad()
def benchmark_pair(parent, candidate, inputs, *, warmups=3, samples=5, repeats=10):
    if (inputs.device.type != "cuda" or inputs.ndim != 2 or inputs.shape[0] != 7
            or parent.in_features != candidate.in_features
            or parent.out_features != candidate.out_features):
        raise ValueError("matched actual seven-row CUDA projection required")
    if not 0 <= warmups <= 20 or not 1 <= samples <= 20 or not 1 <= repeats <= 200:
        raise ValueError("bounded component timing schedule required")
    clock = time.monotonic()
    operators = {"parent": inference_reference(parent), "candidate": inference_reference(candidate)}
    torch.cuda.synchronize(inputs.device)
    preparation_seconds = time.monotonic()-clock
    parity = {name: bool(torch.equal(operator(inputs), module(inputs)))
              for (name, operator), module in zip(operators.items(), (parent, candidate))}
    if not all(parity.values()):
        raise ValueError("timed inference reference differs from training operator forward")
    timings = {name: ([], []) for name in operators}
    for operator in operators.values():
        for _ in range(warmups):
            operator(inputs)
    torch.cuda.synchronize(inputs.device)
    # Alternate order to avoid giving one operator every first measurement.
    for sample in range(samples):
        names = list(operators) if sample % 2 == 0 else list(reversed(operators))
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.monotonic()
            start.record()
            for _ in range(repeats):
                operators[name](inputs)
            end.record()
            end.synchronize()
            timings[name][0].append(start.elapsed_time(end)/repeats)
            timings[name][1].append((time.monotonic()-wall_start)*1000/repeats)
    reports = {name: timing_summary(*values) for name, values in timings.items()}
    return {"schema": "shortlist_binary_component_reference_v1", "input_shape": list(inputs.shape),
            "output_width": parent.out_features, "warmups": warmups, "samples": samples, "repeats": repeats,
            "timings": reports, "candidate_over_parent_cuda_time": reports["candidate"]["cuda_ms_median"]/reports["parent"]["cuda_ms_median"],
            "reference_forward_exact": parity, "one_time_preparation_seconds": preparation_seconds,
            "included": ["routing", "scale mixing", "input/output scaling", "A8 code generation",
                         "row-major INT8 materialization and F32 decode", "factor intermediate", "F32 matmuls"],
            "excluded": ["packed weight reads/unpacking", "native byte layout/codec/kernel dispatch", "full drafter/verifier/request"],
            "scope": "actual-shape PyTorch CUDA F32 reference; no native/backend/full-request speed claim"}

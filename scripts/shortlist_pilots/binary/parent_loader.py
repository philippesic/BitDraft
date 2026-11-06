"""Restore current full48 parent without materializing obsolete GGUF variants.

Retained step1000 full state supplies the original sign guard only. The current
parent GGUF supplies every ordinary frozen payload and the export donor. Before
any training, zero-update export must be byte-identical to that authenticated
current GGUF. This does not establish native ordinary-arithmetic equivalence.
"""
from dataclasses import asdict
import gc
import hashlib
from pathlib import Path


def equal(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def copy_named_state(model, saved, parameter_names, buffer_names):
    """Validate the complete declared transition before copying any values."""
    import torch
    live_groups = {"parameters": parameter_names(model), "binary_buffers": buffer_names(model)}
    for key, live in live_groups.items():
        if set(live) != set(saved[key]):
            raise ValueError("parent state inventory differs: " + key)
        for name, tensor in live.items():
            source = saved[key][name]
            if (source.shape != tensor.shape or source.dtype != tensor.dtype
                    or not bool(torch.isfinite(source).all())):
                raise ValueError("parent tensor schema differs: " + name)
    with torch.no_grad():
        for key, live in live_groups.items():
            for name, tensor in live.items():
                tensor.copy_(saved[key][name])
                if not torch.equal(tensor.detach().cpu(), saved[key][name]):
                    raise ValueError("parent tensor copy differs: " + name)


def sign_hashes(model, parameter_names):
    return {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for name, value in parameter_names(model).items() if name.endswith(".latent_sign")}


def load_full_model(setup, parent, *, source_checkpoint=None, source_gguf=None):
    """Same return contract as objectives.runtime.load_full_model.

    Optional paths relocate authenticated bytes; they never replace expected
    hashes or relax the original sign/norm/Adam/RNG/config/data guards.
    """
    from shortlist_pilots.objectives import runtime
    runtime.configure_imports()
    from source_gate import verify_training_source
    source_validation = verify_training_source(runtime.CAMPAIGN)
    import torch
    import gguf
    from w1a1_eagle.block_qat import BlockDrafter, BlockQATConfig
    from w1a1_eagle.block_training import load_block_gguf, _restore_block_rng, _block_rng_state
    from joint_tools import joint_parameters, binary_buffers, install_joint_parameters, joint_optimizer, assert_joint_policy, export_joint
    from native_frozen_actor import install_native_frozen
    from device_sign_baseline import DeviceSignBaseline
    from train_nine_model_qat import configure_cuda
    from common import guard

    setup = Path(setup)
    for key in ("checkpoint", "model"):
        if runtime.sha(parent[key]["path"]) != parent[key]["sha256"]:
            raise ValueError("authenticated current parent changed: " + key)
    saved = torch.load(parent["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    if saved["step"] != 3504 or len(saved["parameters"]) != 48 or len(saved["binary_buffers"]) != 13:
        raise ValueError("exact full48 parent3504 required")
    config = BlockQATConfig(**saved["config"])
    if (config.family != "dspark" or config.activation_bits != 8
            or config.profile != "ffn13_down14_stock" or config.conditioning != "captured_prefix"
            or config.block_size != 7 or (config.sign_lr, config.scale_lr) != (3e-5, 1e-5)):
        raise ValueError("current native-A8 full48 architecture/config differs")
    config_path = runtime.CAMPAIGN / "command-calibration-pilot01/joint1000.json"
    if runtime.sha(config_path) != "ce3dbc5084ac17172fc57a98b236ebfbb97c2ce660d901a937830fac5061f9b1":
        raise ValueError("original source config changed")
    source_model = dict(runtime.read(config_path)["model"])
    if source_gguf is not None:
        source_model["path"] = str(source_gguf)
    guard()
    tensors = load_block_gguf(source_model["path"], source_model["sha256"], config)
    model = BlockDrafter(tensors, config)
    del tensors
    gc.collect()
    guard()
    reader = gguf.GGUFReader(parent["model"]["path"])
    actor = install_native_frozen(model, reader)
    install_joint_parameters(model, {t.name: t for t in reader.tensors})
    if asdict(model.config) != saved["config"]:
        raise ValueError("parent model config changed during construction")
    original_pin = dict(runtime.read(runtime.CAMPAIGN / "sign-eal01/bindings.json")["parent_checkpoint"])
    if source_checkpoint is not None:
        original_pin["path"] = str(source_checkpoint)
    if runtime.sha(original_pin["path"]) != original_pin["sha256"]:
        raise ValueError("original sign-reference checkpoint changed")
    original = torch.load(original_pin["path"], map_location="cpu", mmap=True, weights_only=False)
    if (original["schema"] != "approx_hidden_KD_timed_continuation_v1"
            or original["state"]["cursor"]["step"] != 1000
            or original["state"]["config"] != saved["config"]
            or original["state"]["source"] != runtime.read(runtime.CAMPAIGN / "approx-kd20002/kd/start.json")["source"]):
        raise ValueError("original sign-reference source/config/step differs")
    copy_named_state(model, original["state"], joint_parameters, binary_buffers)
    if sign_hashes(model, joint_parameters) != saved["initial_sign_hashes"]:
        raise ValueError("original latent sign-reference hashes differ")
    baseline = DeviceSignBaseline(model)
    del original
    gc.collect()
    copy_named_state(model, saved, joint_parameters, binary_buffers)
    if sign_hashes(model, joint_parameters) != saved["final_sign_hashes"]:
        raise ValueError("current parent sign hashes differ")
    # Export directly against current donor, preserving every ordinary tensor.
    # The temporary file is disposable only after its exact digest is checked.
    trial = setup / "restored.gguf"
    exported = export_joint(model, parent["model"]["path"], parent["model"]["sha256"], trial)
    runtime.write(setup / "restore-export.json", exported)
    if exported["output"]["sha256"] != parent["model"]["sha256"]:
        raise ValueError("zero-update current parent native export differs")
    trial.unlink()
    hardware = configure_cuda(zero_updates=False)
    if hardware["compute_capability"] != [12, 0] or "5080" not in hardware["device_name"]:
        raise ValueError("requires admitted RTX5080 lane")
    model.to("cuda:0")
    baseline.bind_device(model)
    optimizer = joint_optimizer(model)
    optimizer.load_state_dict(saved["optimizer"])
    assert_joint_policy(model, optimizer)
    if (not equal(optimizer.state_dict(), saved["optimizer"])
            or any(int(s["step"]) != 3504 for s in optimizer.state.values())
            or baseline.measure(model) != saved["sign_displacement"]):
        raise ValueError("original Adam/sign-reference restoration differs")
    metadata = {key: saved[key] for key in ("rng", "initial_norms", "initial_sign_hashes", "TRAIN_seconds")}
    _restore_block_rng(saved["rng"], model.token_embd.device)
    if not equal(_block_rng_state(model.token_embd.device), saved["rng"]):
        raise ValueError("original RNG restore differs")
    del saved, reader
    gc.collect()
    guard()
    runtime.write(setup / "parent-loader.json", {
        "schema": "shortlist_current_parent_load_v1", "parent": parent,
        "original_sign_reference": original_pin, "source_model": source_model,
        "source_validation": source_validation, "hardware": hardware,
        "frozen_actor": actor, "zero_update_native_byteexact": True,
        "exact48_parameters13buffers_Adam_rng_config_original_sign_reference": True,
        "ordinary_arithmetic_fidelity": "historical failure retained; ordinary weights decoded into F32",
        "dependency_change": "no obsolete old_c0 or row1000 GGUF required"})
    model.train()
    return model, optimizer, baseline, metadata


def native_export(model, parent, output):
    """Current parent donor removes the obsolete intermediate GGUF dependency."""
    from joint_tools import export_joint
    return export_joint(model, parent["model"]["path"], parent["model"]["sha256"], output)

"""Sole-owner actual GPU selected-O QAT plumbing qualification, never generalization.

Two distinct FIT-window updates, plus a repeated second update, qualify compact
Adam/RNG persistence. A separately labelled forced one-bit export tests packing.
All outputs are unique on C:. Run only after native timing has released the GPU.
"""

from __future__ import annotations

# Explicit source bootstrap before optional Torch/project imports.
# ruff: noqa: E402
import argparse
import gc
import json
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path("/home/philip/binary-eagle-decoding")
SOURCE = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts")
MANIFEST = PROJECT / "runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json"
MANIFEST_SHA = "ccc5fb9e89c86daece91e9b29327a285a71c3f35362e9efd60c40b4a92d8f5aa"
INITIALIZER = PROJECT / "runs/w1a8-modern-packed-init-20261103-01/model.gguf"
CONFIG = PROJECT / "runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json"
EXISTING_CACHE = PROJECT / "runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda"
FLOOR = 2 * 1024**3 + 64 * 1024**2
OUTPUT_RESERVE = 320 * 1024**2
sys.path[:0] = [str(SOURCE / "src"), str(PROJECT / "third_party/llama.cpp/gguf-py")]

import modern_actor
import modern_actual_data
import modern_selected_sign
import torch
from modern_actor import build_modern_actor, file_pin
from modern_actual_data import ActualDataset
from modern_selected_sign import (
    SELECTED_BASES,
    SELECTED_PATH,
    clamp_selected_latents_,
    collect_selected_state,
    enable_selected_sign,
    export_selected_sign_gguf,
    load_selected_state,
    selected_parameter_groups,
    selected_sign_metrics,
    selected_state_budget,
)
from modern_train import cpu
from modern_train_protected import forward_semantics

from w1a1_eagle import native_step, recurrent_binary, recurrent_qat
from w1a1_eagle.native_step import NATIVE_NORM_PATHS, ExternalNativeF32Embedding
from w1a1_eagle.recurrent_qat import shared_round_hard_signs


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    """Unique fsynced metadata; never replace existing evidence."""
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(encoded) < 2 * 1024**2, "unexpected metadata size")
    with Path(path).open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def bit_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(bit_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(bit_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def forced_export_audit(learned, forced):
    """Stronger than each export's initializer audit: only ONE O bit may change."""
    name = SELECTED_BASES[0] + ".w1a1_packed"
    first = {p["tensor"]: p for p in learned["replaced_payloads"]}
    second = {p["tensor"]: p for p in forced["replaced_payloads"]}
    require(set(first) == set(second) and len(first) == 10, "native payload contracts differ")
    for key in first:
        require(
            first[key]["offset"] == second[key]["offset"]
            and first[key]["bytes"] == second[key]["bytes"],
            "native payload ranges moved",
        )
        if key != name:
            require(first[key] == second[key], "forced diagnostic changed a scale payload")
    offset, size = first[name]["offset"], first[name]["bytes"]
    flips = 0
    with (
        Path(learned["output"]["path"]).open("rb") as a,
        Path(forced["output"]["path"]).open("rb") as b,
    ):
        remaining = offset
        while remaining:
            count = min(1024 * 1024, remaining)
            require(a.read(count) == b.read(count), "forced diagnostic changed non-O bytes")
            remaining -= count
        original, changed = a.read(size), b.read(size)
        require(len(original) == len(changed) == size, "short selected O payload")
        flips = sum((x ^ y).bit_count() for x, y in zip(original, changed, strict=True))
        while chunk := a.read(1024 * 1024):
            require(chunk == b.read(len(chunk)), "forced diagnostic changed non-O tail")
        require(b.read(1) == b"", "forced diagnostic changed model size")
    require(flips == 1, "forced diagnostic must change exactly one packed bit")
    return {
        "changed_O_packed_bits": flips,
        "all_nine_scale_payloads_identical": True,
        "all_bytes_outside_O_packed_identical": True,
        "O_packed_offset": offset,
        "O_packed_bytes": size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    require(args.allow_cuda, "explicit sole GPU owner --allow-cuda required")
    args.out = args.out.resolve()
    require(
        args.out.is_relative_to(ARTIFACTS.resolve()) and args.out != ARTIFACTS.resolve(),
        "unique C: artifact output required",
    )
    require(not args.out.exists(), "refusing to replace existing qualification")
    require(
        args.manifest.resolve() == MANIFEST.resolve(),
        "only current actual-serving manifest is admitted",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and os.environ.get("CUDA_CACHE_PATH")
        and Path(os.environ["CUDA_CACHE_PATH"]).resolve() == EXISTING_CACHE.resolve()
        and EXISTING_CACHE.is_dir()
        and os.environ.get("CUDA_CACHE_MAXSIZE") == "1073741824"
        and not os.environ.get("LD_PRELOAD"),
        "explicit existing training cache/deterministic CUDA environment required",
    )
    stopping = False
    source_refs = []

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def guard(*, reserve=0):
        control = json.loads(
            (Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text()
        )["rtx5080"]
        require(
            not stopping
            and not control.get("pause_requested")
            and not control.get("stop_all_requested")
            and not (args.out / "STOP").exists(),
            "GPU stop requested",
        )
        for filesystem in (PROJECT, Path("/mnt/c")):
            info = os.statvfs(filesystem)
            needed = FLOOR + (reserve if filesystem == Path("/mnt/c") else 0)
            require(
                info.f_bavail * info.f_frsize >= needed,
                "storage floor/reserve reached: " + str(filesystem),
            )
        for ref in source_refs:
            require(file_pin(ref["path"]) == ref, "qualification source changed: " + ref["path"])

    guard(reserve=OUTPUT_RESERVE)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    costs = {}
    checkpoints = []
    phase = "initialization"
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(37)
        torch.cuda.manual_seed_all(37)
        random.seed(37)
        require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5080", "RTX5080 GPU0 required")
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        hardware = subprocess.check_output(
            [
                "/usr/lib/wsl/lib/nvidia-smi",
                "--id=0",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
        manifest_pin = file_pin(args.manifest)
        require(manifest_pin["sha256"] == MANIFEST_SHA, "actual-serving manifest changed")
        data = ActualDataset(args.manifest)
        require(file_pin(args.manifest) == manifest_pin, "manifest changed while loading")
        require(not data.manifest["missing_embedding_ids"], "native F32 embeddings are missing")
        rows = {}

        def lookup_row(token):
            if token not in rows:
                rows[token] = data.embedding_lookup(token, device=device)
            return rows[token]

        lookup = ExternalNativeF32Embedding(
            lookup=lookup_row,
            vocab_size=248320,
            hidden_size=2560,
            target_identity="frozen native Qwen3.5-4B Q4_K_M actual FIT histories",
            target_sha256=data.manifest["target"]["sha256"],
            capture_sha256=manifest_pin["sha256"],
        )
        actor = build_modern_actor(
            INITIALIZER, CONFIG, embedding_lookup=lookup, device=device, scale_only=False
        )
        selection = enable_selected_sign(actor)
        absolute = actor.drafter.d2t + torch.arange(50000, device=device)
        reverse = torch.full((248320,), -1, device=device, dtype=torch.long)
        reverse[absolute] = torch.arange(50000, device=device)
        supported = set(absolute.detach().cpu().tolist())
        eligible = [
            w
            for w in data.windows
            if w["role"] == "fit"
            and w["max_depth"] == 2
            and w["anchor"] <= 256
            and w["native_supervision_reachable_mask"] == [True, True]
            and all(label in supported for label in w["labels"])
        ]
        eligible.sort(key=lambda w: (w["anchor"], w["id"]))
        windows = []
        for window in eligible:
            if not windows or window["group_id"] != windows[0]["group_id"]:
                windows.append(window)
            if len(windows) == 2:
                break
        require(len(windows) == 2, "two supported depth2 FIT windows from distinct groups required")
        groups = selected_parameter_groups(actor, sign_lr=1e-3, scale_lr=1e-5)
        parameters = tuple(parameter for group in groups for parameter in group["params"])
        optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)

        def forward(window):
            batch = data.materialize(window, device)
            adapter = actor.adapter
            cache = adapter.build_context_cache(
                batch.prefix_tokens, batch.prefix_rawfeatures, chunk_size=64
            )
            state = adapter.encode_feature(batch.seed_rawfeature)
            token = window["seed_token"]
            values = []
            for d in range(window["max_depth"]):
                step = adapter.decode_step(token, state, window["anchor"] + d, cache)
                cache = step.cache
                values.append(step.logits)
                state = step.pre_norm
                token = window["labels"][d]
            return torch.stack(values)

        def inspect(window, values):
            labels = torch.tensor(window["labels"], device=device)
            classes = reverse[labels]
            proposals = absolute[values.argmax(-1)]
            correct = proposals.eq(labels)
            prefix = int(correct.long().cumprod(0).sum())
            margins = []
            for d, cls in enumerate(classes.tolist()):
                if cls < 0:
                    margins.append(None)
                    continue
                other = values[d].scatter(0, classes[d : d + 1], -torch.inf).max()
                margins.append(float((values[d, cls] - other).detach()))
            return {
                "id": window["id"],
                "role": window["role"],
                "native_prefix": window["native_prefix"],
                "actor_prefix": prefix,
                "proposals": proposals.detach().cpu().tolist(),
                "labels": window["labels"],
                "supported": classes.ge(0).cpu().tolist(),
                "margins": margins,
                "protected": [
                    d < min(window["native_prefix"], prefix) for d in range(len(classes))
                ],
            }

        forward_pin = forward_semantics(Path(__file__).read_text())
        require(
            forward_pin
            == forward_semantics(
                Path(modern_actor.__file__).with_name("modern_train.py").read_text()
            ),
            "original actual-history forward changed",
        )
        source_paths = [
            Path(__file__),
            Path(modern_actor.__file__),
            Path(modern_actual_data.__file__),
            Path(modern_selected_sign.__file__),
            Path(native_step.__file__),
            Path(recurrent_binary.__file__),
            Path(recurrent_qat.__file__),
            Path(modern_actor.__file__).with_name("modern_train.py"),
            Path(modern_actor.__file__).with_name("modern_train_protected.py"),
        ]
        source_refs.extend(file_pin(path) for path in source_paths)
        source_directory = args.out / "source"
        source_directory.mkdir()
        snapshots = []
        for index, ref in enumerate(source_refs):
            require(ref["bytes"] <= 1024**2, "unexpected source size")
            destination = source_directory / f"{index:02d}-{Path(ref['path']).name}"
            shutil.copyfile(ref["path"], destination)
            require(file_pin(destination)["sha256"] == ref["sha256"], "source snapshot differs")
            snapshots.append({"source": ref, "snapshot": file_pin(destination)})
        budget = selected_state_budget(actor)
        contract = {
            "schema": "modern_selected_O_actual_GPU_plumbing_v1",
            "sources": snapshots,
            "manifest": manifest_pin,
            "actor": actor.contract,
            "selection": selection,
            "initializer": actor.initializer,
            "config": actor.config,
            "hardware": hardware,
            "precision": "F32Torch/referenceA8, own-prefix F16KV64; TF32disabled",
            "forward_semantics_sha256": forward_pin,
            "window_selection": (
                "first two distinct FIT groups sorted by anchor/id; "
                "depth2,anchor<=256,both static-reachable/supported; no actor outcome selection"
            ),
            "windows": windows,
            "optimizer": {
                "name": "Adam",
                "sign_lr": 1e-3,
                "scale_lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "objective": (
                "diagnostic CE sum/fixed4 over one two-label FIT window/update; "
                "no retention; two distinct unique windows; not a training comparison"
            ),
            "physical_updates": 3,
            "unique_trajectory_updates": 2,
            "seed": 37,
            "environment": {
                key: os.environ[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_CACHE_PATH",
                    "CUDA_CACHE_MAXSIZE",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
            "storage_floor_bytes": FLOOR,
            "preflight_output_reserve_bytes": OUTPUT_RESERVE,
            "payload_estimate": budget,
            "migration": "none; original immutable initializer only",
            "scope": (
                "selected O STE/backward, exact compact resume and packed mutation plumbing "
                "only; no capacity, generalization, screen or throughput claim"
            ),
        }
        guard()
        write(args.out / "freeze.json", contract)
        costs["initialization_seconds"] = time.perf_counter() - started

        def readout(name, model):
            guard()
            begin = time.perf_counter()
            values = []
            with torch.no_grad(), shared_round_hard_signs(actor.linears):
                for window in windows:
                    guard()
                    values.append(inspect(window, forward(window)))
            result = {
                "model": model,
                "manifest": manifest_pin,
                "rows": values,
                "dense_arrays_saved": False,
                "scope": "two reused FIT windows only; model-side not native admission",
                "wall_seconds": time.perf_counter() - begin,
            }
            write(args.out / (name + ".json"), result)
            costs[name + "_seconds"] = result["wall_seconds"]
            return result

        phase = "zero_export"
        guard(reserve=48 * 1024**2)
        begin = time.perf_counter()
        zero_export = export_selected_sign_gguf(actor, args.out / "zero.gguf")
        require(
            zero_export["output"]["sha256"] == actor.initializer["sha256"]
            and zero_export["output"]["bytes"] == actor.initializer["bytes"],
            "zero-update export differs from initializer",
        )
        write(args.out / "zero-export.json", zero_export)
        costs["zero_export_seconds"] = time.perf_counter() - begin
        before = readout("before-fit", zero_export["output"])
        updates = []
        cursor = 0

        def update(window):
            guard()
            require(window["role"] == "fit", "development cannot enter training")
            actor.drafter.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with shared_round_hard_signs(actor.linears):
                values = forward(window)
                classes = reverse[torch.tensor(window["labels"], device=device)]
                require(
                    bool((classes >= 0).all())
                    and window["native_supervision_reachable_mask"] == [True, True],
                    "diagnostic labels are not statically eligible",
                )
                ce = values.sum() * 0.0
                for depth in range(2):
                    ce = ce + torch.nn.functional.cross_entropy(
                        values[depth : depth + 1], classes[depth : depth + 1]
                    )
                loss = ce / 4.0
                require(bool(torch.isfinite(loss)), "nonfinite CE")
                loss.backward()
                sign_gradient = actor.linears[SELECTED_PATH].latent_sign.grad
                require(
                    sign_gradient is not None
                    and bool(torch.isfinite(sign_gradient).all())
                    and bool(sign_gradient.ne(0).any()),
                    "selected O sign gradient must be finite and nonzero",
                )
                sign_stats = {
                    "norm": float(sign_gradient.norm()),
                    "max_absolute": float(sign_gradient.abs().max()),
                    "nonzero_elements": int(sign_gradient.ne(0).sum()),
                }
                scale_norms = {}
                for path, module in actor.linears.items():
                    gradient = module.scale_offset.grad
                    require(
                        gradient is not None and bool(torch.isfinite(gradient).all()),
                        "missing/nonfinite scale gradient: " + path,
                    )
                    scale_norms[path] = float(gradient.norm())
                    if path != SELECTED_PATH:
                        require(
                            module.latent_sign.grad is None
                            and not module.latent_sign.requires_grad,
                            "frozen sign gradient appeared",
                        )
                for path in NATIVE_NORM_PATHS.values():
                    norm = actor.drafter.get_submodule(path).weight
                    require(
                        not norm.requires_grad and norm.grad is None,
                        "frozen norm gradient appeared",
                    )
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                require(bool(torch.isfinite(norm)), "nonfinite combined gradient")
                optimizer.step()
                clamp_selected_latents_(actor)
                for module in actor.linears.values():
                    module.project_scales_()
            for parameter in parameters:
                state = optimizer.state[parameter]
                require(
                    all(
                        state[key].dtype == torch.float32
                        and state[key].shape == parameter.shape
                        and bool(torch.isfinite(state[key]).all())
                        for key in ("exp_avg", "exp_avg_sq")
                    ),
                    "Adam moments must remain finite F32",
                )
            torch.cuda.synchronize()
            return {
                "window": window["id"],
                "loss": float(loss.detach()),
                "sign_gradient": sign_stats,
                "scale_gradient_norms": scale_norms,
                "combined_gradient_norm": float(norm),
                "wall_seconds": time.perf_counter() - begin,
            }

        def numeric_packet():
            return {
                "schema": "modern_selected_O_plumbing_checkpoint_v1",
                "selected_state": collect_selected_state(actor),
                "optimizer": cpu(optimizer.state_dict()),
                "rng": torch.get_rng_state().clone(),
                "cuda_rng": torch.cuda.get_rng_state().clone(),
                "python_rng": random.getstate(),
                "cursor": cursor,
                "contract": contract,
            }

        def save_checkpoint(number):
            guard(reserve=budget["F32_model_plus_Adam_two_moments_payload_bytes"] + 4 * 1024**2)
            begin = time.perf_counter()
            state = numeric_packet()
            require(state["cursor"] == number, "checkpoint cursor differs")
            destination = args.out / f"checkpoint-{number}.pt"
            temporary = destination.with_suffix(".tmp")
            with temporary.open("xb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            loaded = torch.load(temporary, weights_only=False, map_location="cpu")
            require(bit_equal(state, loaded), "checkpoint roundtrip differs")
            os.link(temporary, destination)
            temporary.unlink()
            descriptor = os.open(args.out, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            reference = file_pin(destination)
            checkpoints.append(reference)
            costs[f"checkpoint_{number}_seconds"] = time.perf_counter() - begin
            write(
                args.out / f"checkpoint-{number}-receipt.json",
                {
                    "checkpoint": reference,
                    "roundtrip_bit_exact": True,
                    "cursor": number,
                    "wall_seconds": costs[f"checkpoint_{number}_seconds"],
                },
            )
            return reference

        phase = "unique_update_1"
        first = update(windows[0])
        cursor = 1
        updates.append(first)
        checkpoint1 = save_checkpoint(1)
        phase = "unique_update_2"
        second = update(windows[1])
        cursor = 2
        updates.append(second)
        expected = numeric_packet()
        phase = "checkpoint_restore_and_repeat_update_2"
        guard()
        begin = time.perf_counter()
        restored = torch.load(checkpoint1["path"], weights_only=False, map_location="cpu")
        require(
            restored["contract"] == contract and restored["cursor"] == 1,
            "restore contract/cursor differs",
        )
        require(file_pin(checkpoint1["path"]) == checkpoint1, "checkpoint changed before restore")
        load_selected_state(actor, restored["selected_state"])
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["rng"])
        torch.cuda.set_rng_state(restored["cuda_rng"])
        random.setstate(restored["python_rng"])
        cursor = restored["cursor"]
        del restored
        costs["restore_seconds"] = time.perf_counter() - begin
        repeated = update(windows[1])
        cursor = 2
        actual = numeric_packet()
        require(
            bit_equal(expected, actual) and second["loss"] == repeated["loss"],
            "repeated update2 model/Adam/RNG/cursor differs",
        )
        del expected, actual
        gc.collect()
        save_checkpoint(2)
        actor.drafter.zero_grad(set_to_none=True)
        learned_state = collect_selected_state(actor)
        learned_metrics = selected_sign_metrics(actor)
        require(learned_metrics["changed_latents"] > 0, "no actual selected latent displacement")
        write(args.out / "learned-sign-metrics.json", learned_metrics)
        phase = "learned_export_and_readout"
        guard(reserve=48 * 1024**2)
        begin = time.perf_counter()
        learned_export = export_selected_sign_gguf(actor, args.out / "learned.gguf")
        write(args.out / "learned-export.json", learned_export)
        costs["learned_export_seconds"] = time.perf_counter() - begin
        after = readout("learned-fit", learned_export["output"])
        phase = "forced_one_bit_diagnostic_NOT_TRAINED"
        guard(reserve=48 * 1024**2)
        latent = actor.linears[SELECTED_PATH].latent_sign
        old_value = float(latent[0, 0].detach())
        new_value = -0.5 if old_value >= 0 else 0.5
        with torch.no_grad():
            latent[0, 0] = new_value
        forced_metrics = selected_sign_metrics(actor, learned_state)
        require(
            forced_metrics["hard_sign_flips"] == 1 and forced_metrics["changed_latents"] == 1,
            "forced diagnostic changed more than one latent",
        )
        begin = time.perf_counter()
        forced_export = export_selected_sign_gguf(actor, args.out / "FORCED-DIAGNOSTIC.gguf")
        write(args.out / "forced-export.json", forced_export)
        costs["forced_export_seconds"] = time.perf_counter() - begin
        audit = forced_export_audit(learned_export, forced_export)
        write(
            args.out / "forced-diagnostic.json",
            {
                "classification": "FORCED_PACKING_DIAGNOSTIC_NOT_TRAINED_CANDIDATE",
                "base": SELECTED_BASES[0],
                "row": 0,
                "column": 0,
                "before_latent": old_value,
                "forced_latent": new_value,
                "metrics_against_learned": forced_metrics,
                "learned_model": learned_export["output"],
                "forced_model": forced_export["output"],
                **audit,
            },
        )
        forced_readout = readout("forced-fit", forced_export["output"])
        load_selected_state(actor, learned_state)
        require(
            bit_equal(learned_state, collect_selected_state(actor)),
            "failed to restore learned state after diagnostic",
        )
        guard()
        costs["unique_update_seconds"] = first["wall_seconds"] + second["wall_seconds"]
        costs["physical_update_seconds"] = costs["unique_update_seconds"] + repeated["wall_seconds"]
        result = {
            "status": "SELECTED_O_GPU_BACKWARD_RESUME_EXPORT_PLUMBING_PASS",
            "scope": contract["scope"],
            "physical_updates": 3,
            "unique_trajectory_updates": 2,
            "updates": updates,
            "repeated_update2": repeated,
            "checkpoint_roundtrip_and_resume_bit_exact": True,
            "checkpoints": checkpoints,
            "actual_checkpoint_bytes": [ref["bytes"] for ref in checkpoints],
            "payload_estimate": budget,
            "selected_sign_metrics": learned_metrics,
            "zero_export": zero_export["output"],
            "learned_export": learned_export["output"],
            "forced_diagnostic_export": forced_export["output"],
            "forced_diagnostic_only": audit,
            "before": before,
            "learned": after,
            "forced": forced_readout,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "native_embedding_rows_loaded": len(rows),
            "dense_arrays_saved": False,
            "costs": costs,
            "whole_job_wall_seconds": time.perf_counter() - started,
            "native_admission_passed": False,
            "generalization_or_capacity_success": False,
            "screen_predictor_validated": False,
        }
        write(args.out / "result.json", result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "actual_checkpoint_bytes": result["actual_checkpoint_bytes"],
                    "learned_sign_flips": learned_metrics["hard_sign_flips"],
                    "physical_update_seconds": costs["physical_update_seconds"],
                    "whole_job_wall_seconds": result["whole_job_wall_seconds"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        write(
            args.out / "failure.json",
            {
                "error": repr(error),
                "phase": phase,
                "preserved_checkpoints": checkpoints,
                "costs": costs,
                "whole_job_wall_seconds": time.perf_counter() - started,
                "scope": "plumbing qualification failed; no native/capacity/generalization claim",
            },
        )
        raise


if __name__ == "__main__":
    main()

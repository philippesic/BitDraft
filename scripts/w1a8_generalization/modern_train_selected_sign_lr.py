"""Declared mature-O sign-LR continuation: same3840 state/moments, LR alone varies.

The original selected-sign driver validates the immutable TRUE3840 parent before
loading. Only afterward may O LR change1e-3 to3e-3. Global1920-origin accounting
and Adam bias counters remain intact; separate3840-origin LR-stage deltas describe
this continuation. No new objective, data, reachability or replay-bank change.
"""

from __future__ import annotations

# Project path bootstrap precedes optional Torch/project imports.
# ruff: noqa: E402
import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

PROJECT = Path("/home/philip/binary-eagle-decoding")
SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / "src"), str(PROJECT / "third_party/llama.cpp/gguf-py")]
import modern_selected_sign as sign_helpers
import modern_train_native_replay as native_helpers
import modern_train_protected as protected_helpers
import modern_train_selected_sign as original
import torch
from modern_actor import (
    build_modern_actor,
    file_pin,
)
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
)
from modern_train_native_replay import (
    MARGIN_FLOOR,
    WARM_CURSOR,
    native_fit_references,
    native_schedule,
    stage_totals,
    validate_v1_warm_start,
)
from modern_train_protected import (
    ProtectedReplaySchedule,
    cpu,
    forward_semantics,
    normalized_replay_loss,
    require,
    validate_original_reference,
    write,
)

from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH
from w1a1_eagle.recurrent_qat import shared_round_hard_signs

CHECKPOINT_SCHEMA = "modern_selected_O_mature_LR_checkpoint_v1"
ARTIFACTS = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts")
EXISTING_CACHE = PROJECT / "runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda"
FLOOR = 2 * 1024**3 + 64 * 1024**2
LR_START = 3840
ORIGINAL_TRAINER_SHA256 = "0336f7437a37ea7eedad9f6229927c8beeacbbe3d0a45515cf5c382435151ff1"
ORIGINAL_TRAINER_BYTES = 41321
PARENT_CHECKPOINT_SHA256 = "9f4146226a52fc5e8b07f2b8372da68bd37f0359bcdcc90af6e8d3bb34e10ce1"
PARENT_CHECKPOINT_BYTES = 79712677
ALLOWED_SIGN_LRS = (1e-3, 3e-3)
checkpoint_equal = original.checkpoint_equal
_adam_entry = original._adam_entry
validate_selected_stage = original.validate_selected_stage
prepare_sign_gradient = original.prepare_sign_gradient
assert_sign_mode = original.assert_sign_mode
make_origin = original.make_origin


def original_source_pin():
    reference = file_pin(original.__file__)
    require(
        reference["sha256"] == ORIGINAL_TRAINER_SHA256
        and reference["bytes"] == ORIGINAL_TRAINER_BYTES,
        "immutable original selected-sign trainer source differs",
    )
    return reference


def validate_mature_parent(actor, saved, checkpoint_pin, expected_contract):
    source = original_source_pin()
    # Invoke the actual original validator on the unmodified parent packet.
    original.validate_selected_resume(actor, saved, saved["contract"], True)
    require(
        saved["cursor"] == LR_START
        and saved["contract"] == expected_contract
        and saved["contract"]["trainer"] == source
        and saved["contract"]["horizon"] == 19200
        and saved["contract"]["native_fit_bank_tokens"] == 356
        and saved["contract"]["selected_optimizer"]["learn_signs"] is True
        and saved["contract"]["selected_optimizer"]["sign_lr"] == 1e-3
        and saved["contract"]["selected_optimizer"]["scale_lr"] == 1e-5,
        "only the original TRUE3840 mature parent is supported",
    )
    require(
        checkpoint_pin.get("sha256") == PARENT_CHECKPOINT_SHA256
        and checkpoint_pin.get("bytes") == PARENT_CHECKPOINT_BYTES,
        "selected mature parent checkpoint identity differs",
    )


def make_lr_origin(parent, parent_pin):
    return {
        "parent_checkpoint": copy.deepcopy(parent_pin),
        "original_parent_contract": copy.deepcopy(parent["contract"]),
        "global_start_cursor": LR_START,
        "initial_costs": dict(parent["costs"]),
        "initial_exposure": dict(parent["exposure"]),
        "initial_trainer_seconds": parent["trainer_seconds"],
        "initial_selected_stage": dict(parent["selected_stage"]),
        "sign_moment_origin_cursor": WARM_CURSOR,
        "policy": (
            "load original complete model/optimizer first, then change ONLY sign group LR; "
            "no moment/RNG/cursor/exposure reset"
        ),
    }


def make_lr_contract(parent_contract, lr_origin, sign_lr):
    require(sign_lr in ALLOWED_SIGN_LRS, "supported sign LR1e-3 or3e-3 required")
    require(
        parent_contract == lr_origin["original_parent_contract"], "LR-stage parent contract differs"
    )
    contract = copy.deepcopy(parent_contract)
    contract["trainer"] = file_pin(__file__)
    contract["stage_kind"] = "declared_mature_O_LR_continuation_from_global3840"
    contract["original_selected_trainer_source"] = original_source_pin()
    contract["selected_optimizer"]["sign_lr"] = sign_lr
    contract["selected_optimizer"]["param_group_options"][0]["lr"] = sign_lr
    contract["lr_migration"] = copy.deepcopy(lr_origin)
    contract["lr_migration"]["chosen_sign_lr"] = sign_lr
    contract["lr_cost_detail"] = (
        "LR-stage deltas are subsets of inherited cumulative work and selected-stage detail; "
        "do not sum them again. Sign Adam steps remain global_cursor-1920."
    )
    return contract


def migrate_mature_state(actor, optimizer, parent, parent_pin, sign_lr, expected_contract):
    validate_mature_parent(actor, parent, parent_pin, expected_contract)
    require(
        sign_lr in ALLOWED_SIGN_LRS and not optimizer.state, "fresh optimizer/declared LR required"
    )
    initial_options = [
        {k: v for k, v in group.items() if k != "params"}
        for group in optimizer.state_dict()["param_groups"]
    ]
    require(
        initial_options == parent["contract"]["selected_optimizer"]["param_group_options"],
        "fresh optimizer must begin with original1e-3 group options",
    )
    load_selected_state(actor, parent["selected_state"])
    optimizer.load_state_dict(parent["optimizer"])
    # No optimizer-state reset or mutation precedes loading the original state.
    expected = cpu(parent["optimizer"])
    expected["param_groups"][0]["lr"] = sign_lr
    optimizer.param_groups[0]["lr"] = sign_lr
    require(
        checkpoint_equal(cpu(optimizer.state_dict()), expected),
        "LR migration changed Adam state/options beyond chosen LR",
    )
    require(
        checkpoint_equal(collect_selected_state(actor), parent["selected_state"]),
        "LR migration changed model state",
    )
    return {
        "parent": parent_pin,
        "original_sign_lr": 1e-3,
        "chosen_sign_lr": sign_lr,
        "scale_lr": 1e-5,
        "all_model_and_Adam_tensors_unchanged": True,
        "sign_moments_preserved": True,
        "global_cursor": LR_START,
        "sign_Adam_step": LR_START - WARM_CURSOR,
        "scope": "declared mature LR migration only; not a capacity/native/generalization result",
    }


def lr_stage_totals(cursor, costs, exposure, trainer_seconds, selected_stage, origin):
    require(
        origin["global_start_cursor"] == LR_START
        and origin["sign_moment_origin_cursor"] == WARM_CURSOR,
        "LR-stage/global sign-moment origins differ",
    )
    validate_selected_stage(selected_stage, cursor, True)
    result = stage_totals(cursor, costs, exposure, trainer_seconds, origin)
    before = origin["initial_selected_stage"]
    validate_selected_stage(before, LR_START, True)
    delta = {key: selected_stage[key] - before[key] for key in selected_stage}
    require(
        all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in delta.values())
        and delta["updates"] == delta["sign_optimizer_updates"] == cursor - LR_START,
        "LR-stage selected-cost/counter delta invalid",
    )
    return {
        **result,
        "selected_stage_delta": delta,
        "accounting": "increments within cumulative totals, not additional costs to sum",
    }


def _lr_optimizer_payload(payload, selected_state, cursor, learn_signs, stage_updates, sign_lr):
    groups, states = payload["param_groups"], payload["state"]
    require(
        len(groups) == 2 and [len(g["params"]) for g in groups] == [1, 9],
        "selected optimizer groups differ",
    )
    require(
        [g.get("name") for g in groups] == ["selected_O_latent_sign", "nine_raw_scale_offsets"],
        "selected optimizer group order differs",
    )
    require(
        [g["lr"] for g in groups] == [sign_lr, 1e-5], "selected optimizer learning rates differ"
    )
    for group in groups:
        require(
            tuple(group["betas"]) == (0.9, 0.999)
            and group["eps"] == 1e-8
            and group.get("weight_decay") == 0
            and group.get("amsgrad") is False
            and group.get("maximize") is False
            and group.get("capturable") is False
            and group.get("differentiable") is False
            and group.get("foreach") is None
            and group.get("fused") is None,
            "selected Adam options differ",
        )
    ids = [p for group in groups for p in group["params"]]
    require(
        len(set(ids)) == 10 and all(type(p) is int for p in ids),
        "selected optimizer parameter identities differ",
    )
    sign_id, scale_ids = groups[0]["params"][0], groups[1]["params"]
    expected = set(scale_ids) | ({sign_id} if learn_signs and stage_updates else set())
    require(set(states) == expected, "selected sign moments/scope differ")
    for base, param in zip(CANDIDATE_D_BASE_TO_PATH, scale_ids, strict=True):
        _adam_entry(states[param], selected_state["scale_offsets"][base].shape, cursor)
    if sign_id in states:
        _adam_entry(
            states[sign_id], selected_state["latent_signs"][SELECTED_BASES[0]].shape, stage_updates
        )


def validate_lr_resume(actor, saved, contract, learn_signs):
    require(
        learn_signs is True
        and contract.get("trainer") == file_pin(__file__)
        and contract.get("original_selected_trainer_source") == original_source_pin(),
        "LR-stage source/mode differs",
    )
    sign_lr = contract["selected_optimizer"]["sign_lr"]
    require(
        sign_lr in ALLOWED_SIGN_LRS and contract["lr_migration"]["chosen_sign_lr"] == sign_lr,
        "LR-stage chosen learning rate differs",
    )
    require(
        saved.get("schema") == CHECKPOINT_SCHEMA
        and saved.get("contract") == contract
        and saved.get("status") in ("COMPLETE", "RECOVERY", "INTERRUPTED"),
        "selected resume schema/source/objective differs",
    )
    cursor = saved["cursor"]
    require(
        type(cursor) is int and LR_START <= cursor <= 19200,
        "selected resume horizon/cursor differs",
    )
    require(
        contract["selected_optimizer"]["learn_signs"] is learn_signs,
        "selected resume learn-signs mode differs",
    )
    validate_selected_stage(saved["selected_stage"], cursor, learn_signs)
    sign_helpers._validate_state(actor, saved["selected_state"])
    _lr_optimizer_payload(
        saved["optimizer"],
        saved["selected_state"],
        cursor,
        learn_signs,
        cursor - WARM_CURSOR,
        contract["selected_optimizer"]["sign_lr"],
    )
    require(
        [
            {k: v for k, v in group.items() if k != "params"}
            for group in saved["optimizer"]["param_groups"]
        ]
        == contract["selected_optimizer"]["param_group_options"],
        "selected resume optimizer options differ",
    )
    if not learn_signs:
        require(
            sign_helpers._tensor_hash(saved["selected_state"]["latent_signs"][SELECTED_BASES[0]])
            == actor.initial_sign_hashes[SELECTED_PATH],
            "control checkpoint signs moved",
        )
    require(
        saved["exposure"]["ce_windows"] == 2 * cursor
        and saved["exposure"]["replay_windows"] == cursor,
        "selected resume cumulative exposure differs",
    )
    require(
        type(saved["trainer_seconds"]) in (int, float)
        and math.isfinite(saved["trainer_seconds"])
        and saved["trainer_seconds"] >= 0,
        "selected resume trainer cost invalid",
    )
    stage_totals(
        cursor, saved["costs"], saved["exposure"], saved["trainer_seconds"], contract["warm_start"]
    )
    require(
        all(key in saved for key in ("rng", "cuda_rng", "python_rng")),
        "selected resume RNG missing",
    )
    for key in ("rng", "cuda_rng"):
        value = saved[key]
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.is_contiguous()
            and value.numel() >= 16,
            "selected resume RNG tensor differs",
        )
    require(saved["rng"].numel() == torch.get_rng_state().numel(), "CPU RNG geometry differs")
    random.Random().setstate(saved["python_rng"])
    expected_stage = lr_stage_totals(
        cursor,
        saved["costs"],
        saved["exposure"],
        saved["trainer_seconds"],
        saved["selected_stage"],
        contract["lr_migration"],
    )
    require(
        "lr_stage" in saved and checkpoint_equal(saved["lr_stage"], expected_stage),
        "LR-stage cost/exposure accounting differs",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["train"], default="train")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--original-trainer", type=Path, default=Path(__file__).with_name("modern_train.py")
    )
    parser.add_argument("--scope", choices=["all"], default="all")
    parser.add_argument(
        "--protected-trainer",
        type=Path,
        default=Path(__file__).with_name("modern_train_protected.py"),
    )
    parser.add_argument("--steps", type=int, default=4320)
    parser.add_argument("--learn-signs", choices=["true"], default="true")
    parser.add_argument("--sign-lr", type=float, choices=ALLOWED_SIGN_LRS, required=True)
    parser.add_argument("--scale-lr", type=float, choices=[1e-5], default=1e-5)
    parser.add_argument("--retention", type=float, choices=[10.0], default=10.0)
    continuation = parser.add_mutually_exclusive_group(required=True)
    continuation.add_argument("--warm-selected", type=Path)
    continuation.add_argument("--resume", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if not args.allow_cuda or not 0 < args.steps <= 19200 or args.retention < 0:
        raise ValueError("explicit sole GPU owner / known horizon required")
    args.out = args.out.resolve()
    require(
        args.out.is_relative_to(ARTIFACTS.resolve()) and args.out != ARTIFACTS.resolve(),
        "unique C: output required",
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and os.environ.get("CUDA_CACHE_PATH")
        and Path(os.environ["CUDA_CACHE_PATH"]).resolve() == EXISTING_CACHE.resolve()
        and EXISTING_CACHE.is_dir(),
        "explicit existing training CUDA cache/environment required",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    learn_signs = args.learn_signs == "true"
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def storage_guard(reserve=0):
        for filesystem in (PROJECT, Path("/mnt/c")):
            info = os.statvfs(filesystem)
            required = FLOOR + (reserve if filesystem == Path("/mnt/c") else 0)
            require(
                info.f_bavail * info.f_frsize >= required,
                "project/Windows storage floor or checkpoint reserve reached: " + str(filesystem),
            )

    def guard():
        c = json.loads(
            (Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text()
        )["rtx5080"]
        if (
            stop_requested
            or c.get("pause_requested")
            or c.get("stop_all_requested")
            or (args.out / "STOP").exists()
        ):
            raise RuntimeError("GPU stop requested")
        storage_guard(88 * 1024**2)

    guard()
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5080":
        raise ValueError("RTX5080 required")
    device = torch.device("cuda:0")
    data = ActualDataset(args.manifest)
    if data.manifest["missing_embedding_ids"]:
        raise ValueError("native F32 embeddings missing")
    embeddings = {
        token: data.embedding_lookup(token, device=device)
        for token in data.manifest["required_embedding_ids"]
    }
    lookup = ExternalNativeF32Embedding(
        lookup=embeddings.__getitem__,
        vocab_size=248320,
        hidden_size=2560,
        target_identity="frozen native Qwen3.5-4B Q4_K_M actual parent",
        target_sha256=data.manifest["target"]["sha256"],
        capture_sha256=file_pin(args.manifest)["sha256"],
    )
    actor = build_modern_actor(
        PROJECT / "runs/w1a8-modern-packed-init-20261103-01/model.gguf",
        PROJECT / "runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json",
        embedding_lookup=lookup,
        device=device,
        scale_only=False,
    )
    selected_contract = enable_selected_sign(actor)
    absolute = actor.drafter.d2t + torch.arange(50000, device=device)
    reverse = torch.full((248320,), -1, device=device, dtype=torch.long)
    reverse[absolute] = torch.arange(50000, device=device)
    contract = {
        "manifest": file_pin(args.manifest),
        "actor": actor.contract,
        "trainer": original_source_pin(),
        "view_source": file_pin(Path(__file__).with_name("modern_actual_data.py")),
        "optimizer": {"name": "Adam", "lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8},
        "retention": args.retention,
        "seed": 37,
        "horizon": 19200,
        "batch_windows": 2,
        "margin_cap": 2.0,
        "objective": (
            "Original CE sum/fixed4 from2 ordinary fit windows plus retention*mean "
            "protected-token margin deficit on1 independent group-balanced fit replay; "
            "no retention on ordinary CE windows."
        ),
        "optimizer_scope": args.scope,
        "replay_windows_per_update": 1,
        "context": (
            "detached current-actor KV rebuilt from exact observed rawfeature lineage; "
            "own FC/prenorm in goldprefix unroll"
        ),
        "runtime": "original immutable packed-runtime03; bridgefix overlay deliberately separate",
        "precision": "F32Torch/A8, F16KV, TF32disabled",
    }

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
            "protected": [d < min(window["native_prefix"], prefix) for d in range(len(classes))],
        }

    def readout(name, windows):
        begin = time.perf_counter()
        rows = []
        with torch.no_grad(), shared_round_hard_signs(actor.linears):
            for index, w in enumerate(windows):
                guard()
                rows.append(inspect(w, forward(w)))
                if index % 128 == 0:
                    print(
                        json.dumps({"stage": name, "windows": index, "total": len(windows)}),
                        flush=True,
                    )
        value = {
            "rows": rows,
            "wall_seconds": time.perf_counter() - begin,
            "immutable_native_evaluation_masks": native_masks,
            "immutable_replay_fit_tokens": 356,
            "reference": file_pin(args.reference),
            "readout_protected_flags_are_NOT_training_bank": True,
        }
        write(args.out / (name + ".json"), value)
        return value

    fit = [w for w in data.windows if w["role"] == "fit"]
    dev = [w for w in data.windows if w["role"] == "development"]
    require(
        len(fit) == 1414 and len(dev) == 724, "full actual parent36 fit/development views required"
    )
    native_masks = {
        role: sum(len(w["native_protections"]) for w in data.windows if w["role"] == role)
        for role in ("fit", "development")
    }
    require(
        native_masks == {"fit": 356, "development": 193}, "all549 native success masks must remain"
    )
    reference = json.loads(args.reference.read_text())
    reference_contract = json.loads(args.reference.with_name("freeze.json").read_text())
    refs = validate_original_reference(
        reference, reference_contract, contract, data.windows, args.original_trainer, __file__
    )
    helper_pin = file_pin(protected_helpers.__file__)
    requested_helper = file_pin(args.protected_trainer)
    require(
        all(helper_pin[k] == requested_helper[k] for k in ("sha256", "bytes")),
        "loaded protected helper differs from explicitly pinned original source",
    )
    require(
        forward_semantics(Path(protected_helpers.__file__).read_text())
        == forward_semantics(Path(__file__).read_text()),
        "protected forward semantics changed",
    )
    old_schedule = ProtectedReplaySchedule(fit, refs, seed=37).contract
    native_refs = native_fit_references(fit, refs, set(absolute.detach().cpu().tolist()))
    require(
        sum(sum(row["protected"]) for row in native_refs.values()) == 356,
        "all356 native FIT successes must enter the expanded bank",
    )
    schedule = native_schedule(fit, native_refs)
    require(
        len({w["group_id"] for w in fit}) == len(schedule.groups) == 24,
        "all24 FIT groups must remain in the expanded native bank",
    )
    order = list(range(len(fit)))
    random.Random(37).shuffle(order)
    paths = list(CANDIDATE_D_BASE_TO_PATH.values())
    parameter_groups = selected_parameter_groups(actor, sign_lr=1e-3, scale_lr=args.scale_lr)
    selected_parameters = tuple(p for group in parameter_groups for p in group["params"])
    contract.update(
        {
            "reference": file_pin(args.reference),
            "reference_freeze": file_pin(args.reference.with_name("freeze.json")),
            "original_trainer": file_pin(args.original_trainer),
            "numeric_forward_semantics_sha256": forward_semantics(Path(__file__).read_text()),
            "original_reference_objective": reference_contract["objective"],
            "protected_schedule": schedule.contract,
            "ce_schedule": {
                "seed": 37,
                "order": [fit[i]["id"] for i in order],
                "selection": "(2*cursor+offset)%N; offset=0,1",
            },
            "optimized_scale_paths": paths,
            "native_evaluation_masks": native_masks,
            "readout": "all1414 fit and all724 development windows; dev never sampled for training",
        }
    )
    optimizer = torch.optim.Adam(parameter_groups, betas=(0.9, 0.999), eps=1e-8)
    cursor = 0
    train_seconds = 0.0
    costs = {
        "ce_forward_seconds": 0.0,
        "replay_forward_seconds": 0.0,
        "shared_backward_optimizer_seconds": 0.0,
    }
    exposure = {
        "ce_windows": 0,
        "ce_supported_labels": 0,
        "replay_windows": 0,
        "replay_protected_tokens": 0,
    }
    source_checkpoint = args.resume or args.warm_selected
    source_pin = file_pin(source_checkpoint)
    saved = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    require(
        file_pin(source_checkpoint) == source_pin, "continuation checkpoint changed while reading"
    )
    if args.warm_selected:
        parent, parent_pin = saved, source_pin
    else:
        require(saved.get("schema") == CHECKPOINT_SCHEMA, "own LR-stage resume required")
        parent_pin = saved["contract"]["lr_migration"]["parent_checkpoint"]
        require(file_pin(parent_pin["path"]) == parent_pin, "mature parent checkpoint changed")
        parent = torch.load(parent_pin["path"], map_location="cpu", weights_only=False)
        require(file_pin(parent_pin["path"]) == parent_pin, "mature parent changed while reading")
    origin = parent["contract"]["warm_start"]
    require(
        file_pin(origin["checkpoint"]["path"]) == origin["checkpoint"],
        "original1920 ancestry changed",
    )
    original_saved = torch.load(
        origin["checkpoint"]["path"], map_location="cpu", weights_only=False
    )
    validate_v1_warm_start(original_saved, contract, old_schedule, helper_pin)
    require(
        make_origin(original_saved, origin["checkpoint"]) == origin, "original1920 origin changed"
    )
    del original_saved
    contract.update(
        {
            "objective": (
                "Original CE2windows/fixed4 plus lambda10 mean margin-deficit replay "
                "over ALL356 native FIT successes; "
                "target=max(0.1,min(original_GPU_margin,2)); declared v1 bank/objective migration"
            ),
            "protected_schedule": schedule.contract,
            "warm_start": origin,
            "protected_helper_source": helper_pin,
            "explicit_original_protected_source": requested_helper,
            "native_fit_bank_tokens": 356,
            "native_margin_floor": MARGIN_FLOOR,
            "native_reference_intersection_extras": 356
            - sum(sum(refs[w["id"]]["protected"]) for w in fit),
            "stage_kind": "declared_selected_O_capacity_comparison_from_balanced1920_v1",
            "selected_sign": selected_contract,
            "native_replay_trainer_source": file_pin(native_helpers.__file__),
            "selected_helper_source": file_pin(sign_helpers.__file__),
            "selected_optimizer": {
                "learn_signs": learn_signs,
                "sign_lr": 1e-3,
                "scale_lr": args.scale_lr,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "groups": ["selected_O_latent_sign", "nine_raw_scale_offsets"],
                "param_group_options": [
                    {k: v for k, v in group.items() if k != "params"}
                    for group in optimizer.state_dict()["param_groups"]
                ],
                "control": (
                    "clear O.grad BEFORE global clip/step; no O Adam moments; "
                    "nine scales retain original path order"
                ),
            },
            "cost_detail": (
                "selected-stage phase timers overlap original shared-backward/optimizer total; "
                "never sum them twice"
            ),
        }
    )
    validate_mature_parent(actor, parent, parent_pin, contract)
    lr_origin = make_lr_origin(parent, parent_pin)
    contract = make_lr_contract(contract, lr_origin, args.sign_lr)
    if args.resume:
        validate_lr_resume(actor, saved, contract, True)
    selected_stage = dict(saved["selected_stage"])
    lr_reference_state = parent["selected_state"]
    write(args.out / "freeze.json", contract)
    write(
        args.out / "continuation.json",
        {
            "mode": "warm_selected" if args.warm_selected else "resume",
            "checkpoint": source_pin,
            "loaded_global_cursor": saved["cursor"],
            "requested_global_cursor": args.steps,
            "original_protected_source": helper_pin,
        },
    )
    if args.warm_selected:
        migration_begin = time.perf_counter()
        migration = migrate_mature_state(
            actor, optimizer, parent, parent_pin, args.sign_lr, parent["contract"]
        )
        migration["non_training_wall_seconds"] = time.perf_counter() - migration_begin
        write(args.out / "lr-migration.json", migration)
    else:
        load_selected_state(actor, saved["selected_state"])
        optimizer.load_state_dict(saved["optimizer"])
    cursor, train_seconds = saved["cursor"], saved["trainer_seconds"]
    costs, exposure = dict(saved["costs"]), dict(saved["exposure"])
    assert_sign_mode(actor, optimizer, learn_signs, full=True)
    torch.set_rng_state(saved["rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    require(
        LR_START <= cursor < args.steps <= 19200,
        "continuation must extend global cursor within unchanged19200 horizon",
    )
    del saved, parent

    def save(status):
        assert_sign_mode(actor, optimizer, learn_signs, full=True)
        path = args.out / f"checkpoint-{cursor:08d}.pt"
        if path.exists():
            return path
        storage_guard(88 * 1024**2)
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "contract": contract,
            "selected_state": collect_selected_state(actor),
            "optimizer": cpu(optimizer.state_dict()),
            "cursor": cursor,
            "trainer_seconds": train_seconds,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "costs": dict(costs),
            "exposure": dict(exposure),
            "status": status,
            "selected_stage": dict(selected_stage),
            "lr_stage": lr_stage_totals(
                cursor, costs, exposure, train_seconds, selected_stage, lr_origin
            ),
            "stage": stage_totals(cursor, costs, exposure, train_seconds, origin),
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = torch.load(tmp, map_location="cpu", weights_only=False)
        if not checkpoint_equal(state, loaded):
            raise ValueError("checkpoint roundtrip differs")
        os.replace(tmp, path)
        write(args.out / "latest.json", file_pin(path))
        directory = os.open(args.out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        metric_begin = time.perf_counter()
        metrics = selected_sign_metrics(actor)
        metrics["relative_to_mature3840"] = selected_sign_metrics(actor, lr_reference_state)
        require(
            learn_signs or metrics["changed_latents"] == metrics["hard_sign_flips"] == 0,
            "control signs moved",
        )
        write(
            args.out / f"sign-metrics-{cursor:08d}.json",
            {
                **metrics,
                "global_cursor": cursor,
                "stage_updates": cursor - WARM_CURSOR,
                "interpretation": (
                    "measured movement/flips only; zero flips remains slow-learning/inconclusive, "
                    "never an early rejection"
                ),
                "checkpoint": file_pin(path),
                "wall_seconds": time.perf_counter() - metric_begin,
            },
        )
        return path

    receipts = (args.out / "updates.jsonl").open("x")
    in_update = False
    try:
        while cursor < args.steps:
            guard()
            transaction_begin = time.perf_counter()
            in_update = True
            actor.drafter.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            loss = 0.0
            ce_value = 0.0
            supported = 0
            with shared_round_hard_signs(actor.linears):
                for offset in range(2):
                    w = fit[order[(2 * cursor + offset) % len(order)]]
                    values = forward(w)
                    classes = reverse[torch.tensor(w["labels"], device=device)]
                    ce = values.sum() * 0.0
                    for d, cls in enumerate(classes.tolist()):
                        if cls >= 0 and w["native_supervision_reachable_mask"][d]:
                            ce = ce + torch.nn.functional.cross_entropy(
                                values[d : d + 1], classes[d : d + 1]
                            )
                            supported += 1
                    loss = loss + ce / 4.0
                    ce_value += float(ce.detach()) / 4.0
                torch.cuda.synchronize()
                ce_seconds = time.perf_counter() - t
                replay_begin = time.perf_counter()
                replay_group, replay_window = schedule.at(cursor)
                replay_values = forward(replay_window)
                replay_classes = reverse[torch.tensor(replay_window["labels"], device=device)]
                replay_loss, replay_tokens = normalized_replay_loss(
                    replay_values, replay_classes, native_refs[replay_window["id"]], cap=2.0
                )
                loss = loss + args.retention * replay_loss
                torch.cuda.synchronize()
                replay_seconds = time.perf_counter() - replay_begin
                backward_begin = time.perf_counter()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite loss")
                loss.backward()
                torch.cuda.synchronize()
                phase_backward = time.perf_counter() - backward_begin
                detail_begin = time.perf_counter()
                sign_gradient = prepare_sign_gradient(actor, learn_signs)
                torch.cuda.synchronize()
                phase_inspect = time.perf_counter() - detail_begin
                detail_begin = time.perf_counter()
                norm = torch.nn.utils.clip_grad_norm_(selected_parameters, 1.0)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("nonfinite gradient")
                torch.cuda.synchronize()
                phase_clip = time.perf_counter() - detail_begin
                detail_begin = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize()
                phase_optimizer = time.perf_counter() - detail_begin
                detail_begin = time.perf_counter()
                clamp_selected_latents_(actor)
                torch.cuda.synchronize()
                phase_sign_projection = time.perf_counter() - detail_begin
                detail_begin = time.perf_counter()
                for path in paths:
                    actor.linears[path].project_scales_()
                torch.cuda.synchronize()
                phase_scale_projection = time.perf_counter() - detail_begin
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            train_seconds += elapsed
            costs["ce_forward_seconds"] += ce_seconds
            costs["replay_forward_seconds"] += replay_seconds
            costs["shared_backward_optimizer_seconds"] += time.perf_counter() - backward_begin
            exposure["ce_windows"] += 2
            exposure["ce_supported_labels"] += supported
            exposure["replay_windows"] += 1
            exposure["replay_protected_tokens"] += replay_tokens
            selected_stage["updates"] += 1
            selected_stage["sign_optimizer_updates"] += int(learn_signs)
            for key, value in {
                "backward_seconds": phase_backward,
                "selected_gradient_inspection_seconds": phase_inspect,
                "clip_seconds": phase_clip,
                "optimizer_step_seconds": phase_optimizer,
                "selected_sign_projection_seconds": phase_sign_projection,
                "scale_projection_seconds": phase_scale_projection,
            }.items():
                selected_stage[key] += value
            cursor += 1
            in_update = False
            assert_sign_mode(actor, optimizer, learn_signs)
            row = {
                "update": cursor,
                "CE": ce_value,
                "retention_mean": float(replay_loss.detach()),
                "replay_group": replay_group,
                "replay_window": replay_window["id"],
                "replay_protected_tokens": replay_tokens,
                "ce_forward_seconds": ce_seconds,
                "replay_forward_seconds": replay_seconds,
                "loss": float(loss.detach()),
                "supported_labels": supported,
                "learn_signs": learn_signs,
                "selected_sign_gradient": sign_gradient,
                "selected_stage": dict(selected_stage),
                "lr_stage": lr_stage_totals(
                    cursor, costs, exposure, train_seconds, selected_stage, lr_origin
                ),
                "gradient_norm": float(norm),
                "trainer_seconds": train_seconds,
                **stage_totals(cursor, costs, exposure, train_seconds, origin),
            }
            receipts.write(json.dumps(row) + "\n")
            receipts.flush()
            if cursor % 48 == 0:
                print(json.dumps(row), flush=True)
            if cursor % 480 == 0:
                save("RECOVERY")
        checkpoint = (
            save("COMPLETE")
            if cursor % 480
            else Path(json.loads((args.out / "latest.json").read_text())["path"])
        )
    except BaseException as error:
        if not in_update:
            save("INTERRUPTED")
        # A failed optimizer/projection transaction must never be published with
        # the preceding cursor. Previously verified checkpoints remain intact.
        write(
            args.out / "failure.json",
            {
                "error": repr(error),
                "global_cursor_last_completed": cursor,
                "partial_update_not_published": in_update,
                "failed_update_wall_seconds": (
                    time.perf_counter() - transaction_begin if in_update else 0.0
                ),
                "latest_verified_checkpoint": (
                    json.loads((args.out / "latest.json").read_text())
                    if (args.out / "latest.json").exists()
                    else origin["checkpoint"]
                ),
                "cumulative_completed_train_seconds": train_seconds,
                "whole_job_wall_seconds": time.perf_counter() - started,
            },
        )
        raise
    finally:
        receipts.close()
    readout("after-development", dev)
    readout("after-fit", fit)
    assert_sign_mode(actor, optimizer, learn_signs, full=True)
    storage_guard(48 * 1024**2)
    exported = export_selected_sign_gguf(actor, args.out / "model.gguf")
    write(args.out / "export.json", exported)
    final_metric_begin = time.perf_counter()
    final_sign_metrics = selected_sign_metrics(actor)
    final_metric_seconds = time.perf_counter() - final_metric_begin
    write(
        args.out / "result.json",
        {
            "status": "TRAINED_EXPORTED_NOT_NATIVE_QUALIFIED",
            "updates": cursor,
            "trainer_seconds": train_seconds,
            "wall_seconds": time.perf_counter() - started,
            "checkpoint": file_pin(checkpoint),
            "model": exported["output"],
            "reference": file_pin(args.reference),
            "retention": args.retention,
            "optimizer_scope": args.scope,
            "optimized_scale_paths": paths,
            "frozen_unselected_signs_norms_and_initial_scales_audited": True,
            "learn_signs": learn_signs,
            "selected_stage": dict(selected_stage),
            "sign_metrics": final_sign_metrics,
            "lr_stage": lr_stage_totals(
                cursor, costs, exposure, train_seconds, selected_stage, lr_origin
            ),
            "sign_lr": args.sign_lr,
            "final_sign_metric_seconds": final_metric_seconds,
            "costs": costs,
            "exposure": exposure,
            "native_evaluation_masks": native_masks,
            "warm_start": origin["checkpoint"],
            "mature_parent": lr_origin["parent_checkpoint"],
            "stage": stage_totals(cursor, costs, exposure, train_seconds, origin),
            "scope": (
                "mature-O paired LR capacity diagnosis, unchanged all-native-FIT replay/static CE; "
                "no untouched generalization/quality/throughput claim"
            ),
        },
    )


if __name__ == "__main__":
    main()

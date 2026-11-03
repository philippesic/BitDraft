"""Matched static versus current-prefix CE reachability on immutable FIT histories.

Explicitly migrate the same original balanced1920 checkpoint as native replay.
Both arms preserve its all356 native FIT replay bank, optimizer/RNG/sample order,
fixed CE/4 normalization and global19200 horizon. Dynamic admission uses a detached
current actor GOLD prefix before the current depth, never current-depth success.
The union increases effective CE weight; counts/mass are reported explicitly.
Development labels remain full readout only, never loss or reachability inputs.
"""

from __future__ import annotations

# Project path bootstrap precedes optional Torch/project imports.
# ruff: noqa: E402
import argparse
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
import modern_train_protected as protected_helpers
import torch
from modern_actor import (
    build_modern_actor,
    collect_scale_state,
    export_scale_only_gguf,
    file_pin,
    load_scale_state,
)
from modern_actual_data import ActualDataset
from modern_train_protected import (
    ProtectedReplaySchedule,
    assert_excluded_unchanged,
    cpu,
    equal,
    forward_semantics,
    normalized_replay_loss,
    optimizer_scope,
    require,
    validate_original_reference,
    write,
)

from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_qat import shared_round_hard_signs

V1_OBJECTIVE = (
    "Original CE sum/fixed4 from2 ordinary fit windows plus retention*mean "
    "protected-token margin deficit on1 independent group-balanced fit replay; "
    "no retention on ordinary CE windows."
)
WARM_CURSOR = 1920
MARGIN_FLOOR = 0.1


def native_fit_references(fit_windows, original_refs, supported_target_ids):
    """Expand FIT protections from native emission ancestry, never from DEV labels."""
    result = {}
    for window in fit_windows:
        require(window["role"] == "fit", "native replay accepts fit windows only")
        require(window["id"] not in result, "duplicate native fit window")
        original = original_refs[window["id"]]
        require(
            original["role"] == "fit"
            and original["labels"] == window["labels"]
            and original["native_prefix"] == window["native_prefix"],
            "native reference ancestry differs",
        )
        depth = window["max_depth"]
        require(
            len(window["labels"])
            == len(original["margins"])
            == len(original["supported"])
            == depth,
            "native reference depth differs",
        )
        protections = window["native_protections"]
        require(
            [p["proposal_depth"] for p in protections] == list(range(window["native_prefix"]))
            and window["native_prefix"] <= depth,
            "native emitted mask is not a bounded prefix",
        )
        margins = list(original["margins"])
        for p in protections:
            d, label = p["proposal_depth"], p["token"]
            require(
                label == window["labels"][d]
                and p["response_index"] == window["response_label_start"] + d,
                "native protection label does not join verified response",
            )
            require(
                original["supported"][d] is True and label in supported_target_ids,
                "native protected label lacks draft vocabulary support",
            )
            margin = original["margins"][d]
            require(
                isinstance(margin, (int, float)) and math.isfinite(margin),
                "native protected reference margin must be finite",
            )
            margins[d] = max(MARGIN_FLOOR, min(margin, 2.0))
        result[window["id"]] = {
            **original,
            "protected": [d < window["native_prefix"] for d in range(depth)],
            "margins": margins,
        }
    return result


def native_schedule(fit, references):
    schedule = ProtectedReplaySchedule(fit, references, seed=37)
    empty = schedule.contract.pop("groups_without_actor_matched_protections")
    schedule.contract.update(
        {
            "eligibility": (
                "ALL native-emitted FIT successes, including original GPU-mismatched tokens"
            ),
            "groups_without_native_protections": empty,
            "margin_target": "max(0.1,min(original_GPU_reference_margin,2.0))",
            "selection_cursor": "global training cursor; stage starts at1920",
        }
    )
    return schedule


def validate_v1_warm_start(saved, expected, old_schedule, helper_pin):
    """Only the exact all-scale lambda10 balanced1920 checkpoint may migrate."""
    require(
        saved.get("cursor") == WARM_CURSOR and saved.get("status") in ("RECOVERY", "COMPLETE"),
        "warm-start requires original balanced1920 checkpoint",
    )
    old = saved["contract"]
    require(
        all(old["trainer"][k] == helper_pin[k] for k in ("sha256", "bytes")),
        "pinned original protected trainer helper differs",
    )
    for key in (
        "manifest",
        "actor",
        "view_source",
        "optimizer",
        "retention",
        "seed",
        "horizon",
        "batch_windows",
        "margin_cap",
        "optimizer_scope",
        "replay_windows_per_update",
        "context",
        "runtime",
        "precision",
        "reference",
        "reference_freeze",
        "original_trainer",
        "numeric_forward_semantics_sha256",
        "original_reference_objective",
        "ce_schedule",
        "optimized_scale_paths",
        "native_evaluation_masks",
        "readout",
    ):
        require(old.get(key) == expected.get(key), "warm-start contract differs: " + key)
    require(
        old.get("objective") == V1_OBJECTIVE and old.get("protected_schedule") == old_schedule,
        "warm-start requires original actor-intersection bank/objective",
    )
    require(
        old["optimizer_scope"] == "all" and old["retention"] == 10.0 and old["horizon"] == 19200,
        "warm-start requires all-scale lambda10 original horizon",
    )
    groups = saved["optimizer"]["param_groups"]
    require(
        len(groups) == 1
        and len(groups[0]["params"]) == 9
        and groups[0]["lr"] == 1e-5
        and tuple(groups[0]["betas"]) == (0.9, 0.999)
        and groups[0]["eps"] == 1e-8,
        "warm-start optimizer state differs",
    )
    require(
        saved["exposure"]["ce_windows"] == 2 * WARM_CURSOR
        and saved["exposure"]["replay_windows"] == WARM_CURSOR,
        "warm-start exposure/cursor differs",
    )
    require(
        all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
            for v in saved["costs"].values()
        )
        and math.isfinite(saved["trainer_seconds"]),
        "warm-start physical costs invalid",
    )
    for key in ("scales", "rng", "cuda_rng", "python_rng"):
        require(key in saved, "warm-start missing continuation state: " + key)


def original_validation_packet(saved, origin):
    """Validate ancestry without rewinding the actual resumed optimizer/RNG state."""
    return {
        **saved,
        "contract": origin["original_contract"],
        "cursor": origin["global_start_cursor"],
        "exposure": origin["initial_exposure"],
        "costs": origin["initial_costs"],
        "trainer_seconds": origin["initial_trainer_seconds"],
        "status": origin["original_checkpoint_status"],
    }


def stage_totals(cursor, costs, exposure, trainer_seconds, origin):
    """Keep original cumulative work and explicit migration-stage increments."""
    require(cursor >= origin["global_start_cursor"], "stage cursor precedes warm start")
    incremental_costs = {k: costs[k] - origin["initial_costs"][k] for k in costs}
    incremental_exposure = {k: exposure[k] - origin["initial_exposure"][k] for k in exposure}
    seconds = trainer_seconds - origin["initial_trainer_seconds"]
    require(
        seconds >= 0
        and all(v >= 0 for v in incremental_costs.values())
        and all(v >= 0 for v in incremental_exposure.values()),
        "stage accounting moved backwards",
    )
    return {
        "global_start_cursor": origin["global_start_cursor"],
        "stage_updates": cursor - origin["global_start_cursor"],
        "stage_trainer_seconds": seconds,
        "stage_costs": incremental_costs,
        "stage_exposure": incremental_exposure,
    }


def reachability_contract(mode):
    require(mode in ("static", "current-prefix-union"), "unknown CE reachability mode")
    return {
        "mode": mode,
        "static": "immutable window native_supervision_reachable_mask",
        "current_prefix": (
            "detached current actor argmax equals supported gold at EVERY earlier depth; "
            "empty prefix is reached; NEVER gate on current-depth correctness"
        ),
        "union": (
            "supported AND (static OR current-prefix-reached)"
            if mode != "static" else "supported AND static"
        ),
        "training_roles": ["fit"],
        "normalization": "CE sum divided by fixed4 across2 ordinary windows; no adaptive denominator",
        "weighting_caveat": "additional admitted labels increase effective CE mass under fixed4",
        "logging": (
            "stage-cumulative static/extra labels and normalized mass; "
            "current-depth correctness not consulted"
        ),
    }


def ce_reachable_loss(values, classes, window, mode):
    """Return unnormalized CE and detached admission counts for ONE FIT window.

    A gold-forced depth1 forward is the current actor's reached branch only when
    its depth0 proposal equals supported gold. Old native-reachable slots remain
    admitted even when the current actor's own prefix fails. The gate is a Python
    bool from detached proposals, so no gradient flows through selection.
    Static mode deliberately avoids proposal computation and reproduces the
    original ordered CE additions, including its zero-connected starting loss.
    """
    require(window.get("role") == "fit", "CE reachability accepts FIT windows only")
    require(mode in ("static", "current-prefix-union"), "unknown CE reachability mode")
    mask = window["native_supervision_reachable_mask"]
    labels = classes.tolist()
    require(
        values.ndim == 2 and classes.ndim == 1
        and len(mask) == len(labels) == values.shape[0]
        and all(type(m) is bool for m in mask),
        "CE reachability shapes/mask differ",
    )
    require(all(-1 <= cls < values.shape[1] for cls in labels), "invalid CE class")
    proposals = values.detach().argmax(-1).tolist() if mode != "static" else None
    reached_before = True
    ce = values.sum() * 0.0
    base_supported = extra_supported = 0
    for depth, cls in enumerate(labels):
        old = cls >= 0 and mask[depth]
        admitted = old or (mode != "static" and cls >= 0 and reached_before)
        if admitted:
            ce = ce + torch.nn.functional.cross_entropy(
                values[depth : depth + 1], classes[depth : depth + 1]
            )
            base_supported += int(old)
            extra_supported += int(not old)
        if proposals is not None:
            # Update only AFTER admitting this depth: current-depth correctness
            # cannot decide whether this very depth receives supervised CE.
            reached_before = reached_before and cls >= 0 and proposals[depth] == cls
    return ce, {
        "static_supported_labels": base_supported,
        "extra_supported_labels": extra_supported,
        "effective_supported_labels": base_supported + extra_supported,
        "static_supported_mass": base_supported / 4.0,
        "extra_supported_mass": extra_supported / 4.0,
        "effective_supported_mass": (base_supported + extra_supported) / 4.0,
    }


def reachability_totals(saved=None):
    if saved is None:
        return {
            "updates": 0,
            "static_supported_labels": 0,
            "extra_supported_labels": 0,
            "effective_supported_labels": 0,
            "static_supported_mass": 0.0,
            "extra_supported_mass": 0.0,
            "effective_supported_mass": 0.0,
        }
    result = dict(saved)
    require(set(result) == set(reachability_totals()), "reachability accounting fields differ")
    require(
        all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in result.values())
        and all(
            type(result[k]) is int for k in (
                "updates", "static_supported_labels", "extra_supported_labels",
                "effective_supported_labels",
            )
        )
        and result["effective_supported_labels"]
        == result["static_supported_labels"] + result["extra_supported_labels"]
        and all(
            result[k + "_mass"] == result[k + "_labels"] / 4.0
            for k in ("static_supported", "extra_supported", "effective_supported")
        ),
        "invalid reachability accounting",
    )
    return result


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
    parser.add_argument("--steps", type=int, default=3840)
    parser.add_argument("--ce-reachability", choices=["static", "current-prefix-union"], required=True)
    parser.add_argument(
        "--native-replay-trainer", type=Path,
        default=Path(__file__).with_name("modern_train_native_replay.py"),
    )
    parser.add_argument("--retention", type=float, choices=[10.0], default=10.0)
    continuation = parser.add_mutually_exclusive_group(required=True)
    continuation.add_argument("--warm-start", type=Path)
    continuation.add_argument("--resume", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if not args.allow_cuda or not 0 < args.steps <= 19200 or args.retention < 0:
        raise ValueError("explicit sole GPU owner / known horizon required")
    args.out.mkdir(parents=True, exist_ok=False)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

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
        for filesystem in {PROJECT, args.out}:
            if os.statvfs(filesystem).f_bavail * os.statvfs(filesystem).f_frsize < 2 * 1024**3:
                raise RuntimeError("project/output storage floor reached: " + str(filesystem))
        if os.statvfs("/mnt/c").f_bavail * os.statvfs("/mnt/c").f_frsize < 2 * 1024**3:
            raise RuntimeError("Windows storage floor reached")

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
        scale_only=True,
    )
    absolute = actor.drafter.d2t + torch.arange(50000, device=device)
    reverse = torch.full((248320,), -1, device=device, dtype=torch.long)
    reverse[absolute] = torch.arange(50000, device=device)
    contract = {
        "manifest": file_pin(args.manifest),
        "actor": actor.contract,
        "trainer": file_pin(__file__),
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
        value = {"rows": rows, "wall_seconds": time.perf_counter() - begin}
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
    native_replay_source = Path(__file__).with_name("modern_train_native_replay.py")
    native_replay_pin = file_pin(native_replay_source)
    require(
        file_pin(args.native_replay_trainer) == native_replay_pin,
        "native replay source differs from explicitly pinned matched control",
    )
    require(
        forward_semantics(native_replay_source.read_text())
        == forward_semantics(Path(__file__).read_text()),
        "native replay forward semantics changed",
    )
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
    paths, selected_parameters, excluded = optimizer_scope(actor, args.scope)
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
    optimizer = torch.optim.Adam(selected_parameters, lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
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
    source_checkpoint = args.resume or args.warm_start
    source_pin = file_pin(source_checkpoint)
    saved = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    require(
        file_pin(source_checkpoint) == source_pin, "continuation checkpoint changed while reading"
    )
    if args.warm_start:
        validate_v1_warm_start(saved, contract, old_schedule, helper_pin)
        origin = {
            "checkpoint": source_pin,
            "original_contract": saved["contract"],
            "global_start_cursor": WARM_CURSOR,
            "initial_costs": dict(saved["costs"]),
            "initial_exposure": dict(saved["exposure"]),
            "initial_trainer_seconds": saved["trainer_seconds"],
            "original_checkpoint_status": saved["status"],
        }
    else:
        require(
            "warm_start" in saved["contract"], "ordinary resume must be from this migrated stage"
        )
        origin = saved["contract"]["warm_start"]
        require(
            file_pin(origin["checkpoint"]["path"]) == origin["checkpoint"],
            "original warm-start checkpoint ancestry changed",
        )
        original_packet = original_validation_packet(saved, origin)
        # Validate the original contract/pins against the unchanged v1 helper. The
        # current optimizer still has the same all-scale parameter-group schema.
        validate_v1_warm_start(original_packet, contract, old_schedule, helper_pin)
    contract.update(
        {
            "objective": (
                "CE2windows/fixed4 with explicitly selected static/current-prefix-union "
                "reachability plus lambda10 mean margin-deficit replay "
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
            "stage_kind": "declared_CE_reachability_comparison_from_balanced1920_v1",
            "ce_reachability": reachability_contract(args.ce_reachability),
            "native_replay_trainer_source": native_replay_pin,
            "unchanged_control": (
                "static numeric trajectory must match original all-native-FIT replay "
                "from identical balanced1920 source state; contracts/logging differ explicitly"
            ),
        }
    )
    if args.resume:
        require(saved["contract"] == contract, "resume contract differs")
        require("reachability_exposure" in saved, "resume missing reachability accounting")
    reachability = (
        reachability_totals(saved.get("reachability_exposure"))
        if args.resume else reachability_totals()
    )
    require(
        reachability["updates"] == saved["cursor"] - origin["global_start_cursor"],
        "reachability accounting/global cursor differs",
    )
    write(args.out / "freeze.json", contract)
    write(
        args.out / "continuation.json",
        {
            "mode": "warm_start" if args.warm_start else "resume",
            "checkpoint": source_pin,
            "loaded_global_cursor": saved["cursor"],
            "requested_global_cursor": args.steps,
            "original_protected_source": helper_pin,
        },
    )
    load_scale_state(actor, saved["scales"])
    optimizer.load_state_dict(saved["optimizer"])
    cursor, train_seconds = saved["cursor"], saved["trainer_seconds"]
    costs, exposure = dict(saved["costs"]), dict(saved["exposure"])
    assert_excluded_unchanged(actor, excluded)
    torch.set_rng_state(saved["rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    require(
        WARM_CURSOR <= cursor < args.steps <= 19200,
        "continuation must extend global cursor within unchanged19200 horizon",
    )

    def save(status):
        assert_excluded_unchanged(actor, excluded)
        path = args.out / f"checkpoint-{cursor:08d}.pt"
        if path.exists():
            return path
        state = {
            "contract": contract,
            "scales": collect_scale_state(actor),
            "optimizer": cpu(optimizer.state_dict()),
            "cursor": cursor,
            "trainer_seconds": train_seconds,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "costs": dict(costs),
            "exposure": dict(exposure),
            "status": status,
            "reachability_exposure": dict(reachability),
            "stage": stage_totals(cursor, costs, exposure, train_seconds, origin),
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = torch.load(tmp, map_location="cpu", weights_only=False)
        if not equal(state, loaded):
            raise ValueError("checkpoint roundtrip differs")
        os.replace(tmp, path)
        write(args.out / "latest.json", file_pin(path))
        return path

    receipts = (args.out / "updates.jsonl").open("x")
    try:
        while cursor < args.steps:
            guard()
            actor.drafter.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            loss = 0.0
            ce_value = 0.0
            supported = 0
            ce_admission = {k: 0.0 if k.endswith("_mass") else 0 for k in reachability if k != "updates"}
            with shared_round_hard_signs(actor.linears):
                for offset in range(2):
                    w = fit[order[(2 * cursor + offset) % len(order)]]
                    values = forward(w)
                    classes = reverse[torch.tensor(w["labels"], device=device)]
                    ce, admission = ce_reachable_loss(values, classes, w, args.ce_reachability)
                    supported += admission["effective_supported_labels"]
                    for key, value in admission.items():
                        ce_admission[key] += value
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
                norm = torch.nn.utils.clip_grad_norm_(selected_parameters, 1.0)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                for path in paths:
                    actor.linears[path].project_scales_()
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
            reachability["updates"] += 1
            for key, value in ce_admission.items():
                reachability[key] += value
            cursor += 1
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
                "ce_reachability": args.ce_reachability,
                "ce_admission": ce_admission,
                "reachability_exposure": dict(reachability),
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
    except BaseException:
        save("INTERRUPTED")
        raise
    finally:
        receipts.close()
    readout("after-development", dev)
    readout("after-fit", fit)
    assert_excluded_unchanged(actor, excluded)
    exported = export_scale_only_gguf(actor, args.out / "model.gguf")
    write(args.out / "export.json", exported)
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
            "excluded_scale_offsets_unchanged": True,
            "costs": costs,
            "exposure": exposure,
            "reachability_exposure": dict(reachability),
            "ce_reachability": reachability_contract(args.ce_reachability),
            "native_evaluation_masks": native_masks,
            "warm_start": origin["checkpoint"],
            "stage": stage_totals(cursor, costs, exposure, train_seconds, origin),
            "scope": (
                "declared static/current-prefix CE comparison with unchanged all-native-FIT replay; "
                "no untouched generalization/quality/throughput claim"
            ),
        },
    )


if __name__ == "__main__":
    main()

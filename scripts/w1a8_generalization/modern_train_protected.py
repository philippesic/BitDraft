"""Separate group-balanced protected replay training on immutable actual histories.

Native source lineage, including the original bridge behavior, is preserved.
The separately repaired runtime is not silently substituted into this comparison.
"""

from __future__ import annotations

# Project path bootstrap precedes optional Torch/project imports.
# ruff: noqa: E402
import argparse
import ast
import hashlib
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
import torch
from modern_actor import (
    build_modern_actor,
    collect_scale_state,
    export_scale_only_gguf,
    file_pin,
    load_scale_state,
)
from modern_actual_data import ActualDataset

from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_qat import shared_round_hard_signs


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu(child) for child in value)
    return value


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def require(condition, message):
    if not condition:
        raise ValueError(message)


class ProtectedReplaySchedule:
    """Cursor-only, fit-only balanced replay; no global RNG or development choices."""

    def __init__(self, fit_windows, references, seed=37):
        groups = {}
        all_groups = set()
        seen = set()
        for window in fit_windows:
            require(window["role"] == "fit", "protected replay accepts fit windows only")
            require(window["id"] not in seen, "duplicate fit window")
            seen.add(window["id"])
            group = window["group_id"]
            all_groups.add(group)
            reference = references[window["id"]]
            require(
                reference["role"] == "fit" and reference["labels"] == window["labels"],
                "replay reference/window mismatch",
            )
            if any(reference["protected"]):
                groups.setdefault(group, []).append(window)
        require(groups, "no fit windows have frozen parent protections")
        order = sorted(groups)
        random.Random(seed).shuffle(order)
        self.groups = []
        self.window_ids = []
        for group in order:
            windows = sorted(groups[group], key=lambda w: w["id"])
            mixed_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "little"
            )
            random.Random(mixed_seed).shuffle(windows)
            self.groups.append((group, windows))
            self.window_ids.append({"group_id": group, "windows": [w["id"] for w in windows]})
        self.contract = {
            "version": "fit_group_round_robin_v1",
            "seed": seed,
            "group_order": order,
            "within_group_order": self.window_ids,
            "groups_without_actor_matched_protections": sorted(all_groups - set(groups)),
            "selection": "group=cursor%G; window=(cursor//G)%N_group; no dev labels",
        }

    def at(self, cursor):
        require(type(cursor) is int and cursor >= 0, "nonnegative replay cursor required")
        group_index, group_epoch = cursor % len(self.groups), cursor // len(self.groups)
        group, windows = self.groups[group_index]
        return group, windows[group_epoch % len(windows)]


def normalized_replay_loss(values, classes, reference, cap=2.0):
    """Mean squared frozen-parent margin deficit over this replay's protected tokens."""
    require(values.ndim == 2 and classes.shape == (values.shape[0],), "replay logit/label shape")
    require(
        len(reference["protected"]) == len(reference["margins"]) == len(classes),
        "replay reference depth differs",
    )
    terms = []
    for depth, protected in enumerate(reference["protected"]):
        require(type(protected) is bool, "replay protection flags must be booleans")
        if not protected:
            continue
        cls = int(classes[depth])
        margin_target = reference["margins"][depth]
        require(
            0 <= cls < values.shape[1]
            and isinstance(margin_target, (int, float))
            and math.isfinite(margin_target)
            and margin_target > 0,
            "protected label requires supported class and positive finite parent margin",
        )
        wrong = values[depth].scatter(0, classes[depth : depth + 1], -torch.inf).max()
        margin = values[depth, cls] - wrong
        terms.append(torch.relu(margin.new_tensor(min(margin_target, cap)) - margin).square())
    require(terms, "replay window has no protected tokens")
    return torch.stack(terms).mean(), len(terms)


def optimizer_scope(actor, scope):
    require(scope in ("all", "head"), "unknown scale optimizer scope")
    all_paths = list(actor.linears)
    require(len(all_paths) == 9 and "lm_head" in all_paths, "nine-scale actor required")
    require(
        {id(p) for p in actor.scale_parameters()}
        == {id(actor.linears[path].scale_offset) for path in all_paths},
        "actor scale integrity differs",
    )
    paths = all_paths if scope == "all" else ["lm_head"]
    selected = tuple(actor.linears[path].scale_offset for path in paths)
    excluded = {
        path: actor.linears[path].scale_offset.detach().cpu().clone()
        for path in all_paths
        if path not in paths
    }
    return paths, selected, excluded


def assert_excluded_unchanged(actor, excluded):
    for path, initial in excluded.items():
        require(
            torch.equal(actor.linears[path].scale_offset.detach().cpu(), initial),
            "excluded scale changed: " + path,
        )


def forward_semantics(source):
    """Pin unchanged numeric forward/readout functions independently of training objective."""
    tree = ast.parse(source)
    result = {}
    for name in ("forward", "inspect"):
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        require(len(matches) == 1, "expected unique original numeric function: " + name)
        result[name] = ast.dump(matches[0], include_attributes=False)
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def validate_original_reference(
    reference, reference_contract, contract, windows, original_source, current_source
):
    original_pin = file_pin(original_source)
    expected = reference_contract["trainer"]
    require(
        all(original_pin[k] == expected[k] for k in ("sha256", "bytes")),
        "original reference trainer source changed",
    )
    require(
        forward_semantics(Path(original_source).read_text())
        == forward_semantics(Path(current_source).read_text()),
        "reference numeric forward semantics differ",
    )
    for key in (
        "manifest",
        "actor",
        "view_source",
        "optimizer",
        "seed",
        "batch_windows",
        "margin_cap",
        "horizon",
        "context",
        "runtime",
        "precision",
    ):
        require(
            reference_contract[key] == contract[key], "original reference contract differs: " + key
        )
    refs = {row["id"]: row for row in reference["rows"]}
    require(
        len(refs) == len(reference["rows"]) and set(refs) == {w["id"] for w in windows},
        "reference windows differ",
    )
    for window in windows:
        row = refs[window["id"]]
        require(
            row["role"] == window["role"]
            and row["labels"] == window["labels"]
            and row["native_prefix"] == window["native_prefix"]
            and len(row["protected"]) == len(row["margins"]) == window["max_depth"],
            "reference label/protection metadata differs",
        )
        require(
            row["protected"]
            == [
                d < min(row["native_prefix"], row["actor_prefix"])
                for d in range(window["max_depth"])
            ],
            "reference protection intersection differs",
        )
    return refs


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["train"], default="train")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--original-trainer", type=Path, default=Path(__file__).with_name("modern_train.py")
    )
    parser.add_argument("--scope", choices=["all", "head"], default="all")
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--retention", type=float, choices=[0.0, 10.0], default=0.0)
    parser.add_argument("--resume", type=Path)
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
    schedule = ProtectedReplaySchedule(fit, refs, seed=37)
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
    write(args.out / "freeze.json", contract)
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
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("resume contract differs")
        load_scale_state(actor, saved["scales"])
        optimizer.load_state_dict(saved["optimizer"])
        cursor = saved["cursor"]
        train_seconds = saved["trainer_seconds"]
        costs, exposure = saved["costs"], saved["exposure"]
        assert_excluded_unchanged(actor, excluded)
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        if cursor >= args.steps:
            raise ValueError("resume must extend the saved update cursor")

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
                    replay_values, replay_classes, refs[replay_window["id"]], cap=2.0
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
                "gradient_norm": float(norm),
                "trainer_seconds": train_seconds,
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
            "native_evaluation_masks": native_masks,
            "scope": (
                "actual-state protected replay pilot; "
                "no untouched generalization/quality/throughput claim"
            ),
        },
    )


if __name__ == "__main__":
    main()

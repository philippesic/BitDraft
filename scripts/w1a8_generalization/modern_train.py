"""Matched compact training on immutable actual-parent serving histories.

Native source lineage, including the original bridge behavior, is preserved.
The separately repaired runtime is not silently substituted into this comparison.
"""

from __future__ import annotations

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["reference", "train"], required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--retention", type=float, default=0.0)
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
        if os.statvfs(args.out).f_bavail * os.statvfs(args.out).f_frsize < 2 * 1024**3:
            raise RuntimeError("storage floor reached")
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
        "objective": "sumCE / fixed depth2 on original native-reachable goldprefix; unsupported labels retained without CE. Same-actor parent margin hinge on native+actor correctprefix intersection.",
        "context": "detached current-actor KV rebuilt from exact observed rawfeature lineage; own FC/prenorm in goldprefix unroll",
        "runtime": "original immutable packed-runtime03; bridgefix overlay deliberately separate",
        "precision": "F32Torch/A8, F16KV, TF32disabled",
    }
    write(args.out / "freeze.json", contract)

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
    if args.phase == "reference":
        ref = readout("reference", data.windows)
        write(
            args.out / "result.json",
            {
                "status": "REFERENCE_COMPLETE",
                "contract": contract,
                "windows": len(ref["rows"]),
                "wall_seconds": time.perf_counter() - started,
                "training_updates": 0,
            },
        )
        return
    if not args.reference:
        raise ValueError("frozen actual-parent reference required")
    reference = json.loads(args.reference.read_text())
    refs = {r["id"]: r for r in reference["rows"]}
    if set(refs) != {w["id"] for w in data.windows}:
        raise ValueError("reference windows differ")
    reference_contract = json.loads(args.reference.with_name("freeze.json").read_text())
    for key in [
        "manifest",
        "actor",
        "trainer",
        "view_source",
        "optimizer",
        "seed",
        "batch_windows",
        "margin_cap",
        "objective",
        "context",
        "runtime",
        "precision",
    ]:
        if reference_contract[key] != contract[key]:
            raise ValueError("reference contract differs: " + key)
    optimizer = torch.optim.Adam(actor.scale_parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    cursor = 0
    train_seconds = 0.0
    order = list(range(len(fit)))
    random.Random(37).shuffle(order)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["contract"] != contract:
            raise ValueError("resume contract differs")
        load_scale_state(actor, saved["scales"])
        optimizer.load_state_dict(saved["optimizer"])
        cursor = saved["cursor"]
        train_seconds = saved["trainer_seconds"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        if cursor >= args.steps:
            raise ValueError("resume must extend the saved update cursor")

    def save(status):
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
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            loss = 0.0
            ce_value = 0.0
            ret_value = 0.0
            supported = 0
            with shared_round_hard_signs(actor.linears):
                for offset in range(2):
                    w = fit[order[(2 * cursor + offset) % len(order)]]
                    values = forward(w)
                    classes = reverse[torch.tensor(w["labels"], device=device)]
                    reference_row = refs[w["id"]]
                    ce = values.sum() * 0.0
                    ret = values.sum() * 0.0
                    for d, cls in enumerate(classes.tolist()):
                        if cls >= 0 and w["native_supervision_reachable_mask"][d]:
                            ce = ce + torch.nn.functional.cross_entropy(
                                values[d : d + 1], classes[d : d + 1]
                            )
                            supported += 1
                        if reference_row["protected"][d]:
                            wrong = values[d].scatter(0, classes[d : d + 1], -torch.inf).max()
                            margin = values[d, cls] - wrong
                            ret = (
                                ret
                                + torch.relu(
                                    margin.new_tensor(min(reference_row["margins"][d], 2.0))
                                    - margin
                                ).square()
                            )
                    loss = loss + (ce + args.retention * ret) / 4.0
                    ce_value += float(ce.detach()) / 4.0
                    ret_value += float(ret.detach()) / 4.0
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(actor.scale_parameters(), 1.0)
                if not bool(torch.isfinite(norm)):
                    raise ValueError("nonfinite gradient")
                optimizer.step()
                for module in actor.linears.values():
                    module.project_scales_()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            train_seconds += elapsed
            cursor += 1
            row = {
                "update": cursor,
                "CE": ce_value,
                "retention": ret_value,
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
    readout("after-fit", [w for i, w in enumerate(fit) if i % 8 == 0])
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
            "scope": "actual-state modern pilot; no untouched generalization/quality/throughput claim",
        },
    )


if __name__ == "__main__":
    main()

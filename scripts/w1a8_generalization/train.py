"""Matched historical DSpark coverage/retention experiment using retained training.

This explicit new adapter reuses the verified parent loader, complete Adam state,
native-A8 forward/backward, optimizer transaction, guards and native exporter.
It does not admit modern Qwen3.5, alter old campaigns or run backlog screens.
"""
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from losses import parent_reference, retention_penalty


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def authenticate(ref):
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("input changed: " + ref["path"])


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bootstrap(spec):
    authenticate(spec["base_spec"])
    project = Path(spec["project"])
    source = Path(spec["retained_source"])
    sys.path[1:1] = [str(source / "scripts"), str(project / "scripts")]
    from shortlist_pilots.objectives.runtime import configure_imports
    configure_imports()
    from training_system.io import authenticate as original_authenticate
    base = read(spec["base_spec"]["path"])
    for ref in base["sources"].values():
        original_authenticate(ref)
    if base["scientific_protocol"] != spec["scientific_protocol"]:
        raise ValueError("historical target/verifier protocol changed")
    for ref in spec["sources"]:
        authenticate(ref)
    return base


def train(args):
    spec = read(args.spec)
    if spec["schema"] != "w1a8_generalization_v1" or not args.allow_cuda:
        raise ValueError("explicit historical generalization spec and sole GPU operator required")
    if spec["policy"] not in ("legacy_errors", "occupancy") or spec["retention_coefficient"] < 0:
        raise ValueError("known isolated data/objective treatment required")
    if not 0 < args.updates <= spec["horizon_updates"]:
        raise ValueError("positive exposure within frozen horizon required")
    base = bootstrap(spec)
    import torch
    from data import block
    from training_system.model_cache import load, validate_frozen
    from training_system.step import train_batch, measure_norms
    from shortlist_pilots.binary.parent_loader import equal, copy_named_state
    from shortlist_pilots.objectives.runtime import native_export
    from checkpointing import BoundaryStop, publish_full_state
    from joint_tools import joint_parameters, binary_buffers
    from train_nine_model_qat import tensor_batch
    from w1a1_eagle.block_training import _block_rng_state, _restore_block_rng
    from w1a1_eagle.block_qat import block_supported_mask
    from eal_loss import make_eal_loss
    from common import guard
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < spec["required_free_bytes"]:
        raise ValueError("insufficient ext4 capacity for checkpoint, export and recovery floor")
    phases = {}
    started = time.perf_counter()
    authenticate(spec["manifest"])
    authenticate(spec["order"])
    manifest = read(spec["manifest"]["path"])
    windows = manifest["windows"]
    fit = [w for w in windows if w["role"] == "fit"]
    dev = [w for w in windows if w["role"] == "development"]
    if not fit or not dev or {w["group_id"] for w in fit} & {w["group_id"] for w in dev}:
        raise ValueError("nonempty group-disjoint fit/development required")
    pairs = read(spec["order"]["path"])["pairs"]
    if (not pairs or any(len(p) != 2 or any(type(i) is not int or not 0 <= i < len(fit) for i in p) for p in pairs)
            or sorted(i for p in pairs for i in p) != list(range(len(fit)))):
        raise ValueError("complete exactly-once two-block fit ordering required")
    # Authenticate shared feature payloads once; each view validates its own index
    # and token/mask ancestry before producing a batch.
    for ref in {w["features"]["path"]: w["features"] for w in windows}.values():
        authenticate(ref)
    phases["data_authentication"] = time.perf_counter() - started
    setup = output / "setup"
    setup.mkdir()
    started = time.perf_counter()
    model, optimizer, baseline, metadata = load(setup, base)
    phases["parent_restore"] = time.perf_counter() - started
    params = joint_parameters(model)
    producer = read(base["model_cache_producer"]["path"])
    frozen = metadata["frozen_state"]
    reference = {n: v.to(model.token_embd.device) for n, v in metadata["initial_norms"].items()}
    local, train_seconds = 0, 0.
    refs = []
    spec_contract = dict(spec)
    if args.resume:
        resume = read(args.resume)
        authenticate(resume)
        from training_system.checkpoints import load_private
        saved = load_private(resume)
        if saved["schema"] != "w1a8_generalization_checkpoint_v1" or saved["contract"] != spec_contract:
            raise ValueError("resume source/data/objective/initializer contract differs")
        if saved["config"] != asdict(model.config) or not equal(saved["frozen_state"], frozen):
            raise ValueError("resume config or ordinary frozen tensors differ")
        copy_named_state(model, saved, joint_parameters, binary_buffers)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_block_rng(saved["rng"], model.token_embd.device)
        if not equal(optimizer.state_dict(), saved["optimizer"]) or not equal(_block_rng_state(model.token_embd.device), saved["rng"]):
            raise ValueError("resume Adam/RNG restoration differs")
        local, train_seconds, refs = saved["local_updates"], saved["additional_TRAIN_seconds"], saved["parent_reference"]
        if (saved["pair_index"] != local % len(pairs) or local >= args.updates
                or baseline.measure(model) != saved["sign_displacement"]
                or any(int(s["step"]) != 3504 + local for s in optimizer.state.values())):
            raise ValueError("resume cursor/sign/reference/optimizer clock differs")
        del saved
        gc.collect()
    started = time.perf_counter()
    if not refs:
        rng = _block_rng_state(model.token_embd.device)
        model.eval()
        with torch.no_grad():
            for index, window in enumerate(fit):
                guard()
                batch = tensor_batch(block(window, spec["manifest"]))
                result = model(batch)
                refs.append(parent_reference(result.logits, batch.labels, cap=spec["margin_cap"]))
                del batch, result
                if index % 48 == 0:
                    print(json.dumps({"stage": "parent_reference", "windows": index + 1, "total": len(fit)}), flush=True)
        _restore_block_rng(rng, model.token_embd.device)
        model.train()
    if len(refs) != len(fit):
        raise ValueError("complete TRAIN-only parent margin cache required")
    phases["parent_reference"] = time.perf_counter() - started
    write(output / "parent-reference.json", {"references": refs, "parent": base["parent"], "fit_only": True,
                                           "rng_restored": True, "cap": spec["margin_cap"]})
    def readout(name):
        started = time.perf_counter()
        rng = _block_rng_state(model.token_embd.device)
        training = model.training
        model.eval()
        rows = []
        with torch.no_grad():
            for index, window in enumerate(windows):
                if window["role"] == "fit" and index % spec["fit_readout_stride"]:
                    continue
                guard()
                batch = tensor_batch(block(window, spec["manifest"]))
                result = model(batch)
                labels = batch.labels[:5].to(result.logits.device)
                predictions = result.logits[:5].argmax(-1)
                prefix = int(predictions.eq(labels).long().cumprod(0).sum())
                rows.append({"index": index, "group_id": window["group_id"], "domain": window["domain"],
                             "role": window["role"], "anchor": window["anchor"], "prefix5": prefix,
                             "serving_parent_prefix5": window["accepted_drafts"],
                             "predictions": predictions.cpu().tolist(), "labels": labels.cpu().tolist()})
                del batch, result
        model.train(training)
        _restore_block_rng(rng, model.token_embd.device)
        value = {"scope": "approximate native-A8 actor, reused TRAIN ancestry; not untouched native throughput",
                 "local_updates": local, "rows": rows, "rng_restored": True,
                 "wall_seconds": time.perf_counter() - started}
        write(output / name, value)
        return value["wall_seconds"]
    phases["before_readout"] = readout("before.json")
    write(output / "start.json", {"spec": spec, "local_updates": local, "hardware": read(setup / "template-loader.json")["hardware"],
                                  "exact_parent_restore": True, "fit_windows": len(fit), "dev_windows": len(dev),
                                  "historical_track": "Qwen3-4B F16 / DSpark native W1A8; not modern Qwen3.5"})
    control = make_eal_loss(block_supported_mask)
    current = {}
    # Normalize round occupancy over the fixed fit population. Control sampling
    # remains exactly its historical equal-window objective.
    normalizer = sum(w.get("occupancy_weight", 1.) for w in fit) / len(fit)
    def objective(result, batch, config, *, teacher_callback=None):
        value, counters = control(result, batch, config, teacher_callback=teacher_callback)
        index = current[id(batch)]
        penalty = retention_penalty(result.logits, batch.labels, refs[index])
        weight = fit[index].get("occupancy_weight", 1.) / normalizer if spec["policy"] == "occupancy" else 1.
        value = weight * (value + config.cross_entropy_weight * spec["retention_coefficient"] * penalty)
        return value, {**counters, "retention_penalty": float(penalty.detach()), "occupancy_weight": weight}
    def no_hidden(student, teacher, supported):
        if teacher is not None or not bool(supported.all()):
            raise ValueError("complete captured-prefix rows required")
        return student.new_zeros(()), {"supported_rows": 7, "normalized_error_sum": 0.}
    status = "EXPOSURE_COMPLETE"
    data_seconds, step_seconds, diagnostics_seconds = 0., 0., 0.
    started = time.perf_counter()
    pause_file = Path(spec["project"]).parent / ".config/binary-eagle-decoding/gpu-control.json"
    with BoundaryStop([output / "STOP", Path(spec["stop_path"])]) as stop:
        with (output / "updates.jsonl").open("x") as log:
            while local < args.updates and not args.readout_only:
                pause = read(pause_file).get("rtx5080", {}).get("pause_requested", False)
                if stop.requested() or pause:
                    status = "PAUSED"
                    break
                guard()
                t0 = time.perf_counter()
                indices = pairs[local % len(pairs)]
                batches = [tensor_batch(block(fit[i], spec["manifest"])) for i in indices]
                current = {id(b): i for b, i in zip(batches, indices)}
                data_seconds += time.perf_counter() - t0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                metrics, _ = train_batch(model, optimizer, batches, hidden_targets=[None, None], hidden_weight=0.,
                                         return_outputs=False, loss_fn=objective, hidden_error_fn=no_hidden)
                torch.cuda.synchronize()
                duration = time.perf_counter() - t0
                step_seconds += duration
                t0 = time.perf_counter()
                norms, signs = measure_norms(model, reference), baseline.measure(model)
                torch.cuda.synchronize()
                diagnostic = time.perf_counter() - t0
                diagnostics_seconds += diagnostic
                train_seconds += duration + diagnostic
                local += 1
                record = {"local_updates": local, "indices": indices, "TRAIN_seconds": train_seconds,
                          "step_seconds": duration, "diagnostics_seconds": diagnostic, "metrics": metrics,
                          "norms": norms, "sign_displacement": signs}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                del batches
                if local % 48 == 0:
                    print(json.dumps({"stage": "training", "updates": local, "TRAIN_seconds": train_seconds}), flush=True)
                if signs["fraction"] > .02 or norms["max_relative_l2"] > .1 or norms["max_absolute_change"] > .5:
                    status = "ORIGINAL_GUARD_STOP"
                    break
                if local in spec["readout_milestones"] and local < args.updates:
                    phases["readout_%d" % local] = readout("readout-%08d.json" % local)
                if local % 240 == 0:
                    gc.collect()
    phases.update(training_loop_wall=time.perf_counter() - started, data_seconds=data_seconds,
                  optimizer_seconds=step_seconds, diagnostics_seconds=diagnostics_seconds)
    phases["after_readout"] = readout("after.json")
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if validate_frozen(model, producer["frozen_actor"]["tensors"]) != frozen:
        raise ValueError("frozen ordinary native operands changed")
    checkpoint = exported = None
    if local and not args.readout_only:
        state = {"schema": "w1a8_generalization_checkpoint_v1", "contract": spec_contract,
                 "local_updates": local, "step": 3504 + local, "pair_index": local % len(pairs),
                 "parameters": {n: p.detach() for n, p in params.items()}, "binary_buffers": binary_buffers(model),
                 "optimizer": optimizer.state_dict(), "rng": _block_rng_state(model.token_embd.device),
                 "config": asdict(model.config), "initial_norms": metadata["initial_norms"],
                 "initial_sign_hashes": metadata["initial_sign_hashes"], "frozen_state": frozen,
                 "TRAIN_seconds": metadata["TRAIN_seconds"] + train_seconds, "additional_TRAIN_seconds": train_seconds,
                 "parent_reference": refs, "sign_displacement": baseline.measure(model), "status": status}
        def verify(saved):
            if not equal(saved, state):
                raise ValueError("checkpoint round trip differs")
        t0 = time.perf_counter()
        checkpoint = publish_full_state(output / ("checkpoint-%08d.pt" % local), state, verify)
        phases["checkpoint"] = time.perf_counter() - t0
        write(output / "checkpoint.json", checkpoint)
        del state
        gc.collect()
        t0 = time.perf_counter()
        exported = native_export(model, output / "model.gguf", parent=base["parent"])
        phases["export"] = time.perf_counter() - t0
    result = {"status": status, "local_updates": local, "additional_TRAIN_seconds": train_seconds,
              "checkpoint": checkpoint, "export": exported, "phases": phases, "spec": pin(args.spec),
              "parent": base["parent"], "whole_parent_unseen_proven": False,
              "native_evaluation_required": True, "backlog_screening_started": False}
    write(output / "finish.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--resume", type=Path, help="JSON pin of a verified exact-contract checkpoint")
    parser.add_argument("--readout-only", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    try:
        train(args)
    except BaseException as error:
        if args.output.exists() and not (args.output / "failure.json").exists():
            write(args.output / "failure.json", {"exception": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()

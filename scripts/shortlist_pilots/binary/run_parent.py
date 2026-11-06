#!/usr/bin/env python3
"""Exact parent3504/TRAIN486 binary sanity pilot using the campaign source.

Run only on the owner's admitted lane. Imports deliberately select the frozen
campaign source before importing any model implementation or binary adapter.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import copy
import gc
import json
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from shortlist_pilots.objectives import runtime
runtime.configure_imports()
# The namespace package is now bound. Give the frozen implementation priority
# over the working repository's scripts/src before any model/helper imports.
if str(ROOT / "scripts") in sys.path:
    sys.path.remove(str(ROOT / "scripts"))

NATIVE_TECHNIQUES = {"control", "selective-gradient", "elastic-gradient", "trust-gradient",
                     "calibration-a8", "calibration-weight-mse"}
CAPACITY_TECHNIQUES = {"mos", "mos-static", "double-binary", "ternary-a8"}


def anchor_metrics(model, batches):
    """Repeated TRAIN anchors; teacher-forced D5 prefixes are not native acceptance."""
    import torch
    from shortlist_pilots.binary.readouts import preserve_readout_state
    rows, logits = [], []
    with preserve_readout_state(model), torch.no_grad():
        for batch in batches:
            output = model(batch).logits[:5]
            labels = batch.labels[:5].to(output.device)
            pred = output.argmax(-1)
            hits = pred == labels
            prefix = int(hits.int().cumprod(0).sum())
            top = output.topk(2, dim=-1)
            best_wrong = torch.where(top.indices[:, 0] == labels, top.values[:, 1], top.values[:, 0])
            margins = output[torch.arange(5, device=output.device), labels]-best_wrong
            rows.append({"predicted_ids": pred.cpu().tolist(), "label_ids": labels.cpu().tolist(),
                         "captured_prefix_depth5": prefix, "label_minus_best_wrong": margins.cpu().tolist(),
                         "mean_ce5": float(torch.nn.functional.cross_entropy(output, labels))})
            logits.append(output.cpu())
    return rows, logits


def calibration_update(model, optimizer, batches, projection, reference, metric):
    import torch
    from shortlist_pilots.binary.calibration import capture_inputs, reconstruction_loss
    module = model.binary_linears()[projection]
    inputs = capture_inputs(model, batches, projection)
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for x in inputs:
        loss = reconstruction_loss(module, x, reference, metric)/len(inputs)
        total += float(loss.detach())
        loss.backward()
    params = [p for g in optimizer.param_groups for p in g["params"]]
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError("finite calibration sign/scale gradients required")
    norm = torch.nn.utils.clip_grad_norm_(params, model.config.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        module.latent_sign.clamp_(-1, 1)
        module.project_scales_()
        a8_error = sum(float(reconstruction_loss(module, x, reference, "a8")) for x in inputs)/len(inputs)
    return {"loss": total, "objective": metric, "post_update_a8_relative_sse": a8_error,
            "gradient_norm": float(norm), "supervised_tokens": 0,
            "presented_tokens": sum(len(b.input_tokens) for b in batches),
            "calibration_rows": sum(len(x) for x in inputs), "optimizer_updated": True}


def transition_optimizer(model, old_optimizer, previous):
    """Retain moments for unchanged parameters; fresh moments on replacement only."""
    import torch
    named = dict(model.named_parameters())
    remap = {id(p): n for n, p in previous.items()}
    groups, covered = [], set()
    for group in old_optimizer.param_groups:
        params = [named[remap[id(p)]] for p in group["params"]]
        groups.append({**group, "params": params})
        covered.update(map(id, params))
    added = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
    if added:
        groups.append({"params": added, "lr": 1e-5, "family": "added-capacity"})
    result = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8,
                              weight_decay=0, foreach=False)
    for parameter in model.parameters():
        if parameter in old_optimizer.state:
            result.state[parameter] = old_optimizer.state[parameter]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-binding", type=Path,
                   default=runtime.CAMPAIGN / "loop2-answer-broad01-p1/bindings.json")
    p.add_argument("--parent-checkpoint", type=Path, help="relocation only; binding SHA remains authoritative")
    p.add_argument("--parent-model", type=Path, help="current native donor relocation; binding SHA remains authoritative")
    p.add_argument("--loader", choices=["current", "historical"], default="current")
    p.add_argument("--source-checkpoint", type=Path, help="original step1000 sign-reference relocation; expected hash fixed")
    p.add_argument("--source-gguf", type=Path, help="original BF16 source relocation; expected hash fixed")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--technique", required=True,
                   choices=sorted(NATIVE_TECHNIQUES | CAPACITY_TECHNIQUES))
    p.add_argument("--projection", action="append")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--trust-floor", type=float, default=.1)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--factor-sweeps", type=int, default=2)
    p.add_argument("--max-train-seconds", type=float, default=120)
    p.add_argument("--max-updates", type=int, default=72)
    p.add_argument("--pairs", type=int, default=18)
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--export-operators", action="store_true")
    p.add_argument("--initial-only", action="store_true", help="zero training updates; record actual TRAIN anchor impact")
    p.add_argument("--reference-gguf", type=Path, help="calibration source relocation; original source SHA remains fixed")
    p.add_argument("--skip-component-cost", action="store_true", help="explicitly omit portable CUDA reference timing")
    args = p.parse_args()
    from shortlist_pilots.binary.run import validate_limits, save_checkpoint, export_operators
    validate_limits(args.max_train_seconds, args.max_updates)
    if not args.allow_cuda or not 1 <= args.pairs <= 729 or args.max_updates > 729:
        raise ValueError("explicit owner GPU admission, 1..729 pairs/updates required")
    import torch
    from shortlist_pilots.binary.adapters import SelectiveGradient, install
    from train_nine_model_qat import tensor_batch
    import w1a1_eagle.block_qat as block
    from w1a1_eagle.block_training import BlockCursor
    from eal_loss import make_eal_loss
    from joint_tools import measure_norms
    from common import guard, peaks
    from shortlist_pilots.binary.readouts import development_window, compare, preserve_readout_state
    from shortlist_pilots.binary.sign_guard import measure_sign_guard

    binding = runtime.read(args.parent_binding)
    parent = {k: dict(binding["parent"][k]) for k in ("checkpoint", "model")}
    if args.parent_checkpoint:
        parent["checkpoint"]["path"] = str(args.parent_checkpoint)
    if args.parent_model:
        parent["model"]["path"] = str(args.parent_model)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    setup = args.run_dir / "setup"
    setup.mkdir()
    if args.loader == "current":
        from shortlist_pilots.binary.parent_loader import load_full_model
        model, optimizer, baseline, metadata = load_full_model(
            setup, parent, source_checkpoint=args.source_checkpoint, source_gguf=args.source_gguf)
    else:
        if args.source_checkpoint or args.source_gguf:
            raise ValueError("source relocations require current-parent loader")
        model, optimizer, baseline, metadata = runtime.load_full_model(setup, parent)
    provider = gradient = None
    original_loss = block.block_loss
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    try:
        provider, targets = runtime.open_training_data()
        pairs = runtime.select_pairs(targets, args.pairs)
        with preserve_readout_state(model):
            anchor_batches = [tensor_batch(targets.load(i)) for i in pairs[0]]
            dev_batches, dev_provenance = development_window(targets, provider, pairs)
        parent_anchor, parent_logits = anchor_metrics(model, anchor_batches)
        parent_dev = anchor_metrics(model, dev_batches)[0]
        selected = args.projection
        if selected is None and args.technique in CAPACITY_TECHNIQUES | {"calibration-a8", "calibration-weight-mse"}:
            selected = ["blk.4.ffn_up"]
        parent_components, component_inputs = {}, {}
        if args.technique in CAPACITY_TECHNIQUES and not args.skip_component_cost:
            from shortlist_pilots.binary.calibration import capture_inputs
            with preserve_readout_state(model):
                for name in selected:
                    parent_components[name] = copy.deepcopy(model.binary_linears()[name]).eval()
                    component_inputs[name] = capture_inputs(model, anchor_batches[:1], name)[0]
        previous = dict(model.named_parameters())
        costs = install(model, args.technique, names=selected, rank=args.rank, sweeps=args.factor_sweeps)
        if costs:
            optimizer = transition_optimizer(model, optimizer, previous)
        del previous
        if args.technique in {"selective-gradient", "elastic-gradient", "trust-gradient"}:
            linears = model.binary_linears()
            if selected is not None:
                linears = {n: linears[n] for n in selected}
            mode = {"selective-gradient": "combined", "elastic-gradient": "elastic", "trust-gradient": "trust"}[args.technique]
            gradient = SelectiveGradient(linears, args.window, args.trust_floor, mode=mode)
        candidate_anchor, candidate_logits = anchor_metrics(model, anchor_batches)
        candidate_dev = anchor_metrics(model, dev_batches)[0]
        forward_equal = all(torch.equal(a, b) for a, b in zip(parent_logits, candidate_logits))
        runtime.write(args.run_dir / "initial-impact.json", {
            "parent": parent_anchor, "candidate": candidate_anchor, "exact_logits_equal": forward_equal,
            "indices": pairs[0], "development": {"provenance": dev_provenance,
                "parent": parent_dev, "candidate": candidate_dev, "comparison": compare(parent_dev, candidate_dev)},
            "scope": "fixed reused TRAIN/development actor anchors; no native quality/throughput claim"})
        if args.technique in NATIVE_TECHNIQUES and not forward_equal:
            raise ValueError("training-only arm changed zero-update forward")
        del parent_logits, candidate_logits
        calibration_metric = reference = reference_pin = None
        if args.technique in {"calibration-a8", "calibration-weight-mse"}:
            from shortlist_pilots.binary.calibration import load_reference
            if len(selected) != 1:
                raise ValueError("calibration pilot selects exactly one projection")
            reference_pin = dict(runtime.read(runtime.CAMPAIGN / "command-calibration-pilot01/joint1000.json")["model"])
            if args.reference_gguf or args.source_gguf:
                reference_pin["path"] = str(args.reference_gguf or args.source_gguf)
            module = model.binary_linears()[selected[0]]
            reference = load_reference(reference_pin["path"], reference_pin["sha256"],
                                       selected[0], module.latent_sign.shape).to(model.token_embd.device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            module.latent_sign.requires_grad_(True)
            module.scale_offset.requires_grad_(True)
            optimizer = torch.optim.AdamW([
                {"params": [module.latent_sign], "lr": 3e-5, "family": "sign"},
                {"params": [module.scale_offset], "lr": 1e-5, "family": "scale"}],
                betas=(.9, .999), eps=1e-8, weight_decay=0, foreach=False)
            calibration_metric = "a8" if args.technique == "calibration-a8" else "weight-mse"
        block.block_loss = make_eal_loss(block.block_supported_mask)
        norm_reference = {n: t.to(model.token_embd.device) for n, t in metadata["initial_norms"].items()}
        changed_representation = selected if args.technique in {"double-binary", "ternary-a8"} else ()
        intervention = {"technique": args.technique, "projections": selected, "window": args.window,
                        "trust_floor": args.trust_floor, "rank": args.rank,
                        "factor_sweeps": args.factor_sweeps, "costs": costs,
                        "loss": ("relative projection SSE: " + calibration_metric if calibration_metric
                                 else "unchanged parent EAL48: .1*(.1*depthCE5+1-sum(prefixProb5)/5)"),
                        "optimizer": ("fresh matched projection sign/scale Adam; all other parameters frozen"
                                      if calibration_metric else "inherited parent48 Adam; capacity replaces only changed parameter moments"),
                        "reference": reference_pin,
                        "original_sign_guard": {"fraction_limit": .02,
                            "guarded_projection_count": len(model.binary_linears())-len(changed_representation),
                            "changed_projection_guard_not_applicable": list(changed_representation)},
                        "native_codec": not bool(costs)}
        source = {"parent": parent, "parent_binding": runtime.pin(args.parent_binding),
                  "data_inputs": targets.plan_pin, "order": targets.order_pin,
                  "synthetic": False, "pairs": pairs,
                  "implementation": {str(f.relative_to(ROOT)): runtime.sha(f)
                                     for f in Path(__file__).parent.glob("*.py")},
                  "runtime": runtime.pin(Path(runtime.__file__)), "loader": args.loader,
                  "development": dev_provenance,
                  "component_cost": ("explicitly_skipped" if args.skip_component_cost else
                                     "actual_shape_cuda_reference" if costs else "native_arm_evaluated_separately")}
        runtime.write(args.run_dir / "start.json", {"source": source, "intervention": intervention,
                      "limits": {"seconds": args.max_train_seconds, "updates": args.max_updates},
                      "original_guards": {"sign_fraction": .02, "norm_relative_l2": .1, "norm_absolute": .5},
                      "development_status": "repeated TRAIN486 anchors; no fresh holdout",
                      "native_fidelity": "historical ordinary activation/reduction failures retained"})
        cursor = BlockCursor(data_cursor=targets.cursor(0))
        status, maximum_update = "COMPLETE", 0.
        model.train()
        while not args.initial_only and cursor.step < args.max_updates:
            if stop[0] or (args.run_dir / "STOP").exists():
                status = "STOPPED"
                break
            if cursor.elapsed_seconds >= args.max_train_seconds or (
                    maximum_update and args.max_train_seconds-cursor.elapsed_seconds < 2*maximum_update):
                status = "TIME_CAP_GUARD"
                break
            guard()
            ids = pairs[cursor.step % len(pairs)]
            batches = [tensor_batch(targets.load(i)) for i in ids]
            torch.cuda.synchronize()
            clock = time.monotonic()
            if calibration_metric:
                metrics = calibration_update(model, optimizer, batches, selected[0], reference, calibration_metric)
            else:
                metrics, _ = block.block_train_batch(model, optimizer, batches, return_outputs=False)
            norms = measure_norms(model, norm_reference)
            displacement = measure_sign_guard(baseline, model, excluded=changed_representation)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-clock
            maximum_update = max(maximum_update, elapsed)
            step = cursor.step+1
            cursor = BlockCursor(step=step, block_index=2*step,
                                 supervised_tokens=cursor.supervised_tokens+metrics["supervised_tokens"],
                                 presented_tokens=cursor.presented_tokens+metrics["presented_tokens"],
                                 elapsed_seconds=cursor.elapsed_seconds+elapsed,
                                 data_cursor=targets.cursor(step))
            row = {"cursor": asdict(cursor), "indices": ids, "metrics": metrics, "norms": norms,
                   "sign_displacement": displacement, "update_train_seconds": elapsed,
                   "gradient_intervention": {} if gradient is None else gradient.stats}
            with (args.run_dir / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            if ((displacement["fraction"] is not None and displacement["fraction"] > .02) or norms["max_relative_l2"] > .1
                    or norms["max_absolute_change"] > .5):
                status = "GUARD_STOP"
                break
            del batches
        # Make paid training state durable before optional readouts/timing.
        receipt = (None if args.initial_only else save_checkpoint(
            model, optimizer, cursor, source, intervention, args.run_dir / "checkpoint.pt"))
        if args.technique in NATIVE_TECHNIQUES and not args.initial_only:
            from shortlist_pilots.binary.parent_loader import native_export
            runtime.write(args.run_dir / "export.json", native_export(model, parent, args.run_dir / "model.gguf"))
        if args.export_operators:
            export_operators(model, args.run_dir / "operators.npz")
        post_anchor = anchor_metrics(model, anchor_batches)[0]
        post_dev = anchor_metrics(model, dev_batches)[0]
        runtime.write(args.run_dir / "fixed-readouts.json", {
            "schema": "shortlist_binary_fixed_readouts_v1", "training_indices": pairs[0],
            "development": dev_provenance, "trained_updates": cursor.step,
            "train_anchor": {"parent": parent_anchor, "initial": candidate_anchor, "after": post_anchor,
                "training_impact": compare(candidate_anchor, post_anchor), "total_impact": compare(parent_anchor, post_anchor)},
            "development_anchor": {"parent": parent_dev, "initial": candidate_dev, "after": post_dev,
                "training_impact": compare(candidate_dev, post_dev), "total_impact": compare(parent_dev, post_dev)},
            "RNG_restored_each_readout": True, "optimizer_and_cursor_updated_by_readouts": False,
            "scope": "fixed captured-prefix actor CE/depth, reused development; no native acceptance claim"})
        if parent_components:
            from shortlist_pilots.binary.component_cost import benchmark_pair
            with preserve_readout_state(model):
                component_reports = {name: benchmark_pair(parent_module, model.binary_linears()[name], component_inputs[name])
                                     for name, parent_module in parent_components.items()}
            runtime.write(args.run_dir / "component-cost.json", {
                "operators": component_reports,
                "input_source": "exact pre-training parent, first fixed TRAIN anchor; seven actual operator rows",
                "RNG_restored": True, "included_in_TRAIN_timer": False,
                "native_or_full_request_speed_claim": False})
        del parent_components, component_inputs, anchor_batches, dev_batches
        runtime.write(args.run_dir / "finish.json", {"status": status, "cursor": asdict(cursor),
                      "checkpoint": receipt, "resources": peaks,
                      "native_codec": not bool(costs),
                      "native_acceptance_and_request_throughput": "not measured by trainer",
                      "train_timer": "synchronized updates plus original norms/sign diagnostics",
                      "timer_cap": "840s maximum admission; indivisible update, 2x observed maximum guard"})
    finally:
        block.block_loss = original_loss
        if gradient is not None:
            gradient.close()
        if provider is not None:
            provider.close()
        gc.collect()


if __name__ == "__main__":
    main()

"""Finite independent objective branch; exact existing48/Adam/RNG restore/export.

Run under the root operator's sole-GPU supervisor. This does not launch remote
work, servers or successors. All raw state stays in the specified output folder.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

from runtime import (CAMPAIGN, configure_imports, load_full_model, native_export,
                     open_training_data, pin, read, select_pairs, sha, validate_spec, write)
from losses import dpal_loss, vat_loss, edr_td_loss, soft_round_values


def train(args):
    spec = read(args.spec)
    validate_spec(spec)
    if not args.allow_cuda:
        raise ValueError("sole operator must explicitly provide --allow-cuda")
    configure_imports()
    import torch
    from common import guard
    from joint_tools import joint_parameters, binary_buffers, measure_norms
    from checkpointing import BoundaryStop, publish_full_state
    from restore import equal
    from train_nine_model_qat import tensor_batch
    from w1a1_eagle.block_training import _block_rng_state, _restore_block_rng
    import hidden
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    output_floor = (2 if str(args.output.resolve()).startswith("/mnt/") else 8)*1024**3
    if (shutil.disk_usage(args.output).free < output_floor + 4_600_000_000
            or shutil.disk_usage(CAMPAIGN).free < 8*1024**3):
        raise ValueError("lane live floor plus projected fullstate/GGUF required")
    setup = args.output/"setup"
    setup.mkdir()
    provider = None
    old_loss, old_error = hidden.block_loss, hidden.hidden_error_sum
    elapsed, local = 0., 0
    pilot = spec["pilot"]
    try:
        provider, targets = open_training_data()
        pairs = select_pairs(targets, spec["pairs"])
        indices = sorted({i for pair in pairs for i in pair})
        model, optimizer, baseline, metadata = load_full_model(
            setup, spec["parent"], source_checkpoint=(spec.get("source_checkpoint") or {}).get("path"))
        params = joint_parameters(model)
        norm_reference = {n:v.to(model.token_embd.device) for n,v in metadata["initial_norms"].items()}
        head = head_optimizer = None
        if pilot == "vat":
            # Separate training-only scope; existing model and native GGUF stay
            # exact48. Nonzero head weights give a real hidden gradient at step1.
            with torch.random.fork_rng(devices=[model.token_embd.device.index]):
                torch.manual_seed(8101)
                sample = tensor_batch(targets.load(indices[0]))
                rng = _block_rng_state(model.token_embd.device)
                with torch.no_grad():
                    h = model(sample).layer_states[-1]
                _restore_block_rng(rng, model.token_embd.device)
                head = torch.nn.Linear(h.shape[-1], 1, device=h.device, dtype=torch.float32)
                del h, sample
            head_optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0, foreach=False)
        teacher_cache = {}
        if spec.get("teacher_manifest"):
            ref = spec["teacher_manifest"]
            assert sha(ref["path"]) == ref["sha256"]
            teacher = read(ref["path"])
            assert teacher["inputs"] == targets.plan_pin and teacher["order"] == targets.order_pin
            assert teacher["target_sha256"] == provider.dataset.manifest["producer"]["target_sha256"]
            assert teacher["target_precision"] == teacher["target_kv_type"] == "F16"
            assert teacher["pairs"] == pairs and teacher["producer_closed"]
            for entry in teacher["rows"]:
                assert sha(entry["logits"]["path"]) == entry["logits"]["sha256"]
                assert entry["labels"] == targets.cases[entry["index"]]["labels"][:5]
                import numpy as np
                logits = np.memmap(entry["logits"]["path"], dtype="<f4", mode="r", shape=tuple(entry["shape"]))
                assert logits.shape == (5, model.config.vocab_size) and logits.argmax(-1).tolist() == entry["labels"]
                teacher_cache[entry["index"]] = torch.from_numpy(logits.copy())
                del logits
            assert set(indices) == set(teacher_cache)
        if pilot == "dpal" and not teacher_cache:
            raise ValueError("D-PAL requires authenticated full-vocabulary F16 target probabilities")
        future = {}
        if pilot == "edr":
            replay_ref = spec["edr_native_replay"]
            assert replay_ref and sha(replay_ref["path"]) == replay_ref["sha256"]
            replay = read(replay_ref["path"])
            assert replay["native_counters_checked"] and replay["target_reference_checked"]
            from absolute_views import absolute_block
            rng = _block_rng_state(model.token_embd.device)
            cache_started = time.monotonic()
            cache_rows = []
            with torch.no_grad():
                for index in indices:
                    guard()
                    case = targets.cases[index]
                    cid, anchor = case["chain_id"], case["anchor"]
                    tokens = provider.dataset.receipts[cid]["tokens"]
                    horizon = min(spec["edr_horizon"], len(tokens)-7-anchor, 2042-anchor)
                    assert horizon >= 1
                    acceptance = []
                    for offset in range(horizon):
                        guard()
                        if time.monotonic()-cache_started >= spec.get("edr_cache_max_seconds",120):
                            raise RuntimeError("finite EDR cache boundary; no optimizer update occurred")
                        batch = tensor_batch(absolute_block(provider, cid, anchor+offset))
                        output = model(batch)
                        a = output.logits[:5].float().log_softmax(-1).gather(-1, batch.labels[:5].to(model.token_embd.device)[:,None]).squeeze(-1).exp()
                        acceptance.append(a)
                        del batch, output
                    values = soft_round_values(torch.stack(acceptance)).cpu()
                    future[index] = [float(values[min(horizon,k)]) for k in range(1,7)]
                    cache_rows.append({"index":index,"anchor":anchor,"horizon":horizon,"future":future[index]})
            _restore_block_rng(rng, model.token_embd.device)
            write(args.output/"edr-cross-round-cache.json", {"parent":spec["parent"],"rows":cache_rows,
                  "model_state_updated":False,"rng_restored":True,"finite_horizon":True,
                  "cache_seconds":time.monotonic()-cache_started,
                  "native_arithmetic_exact":False,"stochastic_EDR_theorem_claimed":False})
        details, current = [], {}
        from eal_loss import make_eal_loss
        control_loss = make_eal_loss(hidden.block_supported_mask)
        def objective(output, batch, config, *, teacher_callback=None):
            if teacher_callback is not None or config.conditioning != "captured_prefix":
                raise ValueError("fixed authentic captured-prefix recipe required")
            if not bool(hidden.block_supported_mask(batch, config).all()):
                raise ValueError("complete seven-row authentic block required")
            index = current[id(batch)]
            if pilot == "control":
                return control_loss(output, batch, config)
            if pilot == "dpal":
                loss, record = dpal_loss(output.logits, batch.labels, teacher_cache[index], floor=spec["dpal_floor"])
            elif pilot == "vat":
                loss, record = vat_loss(output.logits,batch.labels,output.layer_states[-1],head,
                                        decay=config.depth_decay,beta=spec["vat_beta"],teacher_logits=teacher_cache.get(index))
            else:
                loss, record = edr_td_loss(output.logits,batch.labels,future[index])
            details.append(record)
            weights = config.depth_decay**torch.arange(7, device=output.logits.device).float()
            return loss, {"weighted_denominator":float(weights.sum()),"ce":record["ce"],"probability_l1":0.,
                          "supervised_tokens":5,"presented_tokens":int(batch.loss_mask.sum()),
                          "prefix_mismatch_tokens":0,"conditioning":output.conditioning}
        hidden.block_loss = objective
        def no_hidden(student, teacher, supported):
            assert teacher is None and bool(supported.all())
            return student.new_zeros(()), {"supported_rows":7,"normalized_error_sum":0.}
        hidden.hidden_error_sum = no_hidden
        def readout():
            """Fixed TRAIN readout; preserves the inherited RNG at update boundary."""
            rng = _block_rng_state(model.token_embd.device)
            rows = []
            with torch.no_grad():
                for index in indices:
                    guard()
                    batch = tensor_batch(targets.load(index))
                    current.clear();current[id(batch)] = index
                    output = model(batch)
                    loss, _ = objective(output,batch,model.config)
                    y = batch.labels[:5].to(output.logits.device)
                    predictions = output.logits[:5].argmax(-1)
                    hits = predictions.eq(y)
                    probabilities = output.logits[:5].log_softmax(-1).gather(-1,y[:,None]).squeeze(-1).exp()
                    row = {"index":index,"group_id":targets.cases[index]["group_id"],
                           "loss":float(loss),"predictions":predictions.cpu().tolist(),"labels":y.cpu().tolist(),
                           "prefix_length":int(hits.long().cumprod(0).sum()),
                           "position_hits":hits.cpu().tolist(),"target_probabilities":probabilities.cpu().tolist()}
                    if pilot != "control":row["objective_details"] = dict(details[-1])
                    rows.append(row)
                    del output,batch,loss
            _restore_block_rng(rng,model.token_embd.device)
            details.clear();current.clear()
            return {"scope":"same fixed TRAIN windows; approximate trainer arithmetic; not native held-out impact",
                    "rows":rows,"mean_prefix_length":sum(r["prefix_length"] for r in rows)/len(rows),
                    "mean_loss":sum(r["loss"] for r in rows)/len(rows),"rng_restored":True}
        write(args.output/"TRAIN-before.json",readout())
        write(args.output/"start.json", {"spec":spec,"inputs":targets.plan_pin,"order":targets.order_pin,
              "pairs":pairs,"exact_parent_restore":True,"parent_step":3504,"original_guards_retained":True,
              "historical_fidelity":"FAILED_RETAINED","ordinary_arithmetic":"approximate native quantized activation/reduction",
              "teacher":spec.get("teacher_manifest"),"training_only_head":pilot=="vat"})
        status = "UPDATES_COMPLETE"
        last_duration = 0.
        model.train()
        with BoundaryStop([args.output/"STOP", CAMPAIGN/"STOP"], time.time()+1800) as stop:
            while local < spec["max_updates"]:
                if stop.requested():
                    status = "STOPPED";break
                # Leave multiple prior-update durations before the hard ceiling.
                if elapsed + max(5., 3*last_duration) >= spec["max_train_seconds"]:
                    status = "FINITE_TRAIN_BOUNDARY";break
                guard()
                ids = pairs[local%len(pairs)]
                batches = [tensor_batch(targets.load(i)) for i in ids]
                current = {id(batch):i for batch,i in zip(batches,ids)}
                details.clear()
                if head_optimizer:
                    head_optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize();before=time.monotonic()
                metrics,_ = hidden.hidden_train_batch(model,optimizer,batches,hidden_targets=[None,None],hidden_weight=0.,return_outputs=False)
                if head_optimizer:
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
                    head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                    head_optimizer.step()
                    metrics["head_gradient_norm"] = float(head_norm)
                norms = measure_norms(model,norm_reference)
                sign = baseline.measure(model)
                torch.cuda.synchronize();last_duration=time.monotonic()-before
                elapsed += last_duration;local += 1
                with (args.output/"updates.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step":3504+local,"local_updates":local,"indices":ids,
                        "TRAIN_seconds":elapsed,"update_TRAIN_seconds":last_duration,"metrics":metrics,
                        "objective_details":details,"norms":norms,"sign_displacement":sign},allow_nan=False)+"\n")
                (args.output/"progress.json").write_text(json.dumps({"local_updates":local,"TRAIN_seconds":elapsed}))
                del batches
                if sign["fraction"]>.02 or norms["max_relative_l2"]>.1 or norms["max_absolute_change"]>.5:
                    status="ORIGINAL_GUARD_STOP";break
                if local%100==0:gc.collect()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        write(args.output/"TRAIN-after.json",readout())
        state = {"schema":"shortlist_objective_full48_v1","pilot":pilot,"spec":spec,"step":3504+local,
                 "parameters":{n:p.detach() for n,p in params.items()},"binary_buffers":binary_buffers(model),
                 "optimizer":optimizer.state_dict(),"rng":_block_rng_state(model.token_embd.device),
                 "config":asdict(model.config),"initial_norms":metadata["initial_norms"],
                 "initial_sign_hashes":metadata["initial_sign_hashes"],"TRAIN_seconds":metadata["TRAIN_seconds"]+elapsed,
                 "additional_TRAIN_seconds":elapsed,"local_updates":local,"pairs":pairs,
                 "data_cursor":{"local_pair_index":local%len(pairs),"blocks_consumed":2*local,"inputs":targets.plan_pin,"order":targets.order_pin},
                 "head":head.state_dict() if head else None,"head_optimizer":head_optimizer.state_dict() if head_optimizer else None,
                 "final_sign_hashes":{n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in params.items() if n.endswith(".latent_sign")},
                 "sign_displacement":baseline.measure(model),"stage_status":status}
        def verify(saved):
            assert equal(state,saved) and len(saved["parameters"])==48
            assert all(int(v["step"])==3504+local for v in saved["optimizer"]["state"].values())
        checkpoint=publish_full_state(args.output/"checkpoint.pt",state,verify)
        del state;gc.collect()
        exported=native_export(model,args.output/"model.gguf",parent=spec["parent"])
        write(args.output/"export.json",exported)
        write(args.output/"finish.json",{"status":status,"pilot":pilot,"step":3504+local,
              "additional_updates":local,"TRAIN_seconds":elapsed,"checkpoint":checkpoint,
              "model":exported["output"],"export":pin(args.output/"export.json"),
              "parent_intact":sha(spec["parent"]["checkpoint"]["path"])==spec["parent"]["checkpoint"]["sha256"],
              "native_acceptance_throughput":"PENDING_ROOT_NATIVE_EVALUATION","bounded_pilot":True})
    finally:
        hidden.block_loss,hidden.hidden_error_sum=old_loss,old_error
        if provider is not None:provider.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-cuda",action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":main()

"""Capture genuine F16 target distributions at the exact admitted TRAIN prefixes.

Separate serial GPU phase; the NativeTeacher context reaps its sole child before
training can start. Requires original decode partitions and checks every argmax
against original greedy labels. No parent-draft probabilities stand in for p.
"""
import argparse
import importlib.util
from pathlib import Path
import signal
import sys
import time

from runtime import CAMPAIGN, configure_imports, open_training_data, pin, read, select_pairs, sha, validate_spec, write


def trim_decode_history(history, length):
    """Keep original target-only decode boundaries, shortening only the tail."""
    result = []
    consumed = 0
    for chunk in history:
        if consumed >= length:
            break
        if chunk["offset"] != consumed:
            raise ValueError("original decode history has an offset gap")
        count = min(chunk["count"], length-consumed)
        result.append({**chunk,"count":count})
        consumed += count
    if consumed != length:
        raise ValueError("original decode history does not cover teacher prefix")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if not args.allow_cuda or sha(args.binary) != args.binary_sha256:
        raise ValueError("root sole-operator CUDA authorization and exact native binary required")
    configure_imports()
    # The authentic indexed client resolves src relative to its own source root.
    # Keep that complete frozen root, rather than the partial operator checkout.
    module_path = CAMPAIGN/"training-source06/source/scripts/capture_block_qat_teacher.py"
    if sha(module_path) != "9496feeb1f38ce83939fc377e55843314073c2237b79cb208c52dd90db20f4be":
        raise ValueError("original indexed teacher client changed")
    module_spec = importlib.util.spec_from_file_location("shortlist_native_teacher", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    NativeTeacher = module.NativeTeacher
    import numpy as np
    spec = read(args.spec)
    validate_spec(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    provider = None
    started = time.monotonic()
    try:
        provider, targets = open_training_data()
        expected_target = provider.dataset.manifest["producer"]["target_sha256"]
        if args.target_sha256 != expected_target:
            raise ValueError("teacher target differs from admitted native F16 TRAIN source")
        pairs = select_pairs(targets,spec["pairs"])
        indices = sorted({i for pair in pairs for i in pair})
        rows = []
        # Signals unwind the context so its exact child is stopped and reaped.
        def stop(signum, frame):
            raise SystemExit(128+signum)
        prior = {sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            with NativeTeacher(args.binary,args.target,args.output/"native",
                 target_sha256=expected_target,max_tokens=2048,gpu_layers=999,
                 producer_source_revision=args.source_revision) as teacher:
                if teacher.ancestry["target_precision"] != "F16":
                    raise ValueError("full F16 target matrices required")
                for index in indices:
                    if (args.output/"STOP").exists() or time.monotonic()-started > 600:
                        raise RuntimeError("finite teacher-capture boundary reached")
                    case = targets.cases[index]
                    block = targets.load(index)
                    source = provider.dataset.receipts[case["chain_id"]]
                    prefix = list(block.prefix_tokens)+block.labels[:4].tolist()
                    first = len(block.prefix_tokens)-1
                    # Original generation history preserves native prefill and
                    # token-wise decode arithmetic at every selected row.
                    history = trim_decode_history(source["decode_history"],len(prefix))
                    receipt = teacher.capture_prefix(prefix,provider.dataset.manifest["taps"],
                        logits_mode="indexed",logits_indices=list(range(first,first+5)),decode_history=history,
                        chain_ancestry={"source_split":"TRAIN","group_id":case["group_id"],
                                       "original_case_index":index,"original_chain":case["chain_id"]})
                    if not receipt["target_precision"] == receipt["target_kv_type"] == "F16":
                        raise ValueError("F16 target/KV contract required")
                    logits = NativeTeacher.array(receipt,"logits")
                    if (logits.shape != (5,provider.dataset.manifest["vocab_size"])
                            or not np.isfinite(logits).all()
                            or logits.argmax(-1).tolist() != block.labels[:5].tolist()):
                        raise ValueError("exact original-prefix full-logit greedy-label agreement failed")
                    logits_ref = receipt["files"]["logits"]
                    rows.append({"index":index,"labels":block.labels[:5].tolist(),
                                 "logits":{k:logits_ref[k] for k in ("path","sha256")},"shape":list(logits.shape),
                                 "receipt":pin(args.output/"native"/receipt["id"]/"receipt.json"),
                                 "original_source_receipt":provider.chains[case["chain_id"]]["native_receipt"],
                                 "decode_history_exact_original_prefix":True})
                    del logits
                    # Features duplicate immutable admitted TRAIN features and
                    # are not consumed by this pilot. Remove only this exact
                    # trial-owned file after validating/hash-recording it.
                    feature = Path(receipt["files"]["features"]["path"])
                    if not feature.resolve().is_relative_to(args.output.resolve()):
                        raise ValueError("refuse cleanup outside owned capture output")
                    feature.unlink()
                    write(args.output/"native"/receipt["id"]/"feature-disposal.json",
                          {"removed":receipt["files"]["features"],"reason":"unused duplicate training features; native receipt and full logits retained"})
        finally:
            for sig, handler in prior.items():signal.signal(sig,handler)
        write(args.output/"manifest.json", {"schema":"shortlist_full_f16_teacher_v1",
              "inputs":targets.plan_pin,"order":targets.order_pin,"pairs":pairs,"rows":rows,
              "target_sha256":expected_target,"target_precision":"F16","target_kv_type":"F16",
              "native_binary":pin(args.binary),"source_revision":args.source_revision,
              "client_source":pin(module_path),"producer_closed":True,"seconds":time.monotonic()-started,
              "heldout_used":False,"synthetic":False})
    finally:
        if provider is not None:provider.close()


if __name__ == "__main__":main()

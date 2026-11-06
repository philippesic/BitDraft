#!/usr/bin/env python3
"""Prepare identical-input historical replay from completed capture attempt 04.

Only measured development traces are used. This checks complete proposal and
verifier identity before copying parent feature prefixes. It does not operate a
GPU, inspect future replay results, read full-vocabulary logits, or train.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("parent3504", "q4_k_m")
FEATURE_DIM = 12800
ROUND_FIELDS = ("schema", "speculative_types", "slot_id", "round_index", "status", "replay",
                "n_draft_max", "n_draft_configured", "draft_p_min", "stopped_low_confidence",
                "stop_probability", "discarded_below_min", "n_proposed", "n_accepted",
                "n_accepted_usable_prefix", "n_accepted_verified", "verified_token_ids",
                "n_emitted", "proposed_token_ids", "emitted_token_ids")


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def read(path):
    return json.loads(Path(path).read_text())


def grouped(path):
    groups = {}
    previous = None
    for number, line in enumerate(Path(path).open(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task = row["task_id"]
        need(task == previous or task not in groups, "interleaved task groups")
        groups.setdefault(task, []).append((number, row))
        previous = task
    return list(groups.items())


def measured_groups(path, requests):
    groups = grouped(path)
    need(len(groups) == len(requests) + 2, "two warmups plus measured task groups required")
    for index, (_task, rows) in enumerate(groups[:2]):
        emitted = [token for _, row in rows for token in row["emitted_token_ids"]]
        expected = requests[index]["result"]["generated_token_ids"]
        need(emitted in (expected, expected[1:]), "warmup output does not match its measured prompt")
    return {request["prompt_id"]: group for request, group in zip(requests, groups[2:])}, [g[0] for g in groups[:2]]


def preserve_decisions(capture, original, analysis):
    original_config = read(original / "config.json")
    captured_config = read(capture / "configuration.json")
    need(set(captured_config["models"]) == set(ARMS), "capture arm contract changed")
    need(captured_config["target"] == original_config["target"], "target contract changed")
    need(captured_config["source_files"] == original_config["source_files"], "numerical helper sources changed")
    need(captured_config["panels"] == original_config["panels"], "prompt/reference contracts changed")
    for arm in ARMS:
        need(captured_config["models"][arm] == original_config["models"][arm], "draft model changed")
    original_libraries = [r for r in original_config["runtime_libraries"]
                          if Path(r["path"]).name != "libllama-server-impl.so"]
    captured_libraries = [r for r in captured_config["runtime_libraries"]
                          if Path(r["path"]).name != "libllama-server-impl.so"]
    need(captured_libraries == original_libraries, "core numerical library declarations changed")
    original_request_path = original / "requests.jsonl"
    need(sha(original_request_path) == analysis["source"]["requests"]["sha256"], "selection request source changed")
    original_requests = [json.loads(line) for line in original_request_path.open()]
    captured_request_path = capture / "requests" / "requests.jsonl"
    captured_requests = [json.loads(line) for line in captured_request_path.open()]
    need(len(original_requests) == 72 and len(captured_requests) == 48, "72 original and 48 captured requests required")
    all_meta = {}
    forced_by_prompt = {}
    checks = Counter()
    arm_reports = {}
    for arm in ARMS:
        old = [q for q in original_requests if q["arm"] == arm]
        new = [q for q in captured_requests if q["arm"] == arm]
        need(len(old) == len(new) == 24, "24 measured requests per arm required")
        need([q["prompt_id"] for q in old] == [q["prompt_id"] for q in new], "request order/IDs differ")
        old_trace = original / "development" / "rep-00" / arm / "rounds.jsonl"
        need(sha(old_trace) == analysis["source"]["traces"][arm]["sha256"], "original round source changed")
        new_trace = capture / "requests" / "development" / "rep-00" / arm / "rounds.jsonl"
        old_groups, old_warmups = measured_groups(old_trace, old)
        new_groups, new_warmups = measured_groups(new_trace, new)
        forced_path = capture / "capture" / arm / "rounds.jsonl"
        forced_groups = dict(grouped(forced_path))
        need(len(forced_groups) == 26, "forced round groups must include two warmups")
        joins = []
        for old_request, new_request in zip(old, new):
            prompt = old_request["prompt_id"]
            need(old_request["panel"] == new_request["panel"] == "development", "non-development request")
            need(old_request["repetition"] == new_request["repetition"] == 0, "unexpected repetition")
            need(old_request["domain"] == new_request["domain"], "request domain changed")
            for field in ("generated_token_ids", "completion_tokens", "prompt_tokens", "completion_sha256",
                          "finish_reason", "speculative"):
                need(new_request["result"][field] == old_request["result"][field],
                     f"captured request {field} differs: {arm}/{prompt}")
            old_task, old_rows = old_groups[prompt]
            new_task, new_rows = new_groups[prompt]
            need(len(old_rows) == len(new_rows), f"round count differs: {arm}/{prompt}")
            source_forced = forced_groups[new_task]
            forced_rounds = {row["round_index"]: (number, row) for number, row in source_forced}
            need(len(forced_rounds) == len(source_forced), "duplicate forced round index")
            need(len(forced_rounds) == sum(r["status"] == "complete" for _, r in new_rows),
                 "forced ledger does not cover all complete rounds")
            ids = old_request["result"]["generated_token_ids"]
            emitted = [token for _, r in new_rows for token in r["emitted_token_ids"]]
            need(emitted == ids[1:], "expected omitted initial seed does not join")
            pos = 1
            forced_at = {}
            for (old_line, before), (new_line, after) in zip(old_rows, new_rows):
                for field in ROUND_FIELDS:
                    need(before[field] == after[field], f"round {field} differs: {arm}/{prompt}/{before['round_index']}")
                need(after["emitted_token_ids"] == ids[pos:pos + after["n_emitted"]], "captured emission anchor mismatch")
                if after["status"] == "complete":
                    forced_line, forced = forced_rounds[after["round_index"]]
                    need(forced["schema"] == "eagle_forced_round_v1", "forced schema changed")
                    need(forced["draft_token_ids"] == after["proposed_token_ids"], "forced proposals differ")
                    need(forced["accepted_drafts"] == after["n_accepted"], "forced acceptance differs")
                    need(forced["verifier_token_ids"] == after["verified_token_ids"], "forced verifier IDs differ")
                    forced_at[pos] = {"row": forced, "line": forced_line, "trace_line": new_line}
                    checks["complete_forced_round_joins"] += 1
                checks["all_measured_round_decisions_identical"] += 1
                pos += after["n_emitted"]
            need(pos == len(ids), "captured trace fails full output coverage")
            checks["measured_requests_identical"] += 1
            joins.append({"prompt_id": prompt, "original_task_id": old_task,
                          "captured_task_id": new_task, "rounds": len(new_rows)})
            all_meta[arm, prompt] = {"original": old_request, "capture": new_request,
                                     "task_id": new_task, "original_task_id": old_task}
            forced_by_prompt[arm, prompt] = forced_at
        release = read(capture / "requests" / "development" / "rep-00" / arm / "release.json")
        need(release.get("owned_server_reaped") is True, "capture server release receipt missing")
        arm_reports[arm] = {"original_warmup_tasks_skipped": old_warmups,
                           "capture_warmup_tasks_skipped": new_warmups, "joins": joins,
                           "original_trace": pin(old_trace), "captured_trace": pin(new_trace),
                           "forced_rounds": pin(forced_path)}
    return all_meta, forced_by_prompt, {"capture_preserved_original_decisions": True,
            "checks": dict(checks), "arms": arm_reports,
            "original_config": pin(original / "config.json"), "captured_config": pin(capture / "configuration.json"),
            "original_requests": pin(original_request_path), "captured_requests": pin(captured_request_path)}


def prepare(capture, original, analysis_path, output, reserve_bytes):
    receipt_path = capture / "receipt.json"
    if not receipt_path.exists() or read(receipt_path).get("completed") is not True:
        return {"status": "CAPTURE_NOT_COMPLETE", "writes": 0}
    need(read(capture / "requests" / "summary.json").get("status") == "COMPLETE", "capture evaluator incomplete")
    need(capture.name == "historical-frontier-capture-04", "only completed attempt 04 is authorized")
    need(not output.exists(), "output exists; preserve it and choose a new directory")
    analysis = read(analysis_path)
    need(analysis["schema"] == "historical_observable_frontier_analysis_v1", "frontier selection schema changed")
    metadata, forced_by_prompt, decision_proof = preserve_decisions(capture, original, analysis)
    helper_path = ROOT / "runs/5080-broader-training-20261031/serving-data.py"
    helper_spec = importlib.util.spec_from_file_location("retained_serving_ledger", helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    ledger_path = capture / "capture" / "parent3504" / "capture.target_features.jsonl"
    decoded, dispositions, _quality = helper.ledger([json.loads(line) for line in ledger_path.open()])
    feature_path = capture / "capture" / "parent3504" / "capture.target_features.f32"
    need(sys.byteorder == "little", "historical feature contract requires little-endian host")
    need(feature_path.stat().st_size == len(decoded) * FEATURE_DIM * 4, "feature row dimensions differ")
    need(set(decoded) == set(range(len(decoded))), "feature-row IDs are not dense")
    committed = {}
    for (arm, prompt), meta in metadata.items():
        if arm != "parent3504":
            continue
        rows = helper.committed_rows(decoded, dispositions, meta["task_id"])
        need(rows, "empty committed parent history")
        for pos, row in rows.items():
            need(row["feature_dim"] == FEATURE_DIM and row["feature_dtype"] == "float32_native_endian",
                 "target feature representation differs")
            need(row["boundary"] == "raw_target_layer_input_before_eagle_encoder" and
                 row["source"] == "target_verifier", "target feature boundary differs")
        committed[prompt] = rows
    table = {row["frontier_id"]: row for row in analysis["frontiers"]}
    selection = analysis["counterfactual_selection"]
    chosen = []
    for index, matched in enumerate(selection["matched_controls"]):
        chosen.append((matched["damaged_frontier"], "damaged", index))
        for key, category in (("unchanged_full_five", "unchanged"), ("improved", "improved")):
            chosen.append((matched["controls"][key]["frontier_id"], category, index))
    need(len(chosen) == 96 and len({x[0] for x in chosen}) == 96, "expected original 96 unique selections")
    need({x[0] for x in chosen} == set(selection["selected_frontier_ids"]), "selected frontier identities changed")
    included = []
    excluded = []
    for frontier_id, category, matched_set in chosen:
        frontier = table[frontier_id]
        prompt = frontier["prompt_id"]
        out_anchor = frontier["output_relative_anchor"]
        meta = metadata["parent3504", prompt]
        original_request = meta["original"]
        forced_ref = forced_by_prompt["parent3504", prompt].get(out_anchor)
        need(forced_ref is not None, "selected frontier absent from actual parent forced ledger")
        forced = forced_ref["row"]
        anchor = original_request["result"]["prompt_tokens"] + out_anchor - 1
        prefix = forced["prefix_token_ids"] + [forced["seed_token_id"]]
        need(forced["pos0"] == anchor and len(prefix) == anchor + 1, "seed position/native pos0 mismatch")
        history = committed[prompt]
        need(anchor in history, "seed has no retained committed feature row")
        need(prefix == [history[p]["token_id"] for p in range(anchor + 1)], "full committed prefix/seed mismatch")
        ids = original_request["result"]["generated_token_ids"]
        need(forced["seed_token_id"] == ids[out_anchor - 1], "output-relative seed differs")
        expectations = frontier["arms"]
        for arm in ARMS:
            if arm not in expectations:
                continue
            expectation = expectations[arm]
            captured_forced = forced_by_prompt[arm, prompt].get(out_anchor)
            need(captured_forced is not None, "available arm expectation has no actual forced round")
            need(captured_forced["row"]["accepted_drafts"] == expectation["n_accepted"] and
                 len(captured_forced["row"]["draft_token_ids"]) == expectation["n_proposed"],
                 "selected expectation differs from actual capture")
        if out_anchor + 7 > len(ids) or anchor + 7 not in history:
            excluded.append({"frontier_id": frontier_id, "category": category, "matched_set": matched_set,
                "reason": "fewer than seven true committed future output labels",
                "available_output_labels": max(0, len(ids) - out_anchor),
                "committed_max_position": max(history), "required_last_position": anchor + 7})
            continue
        labels = [history[p]["token_id"] for p in range(anchor + 1, anchor + 8)]
        need(labels == ids[out_anchor:out_anchor + 7], "committed future labels differ from actual output")
        accepted = forced["accepted_drafts"]
        need(forced["verifier_token_ids"] == labels[:accepted + 1], "actual verifier labels differ")
        need(forced["draft_token_ids"][:accepted] == labels[:accepted], "actual accepted drafts differ")
        if accepted < len(forced["draft_token_ids"]):
            need(forced["draft_token_ids"][accepted] != labels[accepted], "first rejected draft matches actual label")
        indices = [history[p]["feature_row"] for p in range(anchor)]
        chunks = []
        prior = None
        ordinals = []
        for p in range(anchor):
            ordinal = history[p]["decode_ordinal"]
            if ordinal != prior:
                need(prior is None or ordinal > prior, "nonmonotonic committed decode ordinals")
                chunks.append(0)
                ordinals.append(ordinal)
                prior = ordinal
            chunks[-1] += 1
        need(sum(chunks) == anchor and all(0 < n <= 32 for n in chunks), "actual feature chunks violate native replay contract")
        included.append({"frontier_id": frontier_id, "category": category, "matched_set": matched_set,
            "prompt_id": prompt, "domain": frontier["domain"], "output_relative_anchor": out_anchor,
            "anchor": anchor, "anchor_token": forced["seed_token_id"], "labels": labels,
            "native_pos0": forced["pos0"], "source_task_id": meta["task_id"],
            "source_forced_round_line": forced_ref["line"], "source_trace_line": forced_ref["trace_line"],
            "source_round_index": forced["round_index"], "actual_per_arm_expectations": expectations,
            "prefix_token_ids_sha256": hashlib.sha256(canonical(prefix)).hexdigest(),
            "source_feature_row_ids": indices, "source_decode_ordinals": ordinals,
            "feature_chunks": chunks, "chain_id": prompt, "block_index": -1,
            "group_id": frontier.get("group_id"), "source_split": "existing_development_diagnostic",
            "identical_input_scope": "parent actual target feature rows/chunks and seed; reused across all replay models"})
    counts = {"included": dict(Counter(x["category"] for x in included)),
              "excluded": dict(Counter(x["category"] for x in excluded))}
    estimate = sum(x["anchor"] * FEATURE_DIM * 4 for x in included)
    free = shutil.disk_usage(output.parent).free
    need(free >= reserve_bytes + estimate + 16 * 1024**2,
         f"insufficient disk headroom: free={free}, features={estimate}, reserve={reserve_bytes}")
    # Input identity hashing is limited to bytes actually consumed. The large
    # quality-logit files are not read, copied, or claimed validated here.
    sources = {"preparer": pin(__file__), "capture_receipt": pin(receipt_path),
        "capture_summary": pin(capture / "requests" / "summary.json"),
        "selection": pin(analysis_path), "ledger_helper": pin(helper_path),
        "parent_feature_ledger": pin(ledger_path), "parent_features": pin(feature_path)}
    feature_stat = feature_path.stat()
    import numpy as np
    feature_rows = np.memmap(feature_path, dtype="<f4", mode="r", shape=(len(decoded), FEATURE_DIM))
    output.mkdir(exist_ok=False)
    try:
        for index, item in enumerate(included):
            item["global_case_index"] = index
            destination = output / "features" / f"case-{index:03d}.f32"
            destination.parent.mkdir(exist_ok=True)
            values = np.asarray(feature_rows[item["source_feature_row_ids"]], dtype="<f4")
            need(values.shape == (item["anchor"], FEATURE_DIM) and np.isfinite(values).all(), "invalid selected target features")
            with destination.open("xb") as stream:
                stream.write(values.tobytes(order="C"))
            item["features"] = pin(destination)
        after = feature_path.stat()
        need((feature_stat.st_dev, feature_stat.st_ino, feature_stat.st_size, feature_stat.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "source features changed while copying")
        batches = []
        for begin in range(0, len(included), 12):
            cases = [{**item, "case_index": index} for index, item in enumerate(included[begin:begin + 12])]
            batch_path = output / f"cases-{begin // 12:02d}.json"
            write(batch_path, {"schema": "group_transfer12_v1", "cases": cases})
            batches.append({**pin(batch_path), "cases": len(cases), "global_begin": begin})
        manifest = {"schema": "identical_input_frontier_replay_manifest_v1", "completed": True,
            "capture_preserved_original_decisions": True, "decision_identity": decision_proof,
            "sources": sources, "feature_dim": FEATURE_DIM, "feature_dtype": "little_endian_float32",
            "counts": counts, "included_cases": included, "excluded_cases": excluded, "batches": batches,
            "storage": {"estimated_feature_bytes": estimate, "written_feature_bytes": sum(x["features"]["bytes"] for x in included),
                        "free_before_bytes": free, "free_after_bytes": shutil.disk_usage(output).free,
                        "reserve_bytes": reserve_bytes, "reserve_origin": "assistant operational storage guard, not a user research budget"},
            "labels": "seven actual committed future output tokens; no padding, no relabeling, no raw quality-logit read",
            "capture_warmups_used": False, "new_training": False,
            "scope": "selected development counterfactual diagnostics; not held-out or serving performance",
            "limitations": ["Original hidden/cache states can differ across serving arms; replay deliberately fixes parent feature inputs/chunks.",
                "Actual per-arm expectations describe original visits, not a guarantee for the new identical-input replay.",
                "Unavailable original group IDs are not inferred from distinct prompt IDs.",
                "Capture receipt's quality-logit caption is not used as evidence; the raw quality-logit files are not inputs here."]}
        write(output / "manifest.json", manifest)
        return {"status": "FRONTIER_REPLAY_INPUTS_PREPARED", "capture_preserved_original_decisions": True,
                "decision_checks": decision_proof["checks"], "counts": counts, "batches": len(batches),
                "feature_bytes": estimate, "manifest": pin(output / "manifest.json")}
    except BaseException as error:
        write(output / "failure.json", {"completed": False, "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=HERE / "historical-frontier-capture-04")
    parser.add_argument("--original", type=Path, default=HERE / "historical-component-01")
    parser.add_argument("--selection", type=Path, default=HERE / "historical-frontiers.json")
    parser.add_argument("--output", type=Path, default=HERE / "frontier-replay-inputs")
    parser.add_argument("--reserve-gib", type=int, default=16)
    args = parser.parse_args()
    need(args.reserve_gib >= 0, "invalid storage reserve")
    result = prepare(args.capture, args.original, args.selection, args.output, args.reserve_gib * 1024**3)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "FRONTIER_REPLAY_INPUTS_PREPARED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

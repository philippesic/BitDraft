#!/usr/bin/env python3
"""Analyze completed same-input native replay using bounded JSON records only.

No raw-array scan, teacher-forced read, model loading, subprocess or GPU use.
Reported row argmax values are joined and their prefix metric is recomputed;
this does not independently recompute argmax from the underlying logit arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARMS = ("parent3504", "fresh_best")
CATEGORIES = ("damaged", "unchanged", "improved")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path, jsonl=False):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    value = [json.loads(line) for line in data.splitlines() if line.strip()] if jsonl else json.loads(data)
    return value, {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def analyze(run, inputs):
    receipt, receipt_pin = load(run / "receipt.json")
    manifest, manifest_pin = load(inputs / "manifest.json")
    need(receipt.get("completed") is True and manifest.get("completed") is True, "completed replay and input receipts required")
    need(manifest["capture_preserved_original_decisions"] is True, "capture decision identity was not proved")
    records = {(r["arm"], r["batch"]): r for r in receipt["records"]}
    need(len(records) == len(receipt["records"]) == 16, "16 unique model/batch executions required")
    need(set(a for a, _ in records) == set(ARMS), "unexpected replay arms")
    plans = []
    sources = {"receipt": receipt_pin, "input_manifest": manifest_pin, "plans": [], "native_rows": []}
    binary_paths = set()
    actual = {arm: {} for arm in ARMS}
    for batch, declaration in enumerate(manifest["batches"]):
        plan_path = inputs / f"cases-{batch:02d}.json"
        plan, plan_pin = load(plan_path)
        need(plan_pin["sha256"] == declaration["sha256"], "input batch source changed")
        need(plan["schema"] == "group_transfer12_v1" and len(plan["cases"]) == 12, "expected twelve cases per batch")
        need([c["case_index"] for c in plan["cases"]] == list(range(12)), "batch-local indices differ")
        sources["plans"].append(plan_pin)
        plans.extend(plan["cases"])
        for arm in ARMS:
            execution = records[arm, batch]
            need(execution["exit_code"] == 0, "native execution did not finish successfully")
            command = execution["command"]
            need(len(command) == 5 and Path(command[2]).resolve() == plan_path.resolve() and command[4] == "on",
                 "actual native command does not bind this input batch/callback mode")
            binary_paths.add(Path(command[0]).resolve())
            cell = run / arm / f"batch-{batch:02d}"
            need(Path(command[3]).resolve() == cell.resolve(), "actual output directory differs")
            rows, rows_pin = load(cell / "rows.jsonl", jsonl=True)
            need(len(rows) == 12, "native row coverage differs")
            sources["native_rows"].append({"arm": arm, "batch": batch, **rows_pin})
            for case, row in zip(plan["cases"], rows):
                need(row["case_index"] == case["case_index"] and row["anchor"] == case["anchor"] and
                     row["chain_id"] == case["chain_id"], "native case/anchor/history join differs")
                need(row["conditioning"] == "native_greedy" and row["computed_noise_rows"] == 7 and
                     row["callback_mode"] == "on" and row["draft_width_mask"] == 0,
                     "native replay conditioning/geometry differs")
                need(row["raw_floats"] == 7 * 151936 and len(row["rows"]) == len(case["labels"]) == 7,
                     "native vocabulary/label geometry differs")
                need([r["label"] for r in row["rows"]] == case["labels"], "native target labels differ")
                need(all(r["finite_support"] == 151936 and r["label_in_support"] for r in row["rows"]),
                     "reported native finite/support check differs")
                predictions = [r["argmax"] for r in row["rows"]]
                first = next((i for i, (x, y) in enumerate(zip(predictions, case["labels"])) if x != y), 7)
                need(first == row["first_rejection"], "native prefix metric differs from reported argmax sequence")
                expectation = case["actual_per_arm_expectations"][arm]
                need(expectation["nonterminal_full_D5_eligible"] and expectation["n_proposed"] == 5,
                     "serving comparison is not nonterminal full-D5")
                score = min(first, 5)
                expected = expectation["n_accepted"]
                need(0 <= expected <= 5, "invalid serving acceptance")
                record = {"frontier_id": case["frontier_id"], "category": case["category"],
                    "matched_set": case["matched_set"], "prompt_id": case["prompt_id"], "domain": case["domain"],
                    "output_relative_anchor": case["output_relative_anchor"], "batch": batch,
                    "case_index": case["case_index"], "serving_prefix5": expected,
                    "replay_prefix5": score, "replay_minus_serving": score - expected,
                    "fidelity_match": score == expected, "native_first_rejection7": first,
                    "source_rows": str(cell / "rows.jsonl"), "source_row_line": case["case_index"] + 1}
                if score != expected:
                    depth = min(score, expected)
                    summary = row["rows"][depth]
                    record["discrepancy"] = {"first_prefix_length_difference_depth": depth,
                        "actual_label": case["labels"][depth], "replayed_argmax": predictions[depth],
                        "reported_label_logit": summary["label_logit"], "reported_label_margin": summary["margin"],
                        "replayed_predictions_first5": predictions[:5], "labels_first5": case["labels"][:5],
                        "original_serving_trace_line": expectation["trace_line"],
                        "original_serving_task_id": expectation["task_id"],
                        "original_serving_round_index": expectation["round_index"]}
                need(case["frontier_id"] not in actual[arm], "duplicate replay frontier")
                actual[arm][case["frontier_id"]] = record
    need(len(binary_paths) == 1, "replay executable changed between batches")
    _unused, binary_pin = load_binary(next(iter(binary_paths)))
    sources["native_executable"] = binary_pin
    sources["analyzer"] = load_binary(__file__)[1]
    need(len(plans) == len({c["frontier_id"] for c in plans}) == 96, "96 distinct matched input cases required")
    need(set(actual[ARMS[0]]) == set(actual[ARMS[1]]), "replay arms cover different input frontiers")

    def fidelity(rows):
        rows = list(rows)
        return {"cases": len(rows), "matches": sum(r["fidelity_match"] for r in rows),
            "mismatches": sum(not r["fidelity_match"] for r in rows),
            "serving_prefix5_sum": sum(r["serving_prefix5"] for r in rows),
            "replay_prefix5_sum": sum(r["replay_prefix5"] for r in rows),
            "mean_absolute_prefix_error": sum(abs(r["replay_minus_serving"]) for r in rows) / len(rows) if rows else None,
            "discrepancies": [r for r in rows if not r["fidelity_match"]]}

    scores = {arm: {category: fidelity(r for r in actual[arm].values() if category == "all" or r["category"] == category)
                    for category in (*CATEGORIES, "all")} for arm in ARMS}
    paired = {}
    for category in (*CATEGORIES, "all"):
        ids = [key for key, r in actual["parent3504"].items() if category == "all" or r["category"] == category]
        rows = [(actual["parent3504"][key], actual["fresh_best"][key]) for key in ids]
        paired[category] = {
            "cases": len(rows),
            "serving_fresh_minus_parent": sum(f["serving_prefix5"] - p["serving_prefix5"] for p, f in rows),
            "same_input_replay_fresh_minus_parent": sum(f["replay_prefix5"] - p["replay_prefix5"] for p, f in rows),
            "replay_fresh_lower": sum(f["replay_prefix5"] < p["replay_prefix5"] for p, f in rows),
            "replay_tie": sum(f["replay_prefix5"] == p["replay_prefix5"] for p, f in rows),
            "replay_fresh_higher": sum(f["replay_prefix5"] > p["replay_prefix5"] for p, f in rows),
            "both_fidelity_match": sum(p["fidelity_match"] and f["fidelity_match"] for p, f in rows),
            "by_domain": {domain: {"cases": sum(p["domain"] == domain for p, f in rows),
                "replay_fresh_minus_parent": sum(f["replay_prefix5"] - p["replay_prefix5"] for p, f in rows if p["domain"] == domain)}
                for domain in ("code", "prose", "reasoning")}}
        if category == "damaged":
            faithful_parent = [(p, f) for p, f in rows if p["fidelity_match"]]
            paired[category]["restricted_to_faithful_parent"] = {
                "cases": len(faithful_parent),
                "fresh_lower": sum(f["replay_prefix5"] < p["replay_prefix5"] for p, f in faithful_parent),
                "tie": sum(f["replay_prefix5"] == p["replay_prefix5"] for p, f in faithful_parent),
                "fresh_higher": sum(f["replay_prefix5"] > p["replay_prefix5"] for p, f in faithful_parent)}
    need(all(scores[a][c]["cases"] == 32 for a in ARMS for c in CATEGORIES), "category coverage differs")
    return {"schema": "same_input_frontier_counterfactual_analysis_v1", "completed": True,
        "scope": "preselected development frontier diagnostics, not held-out or serving throughput",
        "sources": sources, "fidelity": scores, "paired_category_effects": paired,
        "case_scores": {a: list(r.values()) for a, r in actual.items()},
        "source_and_metric_joins_passed": True, "reported_argmax_rows_checked": 96 * 2 * 7,
        "raw_logit_arrays_recomputed": False, "teacher_forced_files_read": False,
        "preparation_contract": {"identical_features_chunks_and_seed": True,
            "input_manifest": manifest_pin, "capture_decisions_preserved": True,
            "features_not_rehashed_in_this_lightweight_analysis": True},
        "limitations": ["This independently recomputes capped first-rejection metrics from reported native row argmax values, not raw arrays.",
            "Greedy Markov replay reconstructs cache from the fixed parent feature/chunk/seed inputs; it is not the full serving loop.",
            "Teacher-forced outputs are not read or treated as a serving rollout.",
            "Parent and fresh replay are not perfectly faithful to their recorded serving prefix lengths; all discrepancies are retained.",
            "The damaged and matched control strata are purpose-selected development cases, not a population-random estimate.",
            "Persistence supports a weight/model tradeoff under fixed input, but does not uniquely identify the training objective, data selection or numerical mechanism.",
            "Common feature inputs do not prove equality of every original arm's hidden/cache state; replay fidelity bounds interpretation."]}


def load_binary(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return None, {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "frontier-counterfactual-01")
    parser.add_argument("--inputs", type=Path, default=HERE / "frontier-replay-inputs")
    parser.add_argument("--output", type=Path, default=HERE / "frontier-counterfactual-analysis.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run, args.inputs)
    encoded = canonical(result) + b"\n"
    if args.output.exists():
        need(args.output.read_bytes() == encoded, "existing output differs; preserve and choose a new output")
        action = "verified_existing"
    else:
        need(not args.verify, "verification output missing")
        with args.output.open("xb") as stream:
            stream.write(encoded)
        action = "created"
    print(json.dumps({"status": action, "output": str(args.output.resolve()),
        "fidelity": {arm: {category: {k: v for k, v in cell.items() if k != "discrepancies"}
                           for category, cell in categories.items()} for arm, categories in result["fidelity"].items()},
        "paired": result["paired_category_effects"], "raw_arrays_read": False}, indent=2))


if __name__ == "__main__":
    main()

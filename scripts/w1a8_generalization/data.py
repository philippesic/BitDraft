"""CPU-only joins of retained historical F16 native serving histories.

Shared feature arrays stay read-only. Every window names committed rows only;
rejected speculative inputs never become context or labels. These inherited
TRAIN groups are research development data, not an unseen release population.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

TARGET_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
FEATURE_DIM = 12800
TAPS = [2, 10, 18, 26, 34]
POLICIES = ("legacy_errors", "occupancy")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def authenticate(ref):
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("changed input: " + ref["path"])
    return Path(ref["path"])


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def ledger(rows):
    decoded, dispositions, qualities = {}, {}, set()
    for row in rows:
        event, key = row["event"], row["feature_row"]
        if type(key) is not int or key < 0:
            raise ValueError("invalid feature row")
        if event == "quality_logits":
            if key in qualities:
                raise ValueError("duplicate quality row")
            qualities.add(key)
            continue
        destination = {"decoded_row": decoded, "disposition": dispositions}.get(event)
        if destination is None or key in destination:
            raise ValueError("unknown or duplicate ledger event")
        destination[key] = row
    if set(decoded) != set(dispositions) or not qualities <= set(decoded):
        raise ValueError("incomplete disposition ledger")
    if sorted(decoded) != list(range(len(decoded))):
        raise ValueError("feature-row gap")
    tasks = defaultdict(dict)
    for key, row in decoded.items():
        disposition = dispositions[key]
        if row["task_id"] != disposition["task_id"]:
            raise ValueError("task disposition mismatch")
        retained = disposition["retained_input"]
        if type(retained) is not bool:
            raise ValueError("nonboolean disposition")
        if disposition["reason"] != ("accepted_prefix" if retained else "rejected_suffix"):
            raise ValueError("inconsistent disposition reason")
        if row.get("feature_dim") != FEATURE_DIM or row.get("target_layer_ids") != TAPS:
            raise ValueError("historical feature schema differs")
        if row.get("feature_dtype") != "float32_native_endian":
            raise ValueError("feature dtype differs")
        if not retained:
            continue
        position = row["position"]
        task = tasks[row["task_id"]]
        if type(position) is not int or position < 0 or position in task:
            raise ValueError("duplicate or invalid committed position")
        task[position] = row
    for task in tasks.values():
        if sorted(task) != list(range(len(task))):
            raise ValueError("committed position gap")
        tokens = [task[i]["token_id"] for i in range(len(task))]
        for position, row in task.items():
            if row["prefix_token_ids"] != tokens[:position + 1]:
                raise ValueError("committed prefix mismatch")
    return tasks, len(decoded)


def eligible_windows(rounds, committed, task_id):
    """Validate every complete proposal before selecting any training examples."""
    candidates = []
    seen = set()
    tokens = [committed[i]["token_id"] for i in range(len(committed))]
    for row in rounds:
        if row["task_id"] != task_id:
            raise ValueError("cross-task round join")
        index = row["round_index"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("duplicate or invalid round index")
        seen.add(index)
        proposal = row["draft_token_ids"]
        accepted = row["accepted_drafts"]
        if type(accepted) is not int or not 0 <= accepted <= len(proposal) <= 5:
            raise ValueError("invalid accepted/proposed counts")
        prefix = row["prefix_token_ids"] + [row["seed_token_id"]]
        anchor = len(prefix) - 1
        if not 0 <= anchor < len(tokens) or prefix != tokens[:anchor + 1] or row["pos0"] != anchor:
            raise ValueError("round prefix/native position mismatch")
        verifier = row["verifier_token_ids"]
        known = tokens[anchor + 1:anchor + accepted + 2]
        # The final emitted bonus token need not be decoded again, so its
        # feature/disposition row may be absent. Such a tail is never eligible
        # for training; still check every observed token and proposal decision.
        if (len(verifier) != accepted + 1 or verifier[:len(known)] != known
                or len(verifier) - len(known) > 1):
            raise ValueError("verifier/committed continuation mismatch")
        if proposal[:accepted] != verifier[:accepted]:
            raise ValueError("accepted proposal mismatch")
        if accepted < len(proposal) and proposal[accepted] == verifier[accepted]:
            raise ValueError("first rejection mismatch")
        if len(proposal) != 5 or anchor + 7 >= len(tokens):
            continue
        chunks = []
        previous = None
        for position in range(anchor):
            ordinal = committed[position]["decode_ordinal"]
            if ordinal != previous:
                chunks.append(0)
                previous = ordinal
            chunks[-1] += 1
        candidates.append({"anchor": anchor, "prefix_tokens": prefix,
                           "anchor_token": prefix[-1], "labels": tokens[anchor + 1:anchor + 8],
                           "accepted_drafts": accepted, "proposed": proposal,
                           "round_index": index, "native_pos0": row["pos0"],
                           "feature_chunks": chunks,
                           "feature_rows": [committed[i]["feature_row"] for i in range(anchor)]})
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("round-index gap")
    return sorted(candidates, key=lambda item: item["round_index"])


def select(candidates, policy, *, key, count=4, seed="w1a8-generalization-v1"):
    if policy not in POLICIES or type(count) is not int or count <= 0:
        raise ValueError("supported policy and positive count required")
    if len(candidates) < count:
        raise ValueError("insufficient eligible windows for matched-size task")
    if policy == "legacy_errors":
        selected = ([c for c in candidates if c["accepted_drafts"] < 5]
                    + [c for c in candidates if c["accepted_drafts"] == 5])[:count]
    else:
        # Fixed-seed hash priorities implement a reproducible uniform sample
        # without replacement. k/N is the inclusion probability over hash seeds.
        selected = sorted(candidates, key=lambda c: hashlib.sha256(
            f"{seed}:{key}:{c['round_index']}".encode()).digest())[:count]
    selected = sorted(selected, key=lambda item: item["round_index"])
    return [{**window, "eligible_count": len(candidates), "selected_count": count,
             "inclusion_probability": count / len(candidates) if policy == "occupancy" else None,
             "occupancy_weight": len(candidates) / count if policy == "occupancy" else 1.,
             "sampling_design": "fixed-seed hash uniform without replacement" if policy == "occupancy"
             else "earliest errors then saturated fillers; not probability sampling"}
            for window in selected]


def join(split, complete, tasks, rounds, feature_pin, *, policy, seed="w1a8-generalization-v1"):
    if (complete.get("status") != "SERVING_CAPTURE_COMPLETE" or complete.get("source_split") != "TRAIN"
            or complete.get("request_panels_used") is not False or split.get("request_panels_used") is not False):
        raise ValueError("authentic TRAIN capture required")
    cases, records = split["cases"], complete["records"]
    if len(cases) != len(records) or not cases:
        raise ValueError("complete capture/split coverage required")
    groups = {}
    for case in cases:
        if case["role"] not in ("fit", "development") or case.get("source_split") != "TRAIN":
            raise ValueError("TRAIN fit/development case required")
        previous = groups.setdefault(case["group_id"], case["role"])
        if previous != case["role"]:
            raise ValueError("original-group split leakage")
    by_task = defaultdict(list)
    for row in rounds:
        by_task[row["task_id"]].append(row)
    task_ids = [record["task_id"] for record in records]
    if len(set(task_ids)) != len(records) or set(task_ids) != set(tasks) or set(by_task) != set(task_ids):
        raise ValueError("capture task identity/coverage differs")
    windows, coverage = [], []
    for index, (case, record) in enumerate(zip(cases, records)):
        if case["index"] != index or record["case_index"] != index:
            raise ValueError("split/capture case order mismatch")
        task_id = record["task_id"]
        committed = tasks[task_id]
        prompt = case["prompt_tokens"]
        if prompt != [committed[i]["token_id"] for i in range(len(prompt))]:
            raise ValueError("case prompt/committed task mismatch")
        if record["rounds"] != len(by_task[task_id]):
            raise ValueError("capture round count mismatch")
        candidates = eligible_windows(by_task[task_id], committed, task_id)
        selected = select(candidates, policy, key=case["group_id"] + ":" + str(task_id), seed=seed)
        coverage.append({"group_id": case["group_id"], "role": case["role"], "domain": case["domain"],
                         "task_id": task_id, "eligible": len(candidates), "selected": len(selected),
                         "eligible_full5": sum(w["accepted_drafts"] == 5 for w in candidates),
                         "selected_full5": sum(w["accepted_drafts"] == 5 for w in selected)})
        for window in selected:
            windows.append({**window, "case_index": len(windows), "features": feature_pin,
                            "group_id": case["group_id"], "domain": case["domain"], "role": case["role"],
                            "chain_id": case["chain_id"], "block_index": -1, "source_case_index": index,
                            "source_task_id": task_id, "source_split": "TRAIN", "reused_TRAIN": True,
                            "ancestry": case.get("ancestry"), "policy": policy})
    return windows, coverage


def feature_array(window):
    """Read only committed row indices from an authenticated shared capture."""
    import numpy as np
    rows = window.get("feature_rows")
    if rows is None:
        array = np.fromfile(window["features"]["path"], dtype="<f4")
        if array.size != window["anchor"] * FEATURE_DIM:
            raise ValueError("materialized feature size differs")
        if not np.isfinite(array).all():
            raise ValueError("nonfinite materialized features")
        return array.reshape(window["anchor"], 5, 2560)
    if len(rows) != window["anchor"] or any(type(i) is not int or i < 0 for i in rows) or len(set(rows)) != len(rows):
        raise ValueError("invalid committed feature indices")
    size = Path(window["features"]["path"]).stat().st_size
    if size % (4 * FEATURE_DIM) or (rows and max(rows) >= size // (4 * FEATURE_DIM)):
        raise ValueError("shared feature index/shape differs")
    raw = np.memmap(window["features"]["path"], dtype="<f4", mode="r", shape=(size // (4 * FEATURE_DIM), FEATURE_DIM))
    value = np.asarray(raw[rows]).reshape(window["anchor"], 5, 2560)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite committed features")
    return value


def block(window, manifest_pin):
    """Existing seven-row BlockBatch contract, preserving native chunk history."""
    import numpy as np
    from w1a1_eagle.block_data import BlockBatch, token_sha256
    features = feature_array(window)
    labels = np.asarray(window["labels"], dtype=np.int64)
    if labels.shape != (7,) or sum(window["feature_chunks"]) != window["anchor"]:
        raise ValueError("seven labels and exact native history required")
    noise = np.full(7, 151669, dtype=np.int64)
    noise[0] = window["anchor_token"]
    tokens = window["prefix_tokens"]
    return BlockBatch("dspark", window["chain_id"], -1, features, tuple(tokens), noise,
                      np.arange(window["anchor"], window["anchor"] + 7, dtype=np.int64), labels,
                      np.ones(7, dtype=bool), np.ones((7, window["anchor"] + 7), dtype=bool), None,
                      tuple(token_sha256(tokens + labels[:i].tolist()) for i in range(7)),
                      np.asarray([window["anchor_token"], *window["labels"][:6]], dtype=np.int64),
                      TARGET_SHA256, manifest_pin["sha256"])


def materialize(manifest, output, *, max_bytes=512 * 1024**2):
    """Bounded adapter to the original actor/native-replay file format."""
    manifest = read(authenticate(manifest)) if "windows" not in manifest else manifest
    required = sum(w["anchor"] * FEATURE_DIM * 4 for w in manifest["windows"])
    if type(max_bytes) is not int or max_bytes <= 0 or required > max_bytes:
        raise ValueError("materialization exceeds declared byte cap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    refs = {w["features"]["path"]: w["features"] for w in manifest["windows"]}
    for ref in refs.values():
        authenticate(ref)
    windows = []
    for index, window in enumerate(manifest["windows"]):
        path = output / f"features-{index:04d}.f32"
        feature_array(window).astype("<f4").tofile(path)
        windows.append({**{k: v for k, v in window.items() if k != "feature_rows"}, "features": pin(path)})
    result = {**manifest, "windows": windows, "materialized_bytes": required}
    write(output / "manifest.json", result)
    return pin(output / "manifest.json")


def prepare(source, output, *, policy="all", seed="w1a8-generalization-v1"):
    source, output = Path(source), Path(output)
    paths = {"split": source / "split.json", "capture": source / "capture05/complete.json",
             "features": source / "capture05/capture.target_features.f32",
             "ledger": source / "capture05/capture.target_features.jsonl", "rounds": source / "capture05/rounds.jsonl"}
    refs = {name: pin(path) for name, path in paths.items()}
    complete, split = read(paths["capture"]), read(paths["split"])
    for name, filename in (("features", "capture.target_features.f32"), ("ledger", "capture.target_features.jsonl")):
        if refs[name] != complete["files"][filename]:
            raise ValueError("retained capture pin differs: " + name)
    with paths["ledger"].open() as stream:
        tasks, rows = ledger(json.loads(line) for line in stream)
    if paths["features"].stat().st_size != rows * FEATURE_DIM * 4:
        raise ValueError("raw feature rows/ledger size differs")
    with paths["rounds"].open() as stream:
        rounds = [json.loads(line) for line in stream]
    policies = POLICIES if policy == "all" else (policy,)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in policies:
        windows, coverage = join(split, complete, tasks, rounds, refs["features"], policy=name, seed=seed)
        directory = output / name
        directory.mkdir()
        manifest = {"schema": "w1a8_shared_serving_windows_v1", "policy": name, "seed": seed,
                    "windows": windows, "ancestry": refs, "reused_TRAIN": True, "source_split": "TRAIN",
                    "not_pristine_holdout": True, "request_panels_used": False,
                    "target_sha256": TARGET_SHA256, "target_weights": "F16", "target_kv": "F16",
                    "soft_teacher_logits": False, "historical_track": "Qwen3 DSpark F16; not Qwen3.5"}
        fit = [w for w in windows if w["role"] == "fit"]
        if len(fit) % 2:
            raise ValueError("even fit windows required for two-block transactions")
        order = {"schema": "w1a8_serving_pairs_v1", "pairs": [[i, i + 1] for i in range(0, len(fit), 2)],
                 "policy": name, "source_split": "TRAIN"}
        write(directory / "manifest.json", manifest)
        write(directory / "order.json", order)
        write(directory / "coverage.json", {"tasks": coverage,
              "selected_by_role_and_acceptance": {f"{role}:{accepted}": count for (role, accepted), count
                  in sorted(Counter((w["role"], w["accepted_drafts"]) for w in windows).items())}})
        results[name] = {key: pin(directory / (key + ".json")) for key in ("manifest", "order", "coverage")}
    for ref in refs.values():
        authenticate(ref)
    write(output / "prepared.json", {"schema": "w1a8_serving_prepared_v1", "policies": results,
                                     "GPU_used": False, "teacher_soft_logits_required": False})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--policy", choices=("all", *POLICIES), default="all")
    command.add_argument("--seed", default="w1a8-generalization-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, policy=args.policy, seed=args.seed)))


if __name__ == "__main__":
    main()

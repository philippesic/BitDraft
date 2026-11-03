"""Prepare a small, source-indexed existing-window native diagnostic packet.

No model is loaded or copied. Run after timing quiet windows. Native parent
calibration is mandatory before candidate replay; this is not full serving,
untouched generalization, or an all-parent-success qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

from modern_collect import atomic, pin, read, require

SCHEMA = "w1a8_actual_native_cases_v1"
TARGET = "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"
PARENT = "6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048"


def budget(path):
    for root in (Path(path), Path("/mnt/c")):
        require(
            shutil.disk_usage(root).free >= 2 * 1024**3 + 64 * 1024**2,
            "storage floor: " + str(root),
        )


def checked(ref):
    actual = pin(ref["path"])
    require(
        actual["sha256"] == ref["sha256"] and actual["bytes"] == ref["bytes"],
        "source pin changed: " + ref["path"],
    )
    return Path(actual["path"])


def classify(window, parent, candidate):
    require(
        parent["labels"] == candidate["labels"] == window["labels"]
        and parent["native_prefix"] == candidate["native_prefix"] == window["native_prefix"]
        and parent["role"] == candidate["role"] == window["role"],
        "readout/window join differs",
    )
    p, c = parent["actor_prefix"], candidate["actor_prefix"]
    require(
        type(p) is int
        and type(c) is int
        and 0 <= p <= window["max_depth"]
        and 0 <= c <= window["max_depth"],
        "invalid GPU reference prefix",
    )
    return "known_lost" if c < p else "known_gained" if c > p else "unchanged"


def choose(windows, parent_rows, candidate_rows, limit, seed):
    require(1 <= limit <= 128, "bounded selection must contain1..128 windows")
    pools = defaultdict(list)
    for w in windows:
        if w["id"] not in parent_rows or w["id"] not in candidate_rows:
            continue  # Missing sampled fit readouts are not fabricated zero scores.
        require(w["role"] in ("fit", "development"), "only existing fit/development allowed")
        label = classify(w, parent_rows[w["id"]], candidate_rows[w["id"]])
        key = (label, w["role"], bool(w["native_prefix"]), w["domain"])
        pools[key].append(w)
    require(pools, "no matched existing GPU readout windows")
    for pool in pools.values():
        pool.sort(key=lambda w: hashlib.sha256((seed + "\0" + w["id"]).encode()).hexdigest())
    # Round-robin strata preserve both roles, successes, lost/gained and controls.
    keys = sorted(pools)
    selected = []
    while len(selected) < limit and any(pools.values()):
        for key in keys:
            if pools[key] and len(selected) < limit:
                selected.append((pools[key].pop(0), key[0]))
    return selected


def prepare(args):
    begin = time.perf_counter()
    require(not args.output.exists(), "output already exists")
    budget(args.output.parent)
    m = read(args.manifest)
    require(
        m.get("schema") == "w1a8_actual_source_indexed_round_views_v1"
        and m["target"]["sha256"] == TARGET
        and m["parent"]["sha256"] == PARENT,
        "actual original-runtime manifest ancestry differs",
    )
    build = read(args.build)
    require(
        build.get("status") == "CPU_BUILD_CHECKS_ONLY"
        and build.get("original_runtime_only") is True,
        "original-runtime helper build required",
    )
    for ref in [build["executable"]] + build["runtime_libraries"]:
        checked(ref)
    parent_readout = read(args.parent_readout)
    candidate_paths = args.candidate_readout
    require(
        len({p.parent.resolve() for p in candidate_paths}) == 1,
        "candidate fit/development readouts must come from the same frozen training run",
    )

    def index(rows):
        result = {r["id"]: r for r in rows}
        require(len(result) == len(rows), "duplicate GPU readout IDs")
        return result

    parents = index(parent_readout["rows"])
    candidates = index([r for p in candidate_paths for r in read(p)["rows"]])
    manifest_pin = pin(args.manifest)
    # The same-GPU reference contract must be the dataset and fixed initializer.
    parent_contract = read(args.parent_readout.with_name("freeze.json"))
    candidate_contract = read(candidate_paths[0].with_name("freeze.json"))
    require(
        parent_contract["manifest"] == candidate_contract["manifest"] == manifest_pin,
        "GPU readouts do not bind this actual manifest",
    )
    require(parent_contract["actor"] == candidate_contract["actor"], "GPU actor arithmetic differs")
    model = read(args.candidate)
    require(
        isinstance(model, dict)
        and {"path", "sha256", "bytes"} <= model.keys()
        and model["sha256"] != PARENT
        and Path(model["path"]).is_absolute(),
        "candidate model pin JSON required",
    )
    # Bind the candidate GGUF to the completed training run that produced readout.
    finish = read(candidate_paths[0].with_name("result.json"))
    require(
        finish.get("status") == "TRAINED_EXPORTED_NOT_NATIVE_QUALIFIED"
        and finish["model"] == model,
        "candidate readout/model export differs",
    )
    selected = choose(m["windows"], parents, candidates, args.limit, args.seed)
    if args.limit >= 48:
        require(
            {w["role"] for w, _ in selected} == {"fit", "development"},
            "representative panel needs both fit/development; supply both candidate readout files",
        )
    files, file_map, row_table, row_map, cases = [], {}, [], {}, []
    for w, label in selected:
        anchor, depth = w["anchor"], w["max_depth"]
        require(
            type(anchor) is int
            and 0 <= anchor
            and anchor + depth <= 4096
            and len(w["prefix_tokens"]) == len(w["prefix_raw_rows"]) == anchor
            and depth in (1, 2)
            and len(w["labels"]) == depth,
            "actual window geometry differs",
        )
        for key in ("native_success_mask", "native_supervision_reachable_mask"):
            require(
                len(w[key]) == depth and all(type(v) is bool for v in w[key]), "native mask differs"
            )
        require(
            w["native_success_mask"] == [d < w["native_prefix"] for d in range(depth)],
            "native protection prefix differs",
        )
        require(
            all(
                type(t) is int and 0 <= t < 248320
                for t in w["prefix_tokens"] + [w["seed_token"]] + w["labels"]
            ),
            "invalid target token ID",
        )
        for rid in w["prefix_raw_rows"] + [w["seed_raw_row"]]:
            require(type(rid) is int and 0 <= rid < len(m["raw_rows"]), "invalid source raw row")
            if rid in row_map:
                continue
            raw = m["raw_rows"][rid]
            fid = raw["file"]
            source = m["raw_files"][fid]
            require(
                len(source["shape"]) == 2
                and source["shape"][1] == 7680
                and source["bytes"] == source["shape"][0] * 7680 * 4
                and 0 <= raw["row"] < source["shape"][0],
                "raw source geometry differs",
            )
            if fid not in file_map:
                checked(source)
                file_map[fid] = len(files)
                files.append(source)
            row_map[rid] = len(row_table)
            row_table.append(
                [rid, file_map[fid], raw["row"], raw["raw_sha256"], raw["target_position"]]
            )
        fields = (
            "id",
            "role",
            "domain",
            "group",
            "group_id",
            "round",
            "anchor",
            "seed_token",
            "labels",
            "max_depth",
            "native_prefix",
            "native_verified_prefix",
            "native_proposals",
            "native_success_mask",
            "native_supervision_reachable_mask",
            "prefix_tokens",
            "prefix_raw_rows",
            "seed_raw_row",
        )
        cases.append(
            {
                **{k: w[k] for k in fields},
                "classification": label,
                "gpu_parent_prefix": parents[w["id"]]["actor_prefix"],
                "gpu_candidate_prefix": candidates[w["id"]]["actor_prefix"],
            }
        )
    # Reuse an already frozen absolute mapping; no GGUF/model is opened here.
    map_path = args.draft_map.resolve()
    mapping = [int(x) for x in map_path.read_text().split()]
    require(
        len(mapping) == len(set(mapping)) == 50000 and min(mapping) >= 0 and max(mapping) < 248320,
        "absolute draft mapping differs",
    )
    require(
        hashlib.sha256(struct.pack("<50000q", *mapping)).hexdigest()
        == parent_contract["actor"]["semantics"]["absolute_d2t_sha256"],
        "native mapping differs from frozen GPU actor map",
    )
    packet = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "manifest": manifest_pin,
        "build": pin(args.build),
        "helper": build["executable"],
        "runtime_libraries": build["runtime_libraries"],
        "cpu_library": build["cpu_library"],
        "cuda_library": build["cuda_library"],
        "models": {"parent": m["parent"], "candidate": model},
        "target": m["target"],
        "draft_map": pin(map_path),
        "files": files,
        "raw_row_columns": ["source_id", "file", "row", "sha256", "target_position"],
        "raw_rows": row_table,
        "cases": cases,
        "selection": {
            "seed": args.seed,
            "requested": args.limit,
            "selected": len(cases),
            "parent_readout": pin(args.parent_readout),
            "candidate_readouts": [pin(p) for p in candidate_paths],
            "parent_contract": pin(args.parent_readout.with_name("freeze.json")),
            "candidate_contract": pin(candidate_paths[0].with_name("freeze.json")),
            "candidate_finish": pin(candidate_paths[0].with_name("result.json")),
            "classes": dict(Counter(c["classification"] for c in cases)),
            "roles": dict(Counter(c["role"] for c in cases)),
            "groups": len({c["group"] for c in cases}),
            "native_reference_tokens": sum(c["native_prefix"] for c in cases),
        },
    }
    blob = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode()
    require(
        len(blob) <= 2 * 1024**2,
        "case metadata exceeds2MiB; reduce explicit limit, never silently drop cases",
    )
    budget(args.output.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cases.json").write_bytes(blob)
    atomic(
        args.output / "prepare.json",
        {
            "status": "PREPARED_DIAGNOSTIC_NOT_NATIVE_QUALIFIED",
            "source": pin(__file__),
            "packet": pin(args.output / "cases.json"),
            "array_bytes_copied": 0,
            "tensor_files_written": 0,
            "selection": packet["selection"],
            "wall_seconds": time.perf_counter() - begin,
            "scope": "existing source-indexed actor windows only; parent native calibration required; no serving acceptance/throughput or all549 claim",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "manifest",
        "parent-readout",
        "build",
        "candidate",
        "draft-map",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--candidate-readout",
        type=Path,
        required=True,
        action="append",
        help="repeat for after-fit.json and after-development.json from one candidate run",
    )
    parser.add_argument("--limit", type=int, default=96)
    parser.add_argument("--seed", default="actual-native-diagnostic-v1")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

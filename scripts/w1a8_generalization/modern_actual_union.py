"""Audited source-index union of original-initializer actual-serving manifests.

No capture, model loading, array copying or training. A distinct union schema and
reader preserve honest multiple-capture ancestry; the original V1 reader remains
strict and deliberately cannot load this output. CLI storage is unique C: only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from modern_actual_data import (
    HIDDEN,
    PARENT_SHA256,
    TARGET_SHA256,
    WIDTH,
    ActualDataset,
    _Rows,
)
from modern_actual_data import (
    SCHEMA as MEMBER_SCHEMA,
)

SCHEMA = "w1a8_actual_source_indexed_union_v1"
STATUS = "SOURCE_INDEX_UNION_PREPARED_NOT_TRAINING_OR_NATIVE_ADMISSION"
ARTIFACTS = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts")
PROJECT = Path("/home/philip/binary-eagle-decoding")
FLOOR = 2 * 1024**3 + 64 * 1024**2
MAX_METADATA = 192 * 1024**2
MAX_FILE_METADATA = 64 * 1024**2
MAX_OUTPUT = 96 * 1024**2
MAX_HASH_BYTES = 4 * 1024**3
MAX_ROW_BYTES = 2 * 1024**3
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(path, *, limit=MAX_FILE_METADATA):
    path = Path(path).resolve()
    require(path.stat().st_size <= limit, "metadata file too large")
    value = path.read_bytes()
    return {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def opaque(ref):
    require(
        isinstance(ref, dict)
        and isinstance(ref.get("path"), str)
        and Path(ref["path"]).is_absolute()
        and type(ref.get("bytes")) is int
        and ref["bytes"] > 0
        and isinstance(ref.get("sha256"), str)
        and HEX.fullmatch(ref["sha256"]),
        "invalid absolute file identity",
    )


class Audit:
    """Streaming bounded hashes and single-row reads; never materialize dense arrays."""

    def __init__(self, guard=lambda: None):
        self.guard = guard
        self.checked = {}
        self.hash_bytes = self.row_bytes = self.metadata_bytes = 0

    def file(self, ref):
        opaque(ref)
        path = Path(ref["path"])
        require(
            path.suffix not in (".gguf", ".safetensors", ".pt", ".npz"),
            "model/checkpoint access forbidden",
        )
        self.guard()
        stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        require(stat.st_size == ref["bytes"], "source file size changed: " + str(path))
        if str(path) in self.checked:
            previous, identity = self.checked[str(path)]
            require(
                stamp == previous and (ref["sha256"], ref["bytes"]) == identity,
                "source changed or conflicting file pin",
            )
            return path
        require(
            len(self.checked) < 40000 and self.hash_bytes + stat.st_size <= MAX_HASH_BYTES,
            "source hashing budget exceeded",
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024**2):
                self.guard()
                digest.update(chunk)
        after = path.stat()
        require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == stamp
            and digest.hexdigest() == ref["sha256"],
            "source hash changed: " + str(path),
        )
        self.hash_bytes += stat.st_size
        self.checked[str(path)] = (stamp, (ref["sha256"], ref["bytes"]))
        return path

    def read(self, ref):
        require(
            ref["bytes"] <= MAX_FILE_METADATA
            and self.metadata_bytes + ref["bytes"] <= MAX_METADATA,
            "metadata budget exceeded",
        )
        path = self.file(ref)
        self.metadata_bytes += ref["bytes"]
        return json.loads(path.read_bytes())

    def row(self, spec, row, width, expected):
        require(
            isinstance(spec.get("shape"), list)
            and len(spec["shape"]) == 2
            and all(type(n) is int and n > 0 for n in spec["shape"])
            and spec["shape"][1] == width
            and spec["bytes"] == spec["shape"][0] * width * 4
            and type(row) is int
            and 0 <= row < spec["shape"][0]
            and isinstance(expected, str)
            and HEX.fullmatch(expected),
            "source row geometry/hash differs",
        )
        path = self.file(spec)
        require(self.row_bytes + width * 4 <= MAX_ROW_BYTES, "row read budget exceeded")
        with path.open("rb") as stream:
            stream.seek(row * width * 4)
            value = stream.read(width * 4)
        self.row_bytes += len(value)
        require(
            len(value) == width * 4
            and hashlib.sha256(value).hexdigest() == expected
            and np.isfinite(np.frombuffer(value, dtype="<f4")).all(),
            "source row bytes/hash/finite differ",
        )


def _runtime(member, audit):
    source = {ref["path"]: ref for ref in member["source_pins"]}
    capture = audit.read(member["capture"])
    freeze_path = str(Path(member["capture"]["path"]).with_name("freeze.json"))
    require(freeze_path in source, "capture freeze must be an admitted source pin")
    freeze = audit.read(source[freeze_path])
    require(
        freeze["target"] == member["target"]
        and freeze["parent"] == member["parent"]
        and freeze["depth"] == 2
        and capture.get("training_updates") == 0
        and capture.get("status") == "ACTUAL_PARENT_COLLECTION_COMPLETE_WITH_ACCOUNTED_FAILURES",
        "capture parent/target/depth/admission differs",
    )
    audit.file(freeze["trace_build"])
    inputs = {ref["path"]: ref for ref in freeze["input_pins"]}
    runtime_pins = []
    for ref in freeze["input_pins"]:
        opaque(ref)
        if Path(ref["path"]).suffix == ".gguf":
            require(
                ref in (member["target"], member["parent"]),
                "trained/unknown model in capture inputs",
            )
        else:
            audit.file(ref)
            runtime_pins.append(ref)
    admitted = {g["id"]: g for g in capture["groups"]}
    signatures, commands = [], []
    expected = {
        "-c": "4096",
        "-np": "1",
        "-b": "512",
        "-ub": "512",
        "-t": "4",
        "-fa": "on",
        "--cache-type-k": "f16",
        "--cache-type-v": "f16",
        "--cache-type-k-draft": "f16",
        "--cache-type-v-draft": "f16",
        "--spec-type": "draft-eagle3",
        "--spec-draft-n-max": "2",
        "--spec-draft-n-min": "0",
        "--spec-draft-p-min": "0",
        "--spec-draft-sampling": "greedy",
        "-m": member["target"]["path"],
        "-md": member["parent"]["path"],
    }
    for group in member["groups"]:
        g = admitted.get(group["id"], {})
        require(
            g.get("status") == "ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS"
            and g.get("token_text_stop_parity") is True
            and all(g.get(k) == group[k] for k in ("role", "domain", "group_id"))
            and g.get("source_identity", {}).get("source_split") == "TRAIN",
            "group is not prior TRAIN-source actual admission",
        )
        command_pin = pin(Path(member["capture"]["path"]).parent / group["id"] / "command.json")
        command = audit.read(command_pin)
        argv = list(command["argv"])
        for flag, value in expected.items():
            require(
                argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
                "capture runtime option differs: " + flag,
            )
        require(argv.count("--port") == 1, "one local port required")
        argv[argv.index("--port") + 1] = "<per-process-port>"
        env = dict(command["environment"])
        require(
            env.get("GGML_CUDA_DECODE_CONSISTENT") == "32"
            and env.get("GGML_W1AX_ACT_BITS") == "8"
            and env.get("CUDA_VISIBLE_DEVICES") == "0"
            and env.get("OMP_NUM_THREADS") == "4",
            "correctness/native precision environment differs",
        )
        for binary in (argv[0], env.get("LD_PRELOAD"), env.get("W1A8_TRACE_LLAMA")):
            require(binary in inputs, "runtime binary missing frozen input pin")
        env.pop("W1A8_TRACE_DIR", None)
        signatures.append({"argv": argv, "environment": env})
        commands.append(command_pin)
    require(
        signatures and all(s == signatures[0] for s in signatures),
        "capture groups mix runtime settings",
    )
    # Data/roster metadata pins differ legitimately; runtime identity uses exact
    # commands and trace build plus the referenced executable/library identities.
    binary_pins = [
        ref
        for ref in runtime_pins
        if ".so" in Path(ref["path"]).name or ref["path"] == signatures[0]["argv"][0]
    ]
    return {
        "trace_build": freeze["trace_build"],
        "command": signatures[0],
        "binaries": sorted(binary_pins, key=lambda x: x["path"]),
    }, commands


def _validate_member(member, audit):
    require(
        member.get("schema") == MEMBER_SCHEMA
        and member["target"]["sha256"] == TARGET_SHA256
        and member["parent"]["sha256"] == PARENT_SHA256,
        "only original-target/original-parent V1 histories allowed",
    )
    opaque(member["target"])
    opaque(member["parent"])
    require(
        len(member["windows"]) <= 12000
        and len(member["raw_rows"]) <= 60000
        and len(member["groups"]) <= 240,
        "member dimensions exceed bounded union",
    )
    for ref in member["source_pins"]:
        audit.file(ref)
    runtime, commands = _runtime(member, audit)
    groups = {g["id"]: g for g in member["groups"]}
    require(len(groups) == len(member["groups"]), "duplicate member group IDs")
    declared = {ref["path"]: (ref["sha256"], ref["bytes"]) for ref in member["source_pins"]}
    for group in groups.values():
        for key in ("protection", "response", "trace_manifest"):
            ref = group[key]
            require(
                declared.get(ref["path"]) == (ref["sha256"], ref["bytes"]),
                "group proof missing from admitted source pins",
            )
    for row in member["raw_rows"]:
        require(
            type(row["file"]) is int and 0 <= row["file"] < len(member["raw_files"]),
            "unknown raw file",
        )
        require(row["group"] in groups, "raw row belongs to unknown group")
        audit.row(member["raw_files"][row["file"]], row["row"], WIDTH, row["raw_sha256"])
        audit.row(row["native_fc_file"], row["native_fc_row"], HIDDEN, row["native_fc_sha256"])
    totals, windows = Counter(), set()
    group_windows, group_protections = Counter(), Counter()
    used_tokens = set()
    for w in member["windows"]:
        require(w["id"] not in windows, "duplicate member window ID")
        windows.add(w["id"])
        g = groups[w["group"]]
        require(
            w["role"] in ("fit", "development")
            and all(w[k] == g[k] for k in ("role", "domain", "group_id")),
            "window/group role ancestry differs",
        )
        depth = w["max_depth"]
        require(
            type(depth) is int
            and 1 <= depth <= 2
            and len(w["labels"]) == depth
            and type(w["anchor"]) is int
            and len(w["prefix_tokens"]) == len(w["prefix_raw_rows"]) == w["anchor"] <= 4096
            and all(
                type(i) is int and 0 <= i < len(member["raw_rows"])
                for i in w["prefix_raw_rows"] + [w["seed_raw_row"]]
            ),
            "window source/depth geometry differs",
        )
        require(
            all(
                member["raw_rows"][i]["group"] == w["group"]
                for i in w["prefix_raw_rows"] + [w["seed_raw_row"]]
            ),
            "window mixes raw features across source groups",
        )
        require(
            len(w["native_supervision_reachable_mask"]) == depth
            and type(w["native_prefix"]) is int
            and all(type(v) is bool for v in w["native_supervision_reachable_mask"])
            and all(type(v) is bool for v in w["native_success_mask"])
            and w["native_success_mask"] == [d < w["native_prefix"] for d in range(depth)]
            and [p["proposal_depth"] for p in w["native_protections"]]
            == list(range(w["native_prefix"]))
            and 0 <= w["native_prefix"] <= depth,
            "native masks/protections differ",
        )
        for p in w["native_protections"]:
            require(
                p["token"] == w["labels"][p["proposal_depth"]]
                and p["response_index"] == w["response_label_start"] + p["proposal_depth"],
                "protected label/response join differs",
            )
        totals[w["role"]] += w["native_prefix"]
        group_windows[w["group"]] += 1
        group_protections[w["group"]] += w["native_prefix"]
        tokens = w["prefix_tokens"] + [w["seed_token"]] + w["labels"]
        require(
            all(type(token) is int and 0 <= token < 248320 for token in tokens),
            "invalid target token ID",
        )
        used_tokens.update(tokens)
    require(dict(totals) == member["native_protections"], "native protection denominator changed")
    require(
        all(
            g["windows"] == group_windows[g["id"]]
            and g["native_protections"] == group_protections[g["id"]]
            for g in groups.values()
        ),
        "group window/protection counts differ",
    )
    require(used_tokens <= set(member["required_embedding_ids"]), "required token coverage dropped")
    for crossed in member["crossed_fc_source_rows"]:
        require(
            type(crossed["raw_row"]) is int and 0 <= crossed["raw_row"] < len(member["raw_rows"]),
            "unknown crossed raw row",
        )
        require(
            crossed["native_fc_row_sha256"]
            == member["raw_rows"][crossed["raw_row"]]["native_fc_sha256"],
            "crossed FC provenance differs",
        )
        audit.row(
            crossed["input_file"], crossed["decoder_row"], HIDDEN, crossed["input_row_sha256"]
        )
    for token, row in member["embedding_rows"].items():
        require(
            str(int(token)) == token
            and 0 <= int(token) < 248320
            and type(row["file"]) is int
            and 0 <= row["file"] < len(member["embedding_files"]),
            "invalid embedding token/file",
        )
        audit.row(member["embedding_files"][row["file"]], row["row"], HIDDEN, row["sha256"])
    require(
        member["missing_embedding_ids"]
        == sorted(
            set(member["required_embedding_ids"]) - {int(t) for t in member["embedding_rows"]}
        ),
        "member embedding coverage differs",
    )
    return runtime, commands


def build_union(manifest_paths, *, audit=None):
    """Read/audit members and construct new metadata; no files are copied/written."""
    require(2 <= len(manifest_paths) <= 4, "two to four explicit source manifests required")
    audit = audit or Audit()
    refs = [pin(path) for path in manifest_paths]
    require(len({r["path"] for r in refs}) == len(refs), "same manifest path repeated")
    members = [audit.read(ref) for ref in refs]
    require(
        sum(len(m["windows"]) for m in members) <= 24000
        and sum(len(m["raw_rows"]) for m in members) <= 120000
        and sum(w["anchor"] for m in members for w in m["windows"]) <= 2000000,
        "aggregate window/row/prefix metadata budget exceeded",
    )
    first = members[0]
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "target": first["target"],
        "parent": first["parent"],
        "raw_files": [],
        "raw_rows": [],
        "windows": [],
        "groups": [],
        "embedding_files": [],
        "embedding_rows": {},
        "source_pins": [],
        "crossed_fc_source_rows": [],
        "members": [],
        "group_provenance": {},
        "raw_row_provenance": {},
        "embedding_provenance": {},
    }
    caches = {k: {} for k in ("raw_files", "raw_rows", "embedding_files", "source_pins")}
    by_window, by_group, component_group, group_window_ids = {}, {}, {}, {}
    required = set()
    runtime = None

    def intern(key, value):
        signature = encoded(value)
        if signature not in caches[key]:
            caches[key][signature] = len(result[key])
            result[key].append(copy.deepcopy(value))
        return caches[key][signature]

    for number, (ref, member) in enumerate(zip(refs, members, strict=True)):
        require(
            member["target"] == first["target"]
            and member["parent"] == first["parent"]
            and member["rollout_contract"] == first["rollout_contract"]
            and member["source_position_contract"] == first["source_position_contract"],
            "member model/forward ancestry differs",
        )
        current_runtime, commands = _validate_member(member, audit)
        require(
            runtime is None or current_runtime == runtime, "members mix native runtime ancestry"
        )
        runtime = current_runtime
        maps = {}
        for key in ("raw_files", "embedding_files"):
            maps[key] = [intern(key, entry) for entry in member[key]]
        maps["raw_rows"] = []
        for old_index, entry in enumerate(member["raw_rows"]):
            row = copy.deepcopy(entry)
            row["file"] = maps["raw_files"][row["file"]]
            # Only known storage provenance is excluded from semantic identity.
            semantics = {
                k: v
                for k, v in row.items()
                if k not in ("file", "row", "native_fc_file", "native_fc_row")
            }
            signature = encoded(semantics)
            if signature not in caches["raw_rows"]:
                caches["raw_rows"][signature] = len(result["raw_rows"])
                result["raw_rows"].append(row)
            index = caches["raw_rows"][signature]
            maps["raw_rows"].append(index)
            result["raw_row_provenance"].setdefault(str(index), []).append(
                {"member": number, "original_index": old_index, "row": copy.deepcopy(entry)}
            )
        for group in member["groups"]:
            ids = {w["id"] for w in member["windows"] if w["group"] == group["id"]}
            require(
                component_group.get(group["group_id"], group["id"]) == group["id"],
                "original component reused under different group ID",
            )
            component_group[group["group_id"]] = group["id"]
            semantics = {
                k: v
                for k, v in group.items()
                if k not in ("protection", "response", "trace_manifest")
            }
            if group["id"] in by_group:
                require(
                    by_group[group["id"]] == semantics and group_window_ids[group["id"]] == ids,
                    "duplicate group changes role/content/counters",
                )
            else:
                by_group[group["id"]] = semantics
                group_window_ids[group["id"]] = ids
                result["groups"].append(copy.deepcopy(group))
            result["group_provenance"].setdefault(group["id"], []).append(
                {"member": number, "group": copy.deepcopy(group)}
            )
        for entry in member["windows"]:
            window = copy.deepcopy(entry)
            window["prefix_raw_rows"] = [maps["raw_rows"][i] for i in window["prefix_raw_rows"]]
            window["seed_raw_row"] = maps["raw_rows"][window["seed_raw_row"]]
            if window["id"] in by_window:
                require(
                    by_window[window["id"]] == window,
                    "duplicate window differs after semantic index remapping",
                )
            else:
                by_window[window["id"]] = window
                result["windows"].append(window)
        for token, original in member["embedding_rows"].items():
            row = copy.deepcopy(original)
            row["file"] = maps["embedding_files"][row["file"]]
            if token in result["embedding_rows"]:
                require(
                    result["embedding_rows"][token]["sha256"] == row["sha256"],
                    "same token has different native F32 embeddings",
                )
            else:
                result["embedding_rows"][token] = row
            result["embedding_provenance"].setdefault(token, []).append(
                {"member": number, "row": row, "original_row": copy.deepcopy(original)}
            )
        for entry in member["crossed_fc_source_rows"]:
            crossed = copy.deepcopy(entry)
            crossed["raw_row"] = maps["raw_rows"][crossed["raw_row"]]
            result["crossed_fc_source_rows"].append({"member": number, "record": crossed})
        required.update(member["required_embedding_ids"])
        core = {
            "raw_files",
            "raw_rows",
            "windows",
            "groups",
            "embedding_files",
            "embedding_rows",
            "crossed_fc_source_rows",
        }
        result["members"].append(
            {
                "manifest": ref,
                "metadata": {k: copy.deepcopy(v) for k, v in member.items() if k not in core},
                "index_maps": maps,
                "window_ids": [w["id"] for w in member["windows"]],
            }
        )
        for source in member["source_pins"] + [ref] + commands:
            intern("source_pins", source)
    result.update(
        runtime_ancestry=runtime,
        required_embedding_ids=sorted(required),
        missing_embedding_ids=sorted(required - {int(t) for t in result["embedding_rows"]}),
        rollout_contract=first["rollout_contract"],
        source_position_contract=first["source_position_contract"],
    )
    totals = Counter()
    for window in result["windows"]:
        totals[window["role"]] += window["native_prefix"]
    result["native_protections"] = dict(totals)
    result["counts"] = {
        key: len(result[key])
        for key in (
            "raw_files",
            "raw_rows",
            "windows",
            "groups",
            "embedding_files",
            "embedding_rows",
        )
    }
    result["union_source"] = pin(__file__)
    result["materialization_source"] = pin(Path(__file__).with_name("modern_actual_data.py"))
    result["storage_audit"] = {
        "hash_bytes": audit.hash_bytes,
        "row_bytes": audit.row_bytes,
        "metadata_bytes": audit.metadata_bytes,
        "dense_arrays_copied": False,
        "model_weights_opened": False,
        "native_admission_claimed": False,
    }
    result["scope"] = (
        "Multiple admitted original-parent captures, source-index union only; "
        "existing training/reference contracts must not be reused"
    )
    for path, (stamp, _) in audit.checked.items():
        s = Path(path).stat()
        require(
            (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == stamp,
            "source changed before union publication",
        )
    require(len(encoded(result)) + 1 <= MAX_OUTPUT, "union metadata exceeds output budget")
    return result


class UnionActualDataset(ActualDataset):
    """Explicit new-schema reader; unchanged V1 source materialization semantics."""

    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        require(self.manifest_path.stat().st_size <= MAX_OUTPUT, "union manifest too large")
        self.manifest = json.loads(self.manifest_path.read_bytes())
        m = self.manifest
        require(
            m.get("union_source") == pin(__file__)
            and m.get("materialization_source")
            == pin(Path(__file__).with_name("modern_actual_data.py")),
            "union/materialization source changed",
        )
        require(
            m.get("schema") == SCHEMA
            and m.get("status") == STATUS
            and m["target"]["sha256"] == TARGET_SHA256
            and m["parent"]["sha256"] == PARENT_SHA256
            and len(m.get("members", [])) >= 2,
            "explicit original-parent union required",
        )
        self.windows = m["windows"]
        require(
            len({w["id"] for w in self.windows}) == len(self.windows), "duplicate union windows"
        )
        self._window_ids = {w["id"]: w for w in self.windows}
        self._rows = _Rows()


def prepare_union(manifests, output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(ARTIFACTS.resolve())
        and output != ARTIFACTS.resolve()
        and not output.exists(),
        "unique C: artifact output required",
    )

    def guard():
        for path in (PROJECT, Path("/mnt/c")):
            s = os.statvfs(path)
            reserve = MAX_OUTPUT if path == Path("/mnt/c") else 0
            require(
                s.f_bavail * s.f_frsize >= FLOOR + reserve, "union storage floor/reserve reached"
            )

    guard()
    start = time.perf_counter()
    result = build_union(manifests, audit=Audit(guard))
    guard()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(encoded(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "status": STATUS,
        "manifest": pin(output / "manifest.json", limit=MAX_OUTPUT),
        "counts": result["counts"],
        "native_protections": result["native_protections"],
        "wall_seconds": time.perf_counter() - start,
        "reader": "modern_actual_union.UnionActualDataset",
        "training_or_native_admission": False,
    }
    with (output / "result.json").open("xb") as stream:
        stream.write(encoded(receipt) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_union(args.manifest, args.output)), flush=True)


if __name__ == "__main__":
    main()

"""Source-indexed actual-serving round views; no model loading or feature recapture.

Raw features are selected by the observed native FC producer, including crossed
bridge sources. A cache position is never assumed to equal a target-feature row.
Consumers rebuild the current actor cache, encode the seed raw feature, then carry
its own pre-norm state for at most max_depth steps and stop at first gold error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from modern_serving_trace import audit_trace

SCHEMA = "w1a8_actual_source_indexed_round_views_v1"
TARGET_SHA256 = "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"
PARENT_SHA256 = "6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048"
WIDTH, HIDDEN = 7680, 2560


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    require(path.suffix not in (".gguf", ".safetensors", ".npz"), "weight access forbidden")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def row_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def select_source(sources, key):
    """Collapse identical raw-row aliases; reject different possible producers."""
    choices = sources.get(key, [])
    require(choices, "decoder state has no preceding native FC/pre-norm producer")
    signatures = {(x["kind"], x["semantic"]) for x in choices}
    require(len(signatures) == 1, "ambiguous decoder FC/pre-norm source")
    return choices[-1]


def truncate_cache(cache, position):
    require(
        type(position) is int and 0 <= position <= len(cache),
        "native decoder cache has a position hole",
    )
    return cache[:position]


class MaterializedWindow(NamedTuple):
    prefix_tokens: list[int]
    prefix_rawfeatures: Any
    seed_rawfeature: Any


class _Rows:
    """Small read-only mmap LRU; each source authenticated on first use."""

    def __init__(self, limit=4):
        self.limit = limit
        self.maps = OrderedDict()
        self.checked = {}

    def matrix(self, spec):
        path = str(Path(spec["path"]).resolve())
        identity = (spec["sha256"], spec["bytes"], tuple(spec["shape"]))
        if path not in self.checked:
            actual = pin(path)
            require(
                actual["sha256"] == spec["sha256"] and actual["bytes"] == spec["bytes"],
                "array source changed: " + path,
            )
            self.checked[path] = identity
        require(self.checked[path] == identity, "conflicting array source pin")
        require(spec["bytes"] == int(np.prod(spec["shape"])) * 4, "wrong source F32 geometry")
        if path not in self.maps:
            self.maps[path] = np.memmap(path, dtype="<f4", mode="r", shape=tuple(spec["shape"]))
        result = self.maps.pop(path)
        self.maps[path] = result
        while len(self.maps) > self.limit:
            self.maps.popitem(last=False)
        return result


class ActualDataset:
    """Load manifest metadata only; materialize one bounded round on request."""

    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        require(
            m.get("schema") == SCHEMA
            and m.get("target", {}).get("sha256") == TARGET_SHA256
            and m.get("parent", {}).get("sha256") == PARENT_SHA256,
            "view ancestry/schema mismatch",
        )
        self.windows = m["windows"]
        require(len({w["id"] for w in self.windows}) == len(self.windows), "duplicate view IDs")
        self._window_ids = {w["id"]: w for w in self.windows}
        self._rows = _Rows()

    def materialize(self, window, device="cpu") -> MaterializedWindow:
        import torch

        require(
            window.get("id") in self._window_ids and window == self._window_ids[window["id"]],
            "window is not from this manifest",
        )
        anchor = window["anchor"]
        require(
            type(anchor) is int
            and 0 <= anchor <= 4096
            and len(window["prefix_tokens"]) == len(window["prefix_raw_rows"]) == anchor,
            "prefix/cache geometry mismatch",
        )
        required = window["prefix_raw_rows"] + [window["seed_raw_row"]]
        features = np.empty((anchor + 1, WIDTH), dtype=np.float32)
        for index, row_id in enumerate(required):
            require(
                type(row_id) is int and 0 <= row_id < len(self.manifest["raw_rows"]),
                "unknown raw feature row",
            )
            row = self.manifest["raw_rows"][row_id]
            source = self.manifest["raw_files"][row["file"]]
            values = self._rows.matrix(source)[row["row"]]
            require(
                values.shape == (WIDTH,)
                and np.isfinite(values).all()
                and row_hash(values) == row["raw_sha256"],
                "raw source row shape/hash/finite mismatch",
            )
            features[index] = values
        tensor = torch.from_numpy(features).to(device=device)
        return MaterializedWindow(list(window["prefix_tokens"]), tensor[:-1], tensor[-1])

    def embedding_lookup(self, token, device="cpu"):
        """Exact known native layer0 row; missing tokens fail without a dense fallback."""
        import torch

        entry = self.manifest["embedding_rows"].get(str(token))
        if entry is None:
            raise KeyError(f"native F32 embedding unavailable for token {token}")
        source = self.manifest["embedding_files"][entry["file"]]
        value = self._rows.matrix(source)[entry["row"]]
        require(
            value.shape == (HIDDEN,)
            and np.isfinite(value).all()
            and row_hash(value) == entry["sha256"],
            "native embedding row changed",
        )
        return torch.from_numpy(np.array(value, copy=True)).to(device=device)


def prepare(capture, output, *, embedding_results=()):
    """Audit real responses/producer bytes and freeze shared-payload round views."""
    capture, output = Path(capture).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "manifest.json").exists(), "view manifest already exists")
    checked, sources, arrays = {}, {}, _Rows()

    def checked_ref(ref):
        path = str(Path(ref["path"]).resolve())
        if path not in checked:
            actual = pin(path)
            require(
                actual["sha256"] == ref["sha256"]
                and ("bytes" not in ref or actual["bytes"] == ref["bytes"]),
                "source pin changed: " + path,
            )
            checked[path] = actual
        require(checked[path]["sha256"] == ref["sha256"], "conflicting source ancestry")
        sources[path] = checked[path]
        return Path(path)

    def read(path):
        path = Path(path)
        actual = pin(path)
        sources[actual["path"]] = actual
        return json.loads(path.read_text())

    summary, freeze = read(capture / "summary.json"), read(capture / "freeze.json")
    require(
        summary["status"] == "ACTUAL_PARENT_COLLECTION_COMPLETE_WITH_ACCOUNTED_FAILURES",
        "actual parent collection incomplete",
    )
    require(
        freeze["target"]["sha256"] == TARGET_SHA256 and freeze["parent"]["sha256"] == PARENT_SHA256,
        "wrong actual parent/target ancestry",
    )
    for ref in freeze["source_pins"]:
        checked_ref(ref)
    raw_files, raw_file_index, raw_rows, windows, group_records = [], {}, [], [], []
    all_tokens, cross_sources = set(), []
    totals = Counter()
    role_successes = Counter()
    parser_source = Path(__file__).with_name("modern_serving_trace.py")
    read_source = pin(parser_source)
    sources[read_source["path"]] = read_source

    for group in summary["groups"]:
        require(
            group["status"] == "ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS"
            and group["token_text_stop_parity"] is True,
            "group lacks exact serving reference parity",
        )
        directory = capture / group["id"]
        protection = read(checked_ref(group["protection"]))
        response = read(directory / "response.json")
        check_log = checked_ref(protection["inputs"]["log"])
        checked_ref(protection["inputs"]["response"])
        parsed = json.loads(
            json.dumps(audit_trace(check_log.read_text(), response, expected_depth=2))
        )
        for key in ("rounds", "protections", "protection_mask"):
            require(
                parsed[key] == protection[key], "native emitted protection/response join changed"
            )
        trace_root = directory / "trace"
        files = {
            str(Path(x["path"]).resolve()): x
            for x in protection["native_output_pair_audit"]["trace_files"]
        }
        manifest = read(trace_root / "manifest.json")
        require(
            manifest["schema"] == "w1a8_actual_native_actor_trace_v2"
            and not (trace_root / "failure.json").exists(),
            "unqualified actual trace",
        )
        trace_log = trace_root / "calls.jsonl"
        if str(trace_log) in files:
            checked_ref(files[str(trace_log)])
        events = [json.loads(x) for x in trace_log.read_text().splitlines()]
        sources[str(trace_log)] = pin(trace_log)
        require(
            [x["event"] for x in events] == list(range(1, len(events) + 1)), "trace event holes"
        )
        before = {x["call"]: x for x in events if x["phase"] == "before"}
        after = {x["call"]: x for x in events if x["phase"] == "after"}
        require(
            set(before) == set(after)
            and len(events) == 2 * len(before)
            and all(x["result"] == 0 for x in after.values()),
            "incomplete/failed native calls",
        )

        def array(spec):
            path = str((trace_root / spec["file"]).resolve())
            require(
                Path(path).parent == trace_root.resolve() and path in files,
                "trace array missing authenticated source pin",
            )
            descriptor = {**files[path], "shape": spec["shape"]}
            return arrays.matrix(descriptor), descriptor

        producers, target_batches, enc_cursors, cache = {}, {}, {}, []
        fc_at_target = {}
        round_index, active = 0, None
        group_window_start = len(windows)
        prompt_tokens = None
        target_history = []
        native_rows, unreachable = 0, 0
        for event in events:
            call = event["call"]
            if event["phase"] == "after":
                if event["architecture"] == "eagle3" and "prenorm" in event:
                    values, _ = array(event["prenorm"])
                    for selected in event["selected"]:
                        key = row_hash(values[selected["saved_row"]])
                        producers.setdefault(key, []).append(
                            {
                                "kind": "prenorm",
                                "semantic": (call, selected["batch_row"]),
                                "call": call,
                                "row": selected["batch_row"],
                            }
                        )
                continue
            batch, rows = event["batch"], event["batch"]["rows"]
            if event["architecture"].startswith("qwen35"):
                first = rows[0]["positions"][0]
                require(first <= len(target_history), "target prefix history hole")
                target_history = target_history[:first]
                for row in rows:
                    require(
                        row["seq_ids"] == [0] and row["positions"][0] == len(target_history),
                        "target requires one contiguous sequence",
                    )
                    target_history.append(row["token"])
                target_batches[call] = rows
                if len(target_history) >= protection["prompt_tokens"] and prompt_tokens is None:
                    prompt_tokens = target_history[: protection["prompt_tokens"]]
                continue
            if event["architecture"] != "eagle3":
                continue
            if event["process_type_name"] == "encode":
                require(batch["n_embd"] == WIDTH, "ENC raw feature width mismatch")
                raw, descriptor = array(batch["embeddings"])
                values, fc_descriptor = array(after[call]["encoded_fc"])
                require(
                    raw.shape == (len(rows), WIDTH) and values.shape == (len(rows), HIDDEN),
                    "ENC raw/FC rows mismatch",
                )
                upstream = event["active_target_call"] or event["last_completed_target_call"]
                require(upstream in target_batches, "ENC missing upstream target batch")
                offset = enc_cursors.get(upstream, 0)
                target = target_batches[upstream][offset : offset + len(rows)]
                require(len(target) == len(rows), "ENC rows exceed upstream target batch")
                file_key = descriptor["path"]
                if file_key not in raw_file_index:
                    raw_file_index[file_key] = len(raw_files)
                    raw_files.append(descriptor)
                for i, row in enumerate(rows):
                    require(row["saved_embedding_row"] == i, "ENC raw row index mismatch")
                    raw_digest, fc_digest = row_hash(raw[i]), row_hash(values[i])
                    row_id = len(raw_rows)
                    raw_rows.append(
                        {
                            "file": raw_file_index[file_key],
                            "row": i,
                            "group": group["id"],
                            "encoder_call": call,
                            "target_call": upstream,
                            "target_position": target[i]["positions"][0],
                            "raw_sha256": raw_digest,
                            "native_fc_sha256": fc_digest,
                            "native_fc_file": fc_descriptor,
                            "native_fc_row": i,
                        }
                    )
                    producer = {
                        "kind": "fc",
                        "semantic": raw_digest,
                        "raw_row": row_id,
                        "call": call,
                        "row": i,
                        "target_position": target[i]["positions"][0],
                    }
                    producers.setdefault(fc_digest, []).append(producer)
                    fc_at_target[target[i]["positions"][0]] = (fc_digest, producer)
                enc_cursors[upstream] = offset + len(rows)
                continue
            if "embeddings" not in batch:
                totals["warmup_decode_calls_skipped"] += 1
                continue
            require(
                event["process_type_name"] == "decode" and batch["n_embd"] == HIDDEN,
                "unknown actual decoder input geometry",
            )
            supplied, input_descriptor = array(batch["embeddings"])
            require(supplied.shape == (len(rows), HIDDEN), "decoder state rows mismatch")
            first = rows[0]["positions"][0]
            require(first <= len(cache), "native decoder cache has a position hole")
            if first < len(cache):
                totals["cache_overwrite_calls"] += 1
                cache = truncate_cache(cache, first)
            selected = {x["batch_row"]: x for x in after[call].get("selected", [])}
            require(
                set(selected) == {i for i, row in enumerate(rows) if row["output"]},
                "selected output rows differ",
            )
            for i, row in enumerate(rows):
                position, token = row["positions"][0], row["token"]
                require(
                    row["seq_ids"] == [0]
                    and row["has_embd"]
                    and not row["has_state"]
                    and position == len(cache)
                    and row["saved_embedding_row"] == i,
                    "decoder cache/row geometry mismatch",
                )
                all_tokens.add(token)
                key = row_hash(supplied[i])
                source = select_source(producers, key)
                if source["kind"] == "fc" and source["target_position"] != position:
                    prior = fc_at_target.get(position)
                    cross_sources.append(
                        {
                            "group": group["id"],
                            "decoder_call": call,
                            "decoder_row": i,
                            "decoder_position": position,
                            "input_token": token,
                            "input_file": input_descriptor,
                            "input_row_sha256": key,
                            "encoder_call": source["call"],
                            "encoder_row": source["row"],
                            "source_target_position": source["target_position"],
                            "raw_row": source["raw_row"],
                            "native_fc_row_sha256": raw_rows[source["raw_row"]]["native_fc_sha256"],
                            "prior_same_position_native_fc_sha256": None
                            if prior is None
                            else prior[0],
                        }
                    )
                candidate = (
                    protection["rounds"][round_index]
                    if round_index < len(protection["rounds"])
                    else None
                )
                is_seed = (
                    row["output"]
                    and candidate is not None
                    and source["kind"] == "fc"
                    and position == candidate["anchor_position"]
                    and token == candidate["boundary_token"]
                )
                if is_seed:
                    require(prompt_tokens is not None, "round seed before complete prompt")
                    require(
                        all(x["source"]["kind"] == "fc" for x in cache),
                        "round prefix contains recurrent pre-norm without a raw FC source",
                    )
                    expected = (
                        prompt_tokens + response["tokens"][: candidate["parent_response_index"]]
                    )[1 : position + 1]
                    require(
                        [x["token"] for x in cache] == expected,
                        "native cache tokens differ from verified emitted prefix",
                    )
                    start = candidate["first_proposal_response_index"]
                    depth = min(2, candidate["proposed"], len(response["tokens"]) - start)
                    require(depth > 0, "served round lacks an emitted next-token label")
                    protections = [
                        p for p in protection["protections"] if p["round"] == candidate["round"]
                    ]
                    require(
                        [p["proposal_depth"] for p in protections] == list(range(len(protections)))
                        and len(protections) <= depth,
                        "native emitted protections do not form bounded prefix",
                    )
                    active = {
                        "id": f"{group['id']}:round-{candidate['round']:04d}",
                        "role": group["role"],
                        "domain": group["domain"],
                        "group_id": group["group_id"],
                        "group": group["id"],
                        "round": candidate["round"],
                        "anchor": position,
                        "seed_token": token,
                        "labels": response["tokens"][start : start + depth],
                        "max_depth": depth,
                        "response_label_start": start,
                        "native_prefix": len(protections),
                        "native_verified_prefix": candidate["verified_accepted"],
                        "native_proposals": candidate["proposals"],
                        "native_protections": protections,
                        "native_success_mask": [d < len(protections) for d in range(depth)],
                        "native_supervision_reachable_mask": [
                            d == 0
                            or all(
                                candidate["proposals"][j] == response["tokens"][start + j]
                                for j in range(d)
                            )
                            for d in range(depth)
                        ],
                        "prefix_tokens": [x["token"] for x in cache],
                        "prefix_raw_rows": [x["source"]["raw_row"] for x in cache],
                        "seed_raw_row": source["raw_row"],
                        "seed_call": call,
                        "native_selected": [],
                    }
                    windows.append(active)
                    role_successes[group["role"]] += len(protections)
                    round_index += 1
                if row["output"]:
                    require(active is not None, "selected actual draft output lacks a served round")
                    n = selected[i]
                    require(
                        n["logits_mode"] == "backend_explicit_candidate_pairs",
                        "explicit native candidate pairs required",
                    )
                    vals, _ = array(n["candidate_logits"])
                    ids = n["candidate_ids"]
                    require(
                        len(ids) == vals.size == n["candidate_count"]
                        and ids[int(vals.argmax())] == n["argmax_target_token"],
                        "native candidate logit/ID join differs",
                    )
                    d = len(active["native_selected"])
                    if d < len(active["native_proposals"]):
                        require(
                            n["argmax_target_token"] == active["native_proposals"][d]
                            and position == active["anchor"] + d,
                            "native proposal/round/depth join differs",
                        )
                    active["native_selected"].append(
                        {
                            "call": call,
                            "batch_row": i,
                            "depth": d,
                            "decoder_position": position,
                            "input_token": token,
                            "proposed": n["argmax_target_token"],
                            "source_kind": source["kind"],
                            "source_call": source["call"],
                            "source_row": source["row"],
                            "native_candidate_ids": ids,
                            "native_candidate_logits": n["candidate_logits"],
                        }
                    )
                    native_rows += 1
                    if d >= active["max_depth"] or (
                        d
                        and not active["native_supervision_reachable_mask"][
                            min(d, active["max_depth"] - 1)
                        ]
                    ):
                        unreachable += 1
                cache.append({"token": token, "source": source})
        require(round_index == len(protection["rounds"]), "missing actual served round seed")
        group_windows = windows[group_window_start:]
        require(
            sum(len(w["native_protections"]) for w in group_windows)
            == protection["emitted_draft_successes"],
            "an actual native success disappeared from round views",
        )
        all_tokens.update(response["tokens"])
        group_records.append(
            {
                "id": group["id"],
                "role": group["role"],
                "domain": group["domain"],
                "group_id": group["group_id"],
                "windows": len(group_windows),
                "native_protections": protection["emitted_draft_successes"],
                "native_draft_denominator": protection["draft_proposed"],
                "selected_native_rows": native_rows,
                "unreachable_or_extra_selected_rows": unreachable,
                "protection": group["protection"],
                "response": pin(directory / "response.json"),
                "trace_manifest": pin(trace_root / "manifest.json"),
            }
        )

    # Layer0 is a token lookup only. No conditional hidden features enter these views.
    embedding_files, embedding_rows = [], {}
    conditional = Path(freeze["conditional"])
    conditional_summary, conditional_freeze = (
        read(conditional / "summary.json"),
        read(conditional / "freeze.json"),
    )
    require(
        conditional_freeze["target"]["sha256"] == TARGET_SHA256, "known embedding target differs"
    )

    def add_embedding(capture_ref, array_ref):
        metadata = read(checked_ref(capture_ref))
        checked_ref(array_ref)
        require(
            metadata["model"] == freeze["target"]["path"]
            and 0 in metadata["taps"]
            and metadata["dtype"] == "little_endian_float32",
            "embedding source is not exact native target layer0",
        )
        descriptor = {**array_ref, "shape": metadata["feature_shape"]}
        require(
            descriptor["shape"] == [len(metadata["input_tokens"]), HIDDEN],
            "embedding source shape differs",
        )
        values = arrays.matrix(descriptor)
        file_id = len(embedding_files)
        embedding_files.append(descriptor)
        for i, token in enumerate(metadata["input_tokens"]):
            if token not in all_tokens:
                continue
            require(np.isfinite(values[i]).all(), "nonfinite native embedding row")
            item = {"file": file_id, "row": i, "sha256": row_hash(values[i])}
            if str(token) in embedding_rows:
                require(
                    embedding_rows[str(token)]["sha256"] == item["sha256"],
                    "same native token has different layer0 rows",
                )
            else:
                embedding_rows[str(token)] = item

    for group in conditional_summary["groups"]:
        collection = group.get("native_collection", {})
        if "admission" not in collection:
            continue
        admission = read(checked_ref(collection["admission"]))
        evidence = admission["evidence"]
        require(evidence["target"]["sha256"] == TARGET_SHA256, "known embedding ancestry differs")
        add_embedding(evidence["capture"], evidence["arrays"]["layer-0.f32"])
    for root in embedding_results:
        root = Path(root)
        result, captured = read(root / "result.json"), read(root / "freeze.json")
        require(
            captured["target"]["sha256"] == TARGET_SHA256
            and result["status"] == "NATIVE_EMBEDDING_ROWS_CAPTURED",
            "extra native embeddings unqualified",
        )
        add_embedding(result["capture"], result["layer0"])
    require(
        role_successes
        == Counter(
            {
                role: sum(
                    g["emitted_draft_successes"] for g in summary["groups"] if g["role"] == role
                )
                for role in {g["role"] for g in summary["groups"]}
            }
        ),
        "all frozen actual parent protections must remain",
    )
    missing = sorted(all_tokens - {int(x) for x in embedding_rows})
    manifest = {
        "schema": SCHEMA,
        "target": freeze["target"],
        "parent": freeze["parent"],
        "capture": pin(capture / "summary.json"),
        "source_pins": list(sources.values()) + [pin(__file__)],
        "raw_files": raw_files,
        "raw_rows": raw_rows,
        "windows": windows,
        "groups": group_records,
        "embedding_files": embedding_files,
        "embedding_rows": embedding_rows,
        "required_embedding_ids": sorted(all_tokens),
        "missing_embedding_ids": missing,
        "native_protections": dict(role_successes),
        "crossed_fc_source_rows": cross_sources,
        "counts": {
            **dict(totals),
            "groups": len(group_records),
            "windows": len(windows),
            "raw_files": len(raw_files),
            "raw_rows": len(raw_rows),
            "required_embedding_ids": len(all_tokens),
            "known_embedding_ids": len(embedding_rows),
            "missing_embedding_ids": len(missing),
        },
        "rollout_contract": (
            "Rebuild currentactor K/V from prefix raw rows in observed cache order; "
            "encode seed raw row; carry own pre-norm; stop immediately at first gold divergence; "
            "native protection references never removed"
        ),
        "source_position_contract": (
            "Raw target source position may differ from decoder cache position; "
            "observed source lineage preserved without hidden repair"
        ),
        "scope": (
            "Frozen source-indexed actual parent histories, not trained generalization, "
            "actor parity or a repaired-runtime dataset"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), allow_nan=False) + "\n"
    )
    (output / "all-used-token.ids").write_text("\n".join(map(str, sorted(all_tokens))) + "\n")
    (output / "missing-embedding.ids").write_text(
        "\n".join(map(str, missing)) + ("\n" if missing else "")
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding-result", action="append", type=Path, default=[])
    args = parser.parse_args()
    result = prepare(args.capture, args.out, embedding_results=args.embedding_result)
    print(
        json.dumps(
            {
                "manifest": str(args.out / "manifest.json"),
                "counts": result["counts"],
                "native_protections": result["native_protections"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

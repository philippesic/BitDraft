#!/usr/bin/env python3
"""Bounded CPU-only replay of retained native A1 head operands; no eligibility."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"source hash differs: {path}")
    return {"path": str(path), "sha256": actual, "expected_sha256": expected,
            "pinned_hash_verified": expected is not None}


def replay(packed, scales, state, chunk_rows=128):
    state = np.asarray(state, dtype=np.float32)
    k = state.size
    if state.ndim != 1 or not np.isfinite(state).all() or not np.isfinite(scales).all():
        raise ValueError("head inputs must be finite")
    if packed.shape != (len(scales), (k + 31) // 32):
        raise ValueError("packed shape differs from logical width")
    estimated = packed.nbytes + chunk_rows * k * 16 + len(scales) * 16 + 8 * 1024**2
    if estimated > 256 * 1024**2:
        raise ValueError("bounded replay working-set estimate exceeds 256MiB")
    # Raw sign bits preserve negative subnormals; both signed zeros are positive.
    raw = state.view(np.uint32)
    positive = ((raw & 0x80000000) == 0) | ((raw & 0x7fffffff) == 0)
    act_bytes = np.packbits(positive, bitorder="little")
    padded = np.zeros(packed.shape[1] * 4, dtype=np.uint8)
    padded[:len(act_bytes)] = act_bytes
    act_scale = np.float32(np.mean(np.abs(state).astype(np.float64)))
    activation = np.where(positive, np.float32(1), np.float32(-1)) * act_scale
    native, scaled = np.empty(len(scales), np.float32), np.empty(len(scales), np.float32)
    for start in range(0, len(scales), chunk_rows):
        stop = min(start + chunk_rows, len(scales))
        words = np.ascontiguousarray(packed[start:stop], dtype="<u4")
        raw_bytes = words.view(np.uint8).reshape(len(words), -1)
        xor = np.bitwise_xor(raw_bytes, padded)
        if k % 8:
            xor[:, k // 8] &= (1 << (k % 8)) - 1
        xor[:, (k + 7) // 8:] = 0
        mismatch = POPCOUNT[xor].sum(axis=1, dtype=np.int64)
        dots = (k - 2 * mismatch).astype(np.float32)
        native[start:stop] = (dots * scales[start:stop]).astype(np.float32) * act_scale
        signs = np.unpackbits(raw_bytes, axis=1, bitorder="little")[:, :k].astype(np.float32)
        signs *= 2
        signs -= 1
        scaled[start:stop] = (signs @ activation).astype(np.float32) * scales[start:stop]
    return native, scaled, float(act_scale), estimated


def summarize(logits, mapping, native_row):
    best = float(np.max(logits))
    # Native iterates mapped target IDs in increasing order, using strict >.
    argmax = int(np.min(mapping[logits == best]))
    top = np.partition(logits, -2)[-2:]
    reverse = {int(token): i for i, token in enumerate(mapping)}
    probes = []
    for probe in native_row["draft_logit_probe"]:
        token = int(probe["token_id"])
        if token not in reverse:
            raise ValueError("finite native probe is outside mapped draft vocabulary")
        value = float(logits[reverse[token]])
        probes.append({"token_id": token, "native_logit": probe["logit"],
                       "replayed_logit": value, "abs_diff": abs(value - probe["logit"])})
    if len(probes) != 8:
        raise ValueError("expected all eight retained native logit probes")
    native_id = int(native_row["draft_argmax_id"])
    if native_id not in reverse:
        raise ValueError("native argmax is outside mapping")
    return {"mapped_argmax_id": argmax, "argmax_logit": best,
            "argmax_id_agrees": argmax == native_id,
            "native_argmax_logit_replayed": float(logits[reverse[native_id]]),
            "native_argmax_logit_abs_diff": abs(float(logits[reverse[native_id]]) - native_row["draft_argmax_logit"]),
            "top_two_margin": float(top.max() - top.min()), "probes": probes,
            "max_probe_abs_diff": max(p["abs_diff"] for p in probes)}


def load_gguf_reader():
    """One-off reader-only package bootstrap; skip optional vocab/tokenizers."""
    if "gguf" not in sys.modules:
        spec = importlib.util.find_spec("gguf")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("installed gguf package is required")
        package = types.ModuleType("gguf")
        package.__path__ = list(spec.submodule_search_locations)
        package.__spec__ = spec
        package.__package__ = "gguf"
        sys.modules["gguf"] = package
    return importlib.import_module("gguf.gguf_reader").GGUFReader


def run(candidate_path, stages_path):
    reader_class = load_gguf_reader()
    started = time.time()
    inputs = {"candidate": record(candidate_path), "stages_config": record(stages_path)}
    candidate = json.loads(Path(candidate_path).read_text())
    stages = json.loads(Path(stages_path).read_text())
    if candidate["activation_bits"] != 1 or len(candidate["roots"]) != 6:
        raise ValueError("diagnostic requires the retained six-root A1 candidate")
    evidence = candidate["evidence"]
    for name in ("native_cell", "export", "native_capture_manifest"):
        inputs[name] = record(evidence[name]["path"], evidence[name]["sha256"])
    cell_path = Path(evidence["native_cell"]["path"])
    cell = json.loads(cell_path.read_text())
    root = cell_path.parent
    files = cell.get("files", {})
    for name in ("heads.jsonl", "heads.f32"):
        inputs[name] = record(root / name, files.get(name, {}).get("sha256"))
    tasks_path = root.parent / "task_prompt_ids.json"
    tasks = json.loads(tasks_path.read_text())
    capture = json.loads(Path(evidence["native_capture_manifest"]["path"]).read_text())
    tasks_expected = capture.get("task_prompt_ids_sha256")
    inputs["task_prompt_ids"] = record(tasks_path, tasks_expected)
    if "task_prompt_ids" in cell and tasks != cell["task_prompt_ids"]:
        raise ValueError("native task map disagrees with cell manifest")
    map_hash = candidate["common_source_sha256"]["absolute_d2t"]
    if stages["sources"]["sha256"]["absolute_d2t"] != map_hash:
        raise ValueError("candidate and stage mapping source hashes differ")
    map_path = Path(stages["sources"]["absolute_d2t"])
    inputs["absolute_d2t"] = record(map_path, map_hash)
    mapping = np.asarray(np.load(map_path, allow_pickle=False), dtype=np.int64)
    reader = reader_class(evidence["export"]["path"])
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    packed = np.asarray(tensors["output.w1a1_packed"].data, dtype="<i4").view("<u4")
    scales = np.asarray(tensors["output.w1a1_scale"].data, dtype=np.float32)
    k = int(reader.get_field("eagle3.w1a1.tensor.output_weight.logical_k").contents())
    if not np.array_equal(mapping, np.asarray(tensors["d2t"].data, dtype=np.int64)):
        raise ValueError("GGUF mapped target IDs disagree with pinned absolute d2t")
    if mapping.ndim != 1 or len(mapping) != len(scales) or len(np.unique(mapping)) != len(mapping):
        raise ValueError("draft mapping must be unique and match head rows")
    heads = [json.loads(line) for line in (root / "heads.jsonl").read_text().split("\n") if line.strip()]
    states = np.memmap(root / "heads.f32", mode="r", dtype="<f4")
    if states.size != len(heads) * k:
        raise ValueError("native state payload dimensions differ")
    states = states.reshape(-1, k)
    index = {}
    for row in heads:
        if row["depth"] == 0:
            key = (tasks[str(row["task_id"])], tuple(row["prefix_token_ids"]))
            if key in index:
                raise ValueError("ambiguous exact-prefix native root")
            index[key] = row
    roots = []
    working_set = 0
    for selected in candidate["roots"]:
        row = index[(selected["prompt_id"], tuple(selected["prefix_token_ids"]))]
        if row["state_boundary"] != "native_output_norm_f32_before_head_operand_conversion":
            raise ValueError("native state boundary differs")
        if not 0 <= row["state_row"] < len(states):
            raise ValueError("native state row outside payload")
        exact, scaled, scale, estimated = replay(packed, scales, states[row["state_row"]])
        working_set = max(working_set, estimated)
        roots.append({"domain": selected["domain"], "prompt_id": selected["prompt_id"],
                      "round_index": selected["round_index"], "state_row": row["state_row"],
                      "prefix_token_ids": selected["prefix_token_ids"], "activation_scale": scale,
                      "native_argmax_id": row["draft_argmax_id"],
                      "native_argmax_logit": row["draft_argmax_logit"],
                      "candidate_metrics": selected, "integer_dot": summarize(exact, mapping, row),
                      "scaled_before_dot": summarize(scaled, mapping, row),
                      "formula_max_abs_diff": float(np.max(np.abs(exact - scaled)))})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    framework_roots = ("torch", "transformers", "tensorflow", "jax", "jaxlib", "onnxruntime", "tokenizers")
    loaded_framework_modules = sorted(
        name for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in framework_roots)
    )
    return {"schema": "retained_a1_head_cpu_replay_v1", "training_eligible": False,
            "execution_device": "CPU", "hardware": platform.platform(), "machine": platform.machine(),
            "numpy_version": np.__version__,
            "gguf_reader_modules": {name: record(sys.modules[name].__file__) for name in
                ("gguf.gguf_reader", "gguf.constants", "gguf.quants", "gguf.lazy")},
            "started_unix_seconds": started, "duration_seconds": time.time() - started,
            "peak_process_rss_bytes": rss_bytes, "working_set_estimate_bytes": working_set,
            "resource_budget_bytes": 512 * 1024**2,
            "resource_budget_passed": rss_bytes <= 512 * 1024**2,
            "loaded_framework_modules": loaded_framework_modules,
            "inputs": inputs, "helper": record(__file__), "roots": roots,
            "limits_of_claim": "retained native head operands only; NumPy scaled-before-dot is not CUDA reduction parity or body replay"}


def save_report(report, output):
    if output.exists():
        raise FileExistsError("preserve previous diagnostic output; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, output)
    print(json.dumps({"output": str(output), "sha256": sha256(output),
                      "roots": len(report["roots"]), "duration_seconds": report["duration_seconds"],
                      "peak_process_rss_bytes": report["peak_process_rss_bytes"],
                      "resource_budget_passed": report["resource_budget_passed"]}))
    return 0 if report["resource_budget_passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stages-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(save_report(run(args.candidate, args.stages_config), args.output))

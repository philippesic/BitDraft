"""Lossless BF16 EAGLE source-faithful export repair; CPU only, originals preserved.

Usage: PROJECT/.venv/bin/python repair_eagle_export.py --project PROJECT --output-root RUN
Only Q/K row order, eagle3.rope.dimension_count and native input tap indices
change. The new file is a byte copy opened through GGUFReader r+, avoiding
metadata reserialization. Earlier partial-RoPE-only outputs remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np


TRAINING_SOURCE = {
    "url": "https://raw.githubusercontent.com/NikitosKh/mini-specforge/fb78afdb59a20d62efac75fb8c11972ea2ba0a2f/specforge_mini/draft_model.py",
    "sha256": "560d63cac781868c2995edd6349f4de853921689952065532503ce0a85b6a75a",
    "evidence": "Cited by both pinned model cards; Attention lines115-116,134,152-159 honor partial_rotary_factor and rotate the leading slice. Exact checkpoint training commit not recorded by publisher.",
}

TAP_SOURCES = [
    {"url": "https://raw.githubusercontent.com/NikitosKh/mini-specforge/986fc4b127afdb18d9893dc7ba4a3a302ba92b5e/specforge_mini/target.py",
     "sha256": "2a2bbbaf25aab7087c4548827c5afc0fff8b3b903142acbb399542d95d9889ea",
     "evidence": "HuggingFaceBackend default indices [1,15,28] for32layers; lines54-56 register_forward_hook captures layer OUTPUTS."},
    {"url": "https://raw.githubusercontent.com/sgl-project/sglang/e8bf40b8eed8605e9e006857cb8562a076fa4bf7/python/sglang/srt/models/qwen3_vl.py",
     "sha256": "b104ff75c794011e45679493abbbf6b00b8cb3cf972184ff0f17daa582ad7e8d",
     "evidence": "Qwen3.5 inherits this wrapper; set_eagle3_layers_to_capture at1777-1795 adds1 to supplied output-layer IDs before input-boundary capture."},
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def header_hash_except(data: np.ndarray, header_end: int, ranges: list[tuple[int, int]]) -> str:
    h = hashlib.sha256()
    cursor = 0
    for left, right in sorted(ranges):
        assert cursor <= left < right <= header_end
        h.update(data[cursor:left])
        cursor = right
    h.update(data[cursor:header_end])
    return h.hexdigest()


def perm_indices(head: int, rotary: int) -> np.ndarray:
    assert 0 < rotary <= head and rotary % 2 == 0
    return np.concatenate((np.arange(rotary).reshape(2, rotary // 2).T.reshape(-1), np.arange(rotary, head)))


def repair_rows(raw: np.ndarray, heads: int, head: int, rotary: int) -> tuple[np.ndarray, np.ndarray]:
    assert raw.shape[0] == heads * head
    grouped = raw.reshape(heads, head, -1)
    full = perm_indices(head, head)
    partial = perm_indices(head, rotary)
    hf_rows = grouped[:, np.argsort(full), :]
    fixed = np.ascontiguousarray(hf_rows[:, partial, :].reshape(raw.shape))
    reconstructed = fixed.reshape(heads, head, -1)[:, np.argsort(partial), :][:, full, :].reshape(raw.shape)
    assert np.array_equal(reconstructed, raw), "actual Q/K bit roundtrip failed"
    return fixed, np.ascontiguousarray(hf_rows.reshape(raw.shape))


def rope_split(x: np.ndarray, rotary: int, position: int) -> np.ndarray:
    result = x.copy()
    theta = position / (1e7 ** (np.arange(0, rotary, 2, dtype=np.float64) / rotary))
    left, right = x[..., :rotary // 2], x[..., rotary // 2:rotary]
    result[..., :rotary // 2] = left * np.cos(theta) - right * np.sin(theta)
    result[..., rotary // 2:rotary] = right * np.cos(theta) + left * np.sin(theta)
    return result


def rope_norm(x: np.ndarray, rotary: int, position: int) -> np.ndarray:
    result = x.copy()
    theta = position / (1e7 ** (np.arange(0, rotary, 2, dtype=np.float64) / rotary))
    pairs = x[..., :rotary].reshape(*x.shape[:-1], rotary // 2, 2)
    result[..., :rotary] = np.stack((pairs[..., 0] * np.cos(theta) - pairs[..., 1] * np.sin(theta),
                                   pairs[..., 1] * np.cos(theta) + pairs[..., 0] * np.sin(theta)), axis=-1).reshape(*x.shape[:-1], rotary)
    return result


def synthetic_check(head: int, rotary: int) -> dict:
    q = np.linspace(-2, 3, head, dtype=np.float64)
    k = np.cos(np.arange(head, dtype=np.float64) / 7)
    partial, full = perm_indices(head, rotary), perm_indices(head, head)
    expected_q, expected_k = rope_split(q, rotary, 5), rope_split(k, rotary, 37)
    fixed_q, fixed_k = rope_norm(q[partial], rotary, 5), rope_norm(k[partial], rotary, 37)
    error = max(float(np.max(np.abs(fixed_q - expected_q[partial]))), float(np.max(np.abs(fixed_k - expected_k[partial]))))
    attention_error = abs(float(np.dot(fixed_q, fixed_k) - np.dot(expected_q, expected_k)))
    wrong_q, wrong_k = rope_norm(q[full], rotary, 5), rope_norm(k[full], rotary, 37)
    negative_error = abs(float(np.dot(wrong_q, wrong_k) - np.dot(expected_q, expected_k)))
    assert error < 1e-12 and attention_error < 1e-10
    assert negative_error > 1e-3, "known-wrong metadata-only control was not rejected"
    return {"split_half_to_interleaved_max_abs_error": error, "attention_dot_abs_error": attention_error,
            "metadata_only_negative_attention_dot_abs_error": negative_error, "invalid_control_rejected": True}


def decode_bf16(raw: np.ndarray, input_width: int) -> np.ndarray:
    bits = np.ascontiguousarray(raw).view("<u2").reshape(-1, input_width).astype(np.uint32)
    return (bits << np.uint32(16)).view(np.float32).astype(np.float64)


def actual_projection_check(hf_raw: np.ndarray, fixed_raw: np.ndarray, heads: int, head: int, rotary: int, input_width: int) -> dict:
    # Deterministic complete-width inputs and ALL actual Q/K rows, not a sampled
    # implementation-mirroring row check. No target features or CUDA involved.
    idx = np.arange(input_width, dtype=np.float64)
    inputs = np.stack((np.sin(idx * .031) / np.sqrt(input_width), np.cos(idx * .017) / np.sqrt(input_width)))
    hf = inputs @ decode_bf16(hf_raw, input_width).T
    fixed = inputs @ decode_bf16(fixed_raw, input_width).T
    hf, fixed = hf.reshape(2, heads, head), fixed.reshape(2, heads, head)
    partial = perm_indices(head, rotary)
    errors = []
    for position in (0, 1, 37):
        expected = rope_split(hf, rotary, position)[..., partial]
        actual = rope_norm(fixed, rotary, position)
        errors.append(float(np.max(np.abs(actual - expected))))
    assert max(errors) < 1e-10, "actual BF16 Q/K projection plus RoPE reference failed"
    return {"inputs": 2, "positions": [0, 1, 37], "all_projection_rows_checked": int(heads * head),
            "input_width": input_width, "float64_reference_max_abs_errors": errors,
            "scope": "same complete-width deterministic inputs, real BF16 weights, CPU Q/K linear plus RoPE only"}


def repair_model(project: Path, output_root: Path, size: str, gguf) -> dict:
    start = time.perf_counter()
    old_root = project / "runs/spec-shortlist-20261030"
    source = old_root / "models" / f"Qwen3.5-{size}-eagle3-BF16.gguf"
    config_path = old_root / "models" / f"hf-eagle-{size}" / "config.source.json"
    config = json.loads(config_path.read_text())
    head = int(config["head_dim"])
    factor = float(config["partial_rotary_factor"])
    assert factor == float(config["rope_parameters"]["partial_rotary_factor"]) == .25
    rotary = int(head * factor)
    assert (head, rotary) == ({"4B": (160, 40), "9B": (256, 64)}[size])
    assert config["rope_parameters"]["rope_theta"] == 10000000
    manifest = json.loads((old_root / "environment-model-manifest.json").read_text())
    expected_sha = next(r["sha256"] for r in manifest["models"] if Path(r["path"]).name == source.name)
    source_sha = sha(source)
    assert source_sha == expected_sha
    output = output_root / "models" / f"Qwen3.5-{size}-eagle3-sourcefaithful-BF16.gguf"
    incomplete = output.with_suffix(".gguf.incomplete")
    assert not output.exists() and not incomplete.exists(), "never overwrite an existing artifact"
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, incomplete.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    reader = gguf.GGUFReader(source)
    assert reader.fields["general.architecture"].contents() == "eagle3"
    assert reader.fields["eagle3.rope.dimension_count"].contents() == head
    training_taps = config["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
    native_taps = [idx + 1 for idx in training_taps]
    assert reader.fields["eagle3.target_layers"].contents() == training_taps == [1, 15, 28]
    assert native_taps == [2, 16, 29]
    writable = gguf.GGUFReader(incomplete, mode="r+")
    assert writable.data_offset == reader.data_offset and writable.data.nbytes == reader.data.nbytes
    old_tensors = {t.name: t for t in reader.tensors}
    new_tensors = {t.name: t for t in writable.tensors}
    assert old_tensors.keys() == new_tensors.keys()
    changed = {}
    for name, count_key in (("blk.0.attn_q.weight", "eagle3.attention.head_count"), ("blk.0.attn_k.weight", "eagle3.attention.head_count_kv")):
        old, new = old_tensors[name], new_tensors[name]
        assert old.tensor_type == gguf.GGMLQuantizationType.BF16
        assert list(old.shape) == list(new.shape) and old.data_offset == new.data_offset
        heads = int(reader.fields[count_key].contents())
        fixed, hf_raw = repair_rows(old.data, heads, head, rotary)
        changed[name] = {"input_sha256": array_sha(old.data), "output_sha256": array_sha(fixed), "bit_roundtrip": True,
                         "projection_rope_check": actual_projection_check(hf_raw, fixed, heads, head, rotary, int(old.shape[0]))}
        new.data[...] = fixed
        assert array_sha(new.data) == changed[name]["output_sha256"]
        del fixed, hf_raw
    changed_header_ranges = []
    metadata_changes = {}
    for key, old_values, new_values in (("eagle3.rope.dimension_count", [head], [rotary]),
                                       ("eagle3.target_layers", training_taps, native_taps)):
        field = writable.fields[key]
        assert len(field.data) == len(new_values)
        offsets = []
        for part_idx, old_value, new_value in zip(field.data, old_values, new_values):
            scalar = field.parts[part_idx]
            offset = int(scalar.ctypes.data - writable.data.ctypes.data)
            assert scalar.size == 1 and scalar.dtype.itemsize == 4 and int(scalar[0]) == old_value
            scalar[...] = new_value
            changed_header_ranges.append((offset, offset + scalar.nbytes))
            offsets.append(offset)
        metadata_changes[key] = {"old": old_values, "new": new_values, "scalar_file_offsets": offsets}
    writable.data.flush()
    # Every header byte, including every other key/value, tensor descriptor and
    # padding byte, must match outside the rotary scalar and three tap scalars.
    header_source = header_hash_except(reader.data, reader.data_offset, changed_header_ranges)
    header_output = header_hash_except(writable.data, writable.data_offset, changed_header_ranges)
    assert header_source == header_output
    untouched = {}
    for name, old in old_tensors.items():
        new = new_tensors[name]
        assert old.tensor_type == new.tensor_type and list(old.shape) == list(new.shape)
        if name not in changed:
            before, after = array_sha(old.data), array_sha(new.data)
            assert before == after, f"untouched tensor changed: {name}"
            untouched[name] = before
    # Also assert exact byte identity of gaps/padding outside actual tensors.
    regions = sorted((t.data_offset, t.data_offset + t.n_bytes) for t in reader.tensors)
    cursor = reader.data_offset
    for left, right in regions:
        assert np.array_equal(reader.data[cursor:left], writable.data[cursor:left])
        cursor = right
    assert np.array_equal(reader.data[cursor:], writable.data[cursor:])
    assert int(writable.fields["eagle3.rope.dimension_count"].contents()) == rotary
    assert writable.fields["eagle3.target_layers"].contents() == native_taps
    assert sha(source) == source_sha, "source changed during repair"
    incomplete.rename(output)
    return {"size": size, "source": str(source), "source_sha256": source_sha,
            "output": str(output), "output_sha256": sha(output), "output_bytes": output.stat().st_size,
            "config_source": str(config_path), "config_sha256": sha(config_path),
            "head_dim": head, "partial_rotary_factor": factor, "old_rotary_dim": head, "new_rotary_dim": rotary,
            "training_output_layer_ids": training_taps, "native_input_layer_ids": native_taps,
            "metadata_changes": metadata_changes,
            "changed_tensor_checks": changed, "untouched_tensor_sha256": untouched,
            "all_other_header_bytes_sha256": header_source, "all_other_header_bytes_and_padding_preserved": True,
            "synthetic_check": synthetic_check(head, rotary), "cpu_wall_seconds": time.perf_counter() - start,
            "full_native_drafter_fidelity_admitted": False, "full_request_correctness_admitted": False,
            "limitation": "No GPU or full native same-history drafter comparison; this proves artifact transformation and Q/K+RoPE fidelity only."}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    project, output_root = args.project.resolve(), args.output_root.resolve()
    sys.path.insert(0, str(project / "third_party/llama.cpp/gguf-py"))
    import gguf
    receipt_path = output_root / "eagle-sourcefaithful-repair.json"
    assert not receipt_path.exists(), "never overwrite an existing receipt"
    start = time.perf_counter()
    receipt = {"completed": False, "training_seconds": 0, "gpu_seconds": 0,
               "training_source": TRAINING_SOURCE, "tap_sources": TAP_SOURCES,
               "native_input_capture_sources": [{"path": str(project / "runs/spec-shortlist-20261030/runtime" / path),
                                                  "sha256": sha(project / "runs/spec-shortlist-20261030/runtime" / path)}
                                                 for path in ("src/models/qwen35.cpp", "src/llama-context.cpp", "common/speculative.cpp")],
               "repair_script": str(Path(__file__).resolve()),
               "repair_script_sha256": sha(Path(__file__)), "records": []}
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        for size in ("4B", "9B"):
            result = repair_model(project, output_root, size, gguf)
            receipt["records"].append(result)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({k: result[k] for k in ("size", "output", "output_sha256", "cpu_wall_seconds")}), flush=True)
        receipt["completed"] = True
    except BaseException as exc:
        receipt["failure"] = repr(exc)
        raise
    finally:
        receipt["total_cpu_wall_seconds"] = time.perf_counter() - start
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

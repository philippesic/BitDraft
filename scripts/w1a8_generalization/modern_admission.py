"""CPU-only admission of captured Qwen3.5-4B native development rows.

This gate joins an authenticated native capture to an actual target-only server
response. Agreement admits a supervised row and a *numeric* protection gate;
it never proves parent success, actor fidelity, answer quality or generalization.
Raw arrays stay shared. No model, backend or CUDA context is initialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT = Path("/home/philip/binary-eagle-decoding")
TRUSTED_BASELINE = {
    "path": str(PROJECT / "experiments/w1a8-standing-20261102/baseline.json"),
    "sha256": "a79f08524fd7ed9f799429b3b3cb6239c52bd76551244ec9461196c2349a3116",
}
TARGET_ID = "4B-Q4_K_M"
WIDTH, VOCAB = 2560, 248320
ARRAY_NAMES = {"logits.f32", *(f"layer-{tap}.f32" for tap in (0, 2, 16, 29))}
SETTINGS = {
    "context": 4096,
    "batch": 512,
    "ubatch": 512,
    "seq_max": 1,
    "kv": "F16",
    "flash_attention": "on",
    "consistent": 32,
    "sampling": "raw_argmax_lowest_id_tie",
    "load_mtp": False,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def same_pin(left, right):
    return (str(Path(left["path"]).resolve()), left["sha256"]) == (
        str(Path(right["path"]).resolve()),
        right["sha256"],
    )


class Authentication:
    def __init__(self):
        self.files = {}
        self.bytes = 0

    def checked(self, ref):
        require(
            isinstance(ref, dict)
            and {"path", "sha256"} <= ref.keys()
            and set(ref) <= {"path", "sha256", "bytes"},
            "file pin required",
        )
        require(
            isinstance(ref["path"], str)
            and Path(ref["path"]).is_absolute()
            and isinstance(ref["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
            "absolute SHA256 pin required",
        )
        path = Path(ref["path"]).resolve(strict=True)
        before = identity(path)
        if path in self.files:
            saved, stamp = self.files[path]
            require(saved == ref["sha256"] and stamp == before, "conflicting or changed file pin")
            return path
        if "bytes" in ref:
            require(type(ref["bytes"]) is int and ref["bytes"] == before[2], "pinned size differs")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(
            actual == ref["sha256"] and identity(path) == before, "file pin changed: " + str(path)
        )
        self.files[path] = actual, before
        self.bytes += before[2]
        return path

    def read(self, ref):
        path = self.checked(ref)
        require(path.stat().st_size <= 2 * 1024**2, "metadata exceeds 2 MiB")
        return json.loads(path.read_text())

    def unchanged(self):
        for path, (_, stamp) in self.files.items():
            require(identity(path) == stamp, "input changed during admission: " + str(path))


def arguments(argv, value_keys, flags=()):
    require(
        isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv),
        "command argv required",
    )
    values = {}
    i = 1
    while i < len(argv):
        key = argv[i]
        require(key not in values, "duplicate command option")
        if key in flags:
            values[key] = True
            i += 1
        else:
            require(key in value_keys and i + 1 < len(argv), "unknown or incomplete command option")
            values[key] = argv[i + 1]
            i += 2
    return values


def token_ids(path):
    require(path.stat().st_size <= 65536, "token file exceeds 64 KiB")
    words = path.read_text().replace(",", " ").split()
    require(
        0 < len(words) <= 4096 and all(re.fullmatch("[0-9]+", x) for x in words),
        "bounded decimal token IDs required",
    )
    values = [int(x) for x in words]
    require(all(0 <= x < VOCAB for x in values), "token outside target vocabulary")
    return values


def runtime_maps(auth, ref, trusted, *, required):
    text = auth.checked(ref).read_text()
    require(len(text) <= 2 * 1024**2, "library maps exceed 2 MiB")
    paths = set()
    for line in text.splitlines():
        if any(name in line for name in ("libllama", "libggml", "libmtmd")):
            require("(deleted)" not in line, "deleted runtime library mapping")
            path = line.split()[-1]
            require(path in trusted, "unfrozen runtime library mapped: " + path)
            auth.checked(trusted[path])
            paths.add(path)
    require(set(required) <= paths, "required runtime library missing from maps")
    return sorted(paths)


def checked_build(auth, ref, baseline_ref, trusted):
    build = auth.read(ref)
    require(
        build.get("status") == "compiled_linked_cpu_checks_only"
        and same_pin(build["baseline"], baseline_ref),
        "native build/baseline binding differs",
    )
    require(
        build["environment"].get("GGML_CUDA_DECODE_CONSISTENT") == "32",
        "native build correctness mode differs",
    )
    for key in ("source", "builder", "compiler", "executable"):
        auth.checked(build[key])
    for key in ("runtime_sources", "included_local_headers"):
        require(
            isinstance(build.get(key), list) and build[key], "native build dependencies missing"
        )
        for dependency in build[key]:
            auth.checked(dependency)
    require(
        len(build.get("runtime_libraries", [])) == 4, "native runtime library inventory differs"
    )
    for dependency in build["runtime_libraries"] + [build["cuda_overlay"]]:
        require(
            dependency["path"] in trusted and same_pin(dependency, trusted[dependency["path"]]),
            "native runtime differs from frozen baseline",
        )
        auth.checked(dependency)
    return build


def admit_packet(packet):
    """Authenticate one fixed-history packet and return a numeric row receipt."""
    started = time.perf_counter()
    require(
        isinstance(packet, dict)
        and set(packet)
        == {
            "schema",
            "baseline",
            "target",
            "build",
            "capture",
            "capture_command",
            "capture_library_maps",
            "continuation_tokens",
            "arrays",
            "tokenizer",
            "serving",
        },
        "exact admission packet fields required; overrides are unsupported",
    )
    require(packet["schema"] == "w1a8_modern_capture_packet_v1", "unsupported packet schema")
    require(same_pin(packet["baseline"], TRUSTED_BASELINE), "untrusted baseline pin")
    auth = Authentication()
    baseline = auth.read(packet["baseline"])
    rows = [x for x in baseline["rows"] if x["target"] == TARGET_ID]
    require(
        len(rows) == 1 and same_pin(packet["target"], rows[0]["target_pin"]),
        "capture target differs from frozen 4B Q4_K_M target",
    )
    target_path = auth.checked(packet["target"])
    frozen_refs = [x for x in baseline["sources"] if x["path"].endswith("/final-freeze.json")]
    require(len(frozen_refs) == 1, "frozen runtime source missing")
    frozen_ref = dict(frozen_refs[0])
    if not Path(frozen_ref["path"]).is_absolute():
        frozen_ref["path"] = str(PROJECT / frozen_ref["path"])
    frozen = auth.read(frozen_ref)
    trusted = {x["path"]: x for x in frozen["pins"] if ".so" in x["path"]}
    require(same_pin(frozen["cuda"], baseline["cuda"]), "frozen CUDA source differs")
    cuda_path = str(auth.checked(baseline["cuda"]))
    build = checked_build(auth, packet["build"], packet["baseline"], trusted)

    capture = auth.read(packet["capture"])
    require(
        capture.get("status") == "native_capture_complete"
        and capture.get("dtype") == "little_endian_float32"
        and capture.get("feature_semantics") == "zero_based_pre_block_input_all_tokens"
        and capture.get("continuation_mode") == "fixed_token_ids"
        and capture.get("taps") == [0, 2, 16, 29]
        and capture.get("settings") == SETTINGS,
        "capture contract or runtime differs",
    )
    require(
        capture.get("model") == str(target_path)
        and capture.get("model_bytes") == target_path.stat().st_size
        and capture.get("cuda_library") == cuda_path
        and capture.get("gpu_description") == "NVIDIA GeForce RTX 5080",
        "capture target, CUDA or hardware differs",
    )
    expected_cpu = [x["path"] for x in build["runtime_libraries"] if "libggml-cpu.so" in x["path"]]
    require(
        len(expected_cpu) == 1 and capture.get("cpu_library") == expected_cpu[0],
        "capture CPU library differs",
    )
    inputs, prompt_count, positions = (
        capture.get("input_tokens"),
        capture.get("prompt_tokens"),
        capture.get("logit_positions"),
    )
    require(
        isinstance(inputs, list)
        and 1 <= len(inputs) <= 4096
        and all(type(x) is int and 0 <= x < VOCAB for x in inputs)
        and type(prompt_count) is int
        and 1 <= prompt_count < len(inputs),
        "valid prompt and continuation token history required",
    )
    require(
        capture.get("feature_shape") == [len(inputs), WIDTH]
        and isinstance(positions, list)
        and positions
        and all(type(position) is int for position in positions)
        and prompt_count - 1 <= positions[0] <= positions[-1] < len(inputs)
        and positions == list(range(positions[0], positions[-1] + 1))
        and capture.get("logits_shape") == [len(positions), VOCAB],
        "capture shape or next-token positions differ",
    )

    tokenizer = packet["tokenizer"]
    require(
        isinstance(tokenizer, dict)
        and set(tokenizer) == {"build", "metadata", "command", "tokens", "text"},
        "exact native tokenizer ancestry required",
    )
    tokenizer_build = checked_build(auth, tokenizer["build"], packet["baseline"], trusted)
    token_meta = auth.read(tokenizer["metadata"])
    prompt_ids = token_ids(auth.checked(tokenizer["tokens"]))
    text_path = auth.checked(tokenizer["text"])
    raw_text = text_path.read_bytes().decode("utf-8")
    require(text_path.stat().st_size <= 65536, "raw text exceeds 64 KiB")
    require(
        token_meta.get("status") == "native_tokenization_complete"
        and token_meta.get("model") == str(target_path)
        and token_meta.get("model_bytes") == target_path.stat().st_size
        and token_meta.get("text_file") == str(text_path)
        and token_meta.get("text_bytes") == text_path.stat().st_size
        and token_meta.get("token_count") == prompt_count
        and token_meta.get("token_ids_file") == "tokens.ids"
        and all(
            token_meta.get(key) is value
            for key, value in {
                "add_special": True,
                "parse_special": True,
                "chat_template_applied": False,
                "vocab_only": True,
                "backend_initialized": False,
                "weights_loaded": False,
            }.items()
        )
        and prompt_ids == inputs[:prompt_count],
        "native tokenizer/prompt ancestry differs",
    )
    require(
        Path(tokenizer["tokens"]["path"]).parent == Path(tokenizer["metadata"]["path"]).parent
        and Path(tokenizer["tokens"]["path"]).name == "tokens.ids",
        "tokenizer output binding differs",
    )
    token_command = auth.read(tokenizer["command"])
    token_argv = token_command["argv"]
    tokenizer_environment = dict(tokenizer_build["environment"])
    if token_command.get("environment", {}).get("CUDA_VISIBLE_DEVICES") == "":
        tokenizer_environment["CUDA_VISIBLE_DEVICES"] = ""
    require(
        token_argv[0] == tokenizer_build["executable"]["path"]
        and arguments(token_argv, {"--model", "--text-file", "--out"}, {"--tokenize"})
        == {
            "--tokenize": True,
            "--model": str(target_path),
            "--text-file": str(text_path),
            "--out": str(Path(tokenizer["metadata"]["path"]).parent),
        }
        and token_command["environment"] == tokenizer_environment,
        "native tokenizer command differs",
    )
    fixed = token_ids(auth.checked(packet["continuation_tokens"]))
    require(inputs[prompt_count:] == fixed, "fixed continuation input history differs")
    command = auth.read(packet["capture_command"])
    argv = command["argv"]
    values = arguments(
        argv,
        {
            "--model",
            "--tokens",
            "--out",
            "--cpu-library",
            "--cuda-library",
            "--decode-tokens",
            "--prefill-chunk",
            "--decode-chunk",
            "--output-cap",
            "--threads",
            "--taps",
            "--max-output-mib",
            "--logit-positions",
        },
    )
    expected_values = {
        "--model": str(target_path),
        "--tokens": tokenizer["tokens"]["path"],
        "--out": str(Path(packet["capture"]["path"]).parent),
        "--cpu-library": expected_cpu[0],
        "--cuda-library": cuda_path,
        "--decode-tokens": packet["continuation_tokens"]["path"],
        "--prefill-chunk": str(capture["prefill_chunk"]),
        "--decode-chunk": str(capture["decode_chunk"]),
        "--output-cap": str(capture["output_cap"]),
        "--threads": str(capture["threads"]),
        "--taps": "0,2,16,29",
        "--max-output-mib": values.get("--max-output-mib"),
    }
    if "--logit-positions" in values:
        expected_values["--logit-positions"] = ",".join(map(str, positions))
    else:
        require(
            positions == list(range(prompt_count - 1, len(inputs))),
            "selected next-token positions require an explicit capture command",
        )
    require(
        argv[0] == build["executable"]["path"]
        and values == expected_values
        and command["environment"] == build["environment"],
        "capture command/build binding differs",
    )
    require(
        type(capture["prefill_chunk"]) is int
        and 1 <= capture["prefill_chunk"] <= 512
        and type(capture["decode_chunk"]) is int
        and 1 <= capture["decode_chunk"] <= 32
        and type(capture["output_cap"]) is int
        and 1 <= capture["output_cap"] <= 512
        and type(capture["threads"]) is int
        and 1 <= capture["threads"] <= 64
        and values["--max-output-mib"].isdigit()
        and 1 <= int(values["--max-output-mib"]) <= 128,
        "capture bounds differ",
    )
    required_maps = [x["path"] for x in build["runtime_libraries"]] + [cuda_path]
    runtime_maps(auth, packet["capture_library_maps"], trusted, required=required_maps)

    serving = packet["serving"]
    require(
        isinstance(serving, dict)
        and set(serving) == {"freeze", "request", "response", "library_maps"},
        "exact serving evidence required",
    )
    freeze = auth.read(serving["freeze"])
    require(
        same_pin(freeze["target"], packet["target"])
        and same_pin(freeze["cuda"], baseline["cuda"])
        and same_pin(freeze["server"], frozen["server"])
        and freeze["environment"].get("GGML_CUDA_DECODE_CONSISTENT") == "32",
        "serving target/runtime differs",
    )
    for name in ("source", "server", "command_ancestry"):
        auth.checked(freeze[name])
    server_argv = freeze["command"]
    server_values = arguments(
        server_argv,
        {
            "-m",
            "-ngl",
            "-c",
            "-np",
            "-b",
            "-ub",
            "-t",
            "-fa",
            "--cache-type-k",
            "--cache-type-v",
            "--host",
            "--port",
            "--spec-type",
        },
        {"--metrics", "--no-webui", "--no-warmup"},
    )
    require(
        server_argv[0] == frozen["server"]["path"]
        and server_values
        == {
            "-m": str(target_path),
            "-ngl": "99",
            "-c": "4096",
            "-np": "1",
            "-b": "512",
            "-ub": "512",
            "-t": "4",
            "-fa": "on",
            "--cache-type-k": "f16",
            "--cache-type-v": "f16",
            "--host": "127.0.0.1",
            "--port": server_values.get("--port"),
            "--metrics": True,
            "--no-webui": True,
            "--no-warmup": True,
            "--spec-type": "none",
        }
        and server_values["--port"].isdigit()
        and 1 <= int(server_values["--port"]) <= 65535,
        "serving command/settings differ",
    )
    runtime_maps(auth, serving["library_maps"], trusted, required=required_maps)
    request, response = auth.read(serving["request"]), auth.read(serving["response"])
    require(
        set(request)
        == {"prompt", "n_predict", "temperature", "seed", "cache_prompt", "return_tokens", "stream"}
        and request["prompt"] == raw_text
        and request["temperature"] == 0
        and request["seed"] == 1234
        and request["cache_prompt"] is False
        and request["return_tokens"] is True
        and request["stream"] is False
        and type(request["n_predict"]) is int
        and 1 <= request["n_predict"] <= 256,
        "serving request or raw prompt differs",
    )
    labels = response.get("tokens")
    require(
        isinstance(labels, list)
        and labels
        and all(type(x) is int and 0 <= x < VOCAB for x in labels)
        and len(labels) == response.get("tokens_predicted")
        and response.get("tokens_evaluated") == prompt_count
        and response.get("prompt") == raw_text
        and response.get("model") == str(target_path)
        and response.get("truncated") is False
        and response.get("stop_type") in ("eos", "limit", "word")
        and isinstance(response.get("stopping_word"), str)
        and len(labels) >= positions[-1] - prompt_count + 2
        and labels[: len(fixed)] == fixed,
        "serving response, labels or history differ",
    )
    sampling = response.get("generation_settings", {})
    for key, value in {
        "temperature": 0,
        "seed": 1234,
        "repeat_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "dry_multiplier": 0,
        "logit_bias": [],
        "grammar": "",
        "lora": [],
        "n_predict": request["n_predict"],
        "ignore_eos": False,
        "backend_sampling": False,
    }.items():
        require(sampling.get(key) == value, "serving sampling semantics differ: " + key)

    require(
        isinstance(packet["arrays"], dict) and set(packet["arrays"]) == ARRAY_NAMES,
        "exact native array inventory required",
    )
    matrices = {}
    total_bytes = 0
    for name, ref in packet["arrays"].items():
        path = auth.checked(ref)
        require(
            path == (Path(packet["capture"]["path"]).parent / name).resolve(),
            "native array path binding differs",
        )
        shape = capture["logits_shape"] if name == "logits.f32" else capture["feature_shape"]
        expected_bytes = shape[0] * shape[1] * 4
        require(
            path.stat().st_size == expected_bytes and expected_bytes <= 128 * 1024**2,
            "native array shape/size differs",
        )
        matrix = np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))
        for first in range(0, len(matrix), 32):
            require(np.isfinite(matrix[first : first + 32]).all(), "nonfinite native array")
        matrices[name] = matrix
        total_bytes += expected_bytes
    require(
        capture.get("array_bytes") == total_bytes
        and total_bytes <= int(values["--max-output-mib"]) * 1024**2,
        "array byte budget differs",
    )
    logits = matrices["logits.f32"]
    argmax = logits.argmax(axis=1).tolist()
    require(argmax == capture.get("greedy_ids"), "captured raw argmax differs from metadata")
    records = []
    for index, position in enumerate(positions):
        serving_index = position - prompt_count + 1
        label, native = labels[serving_index], argmax[index]
        agrees = label == native
        records.append(
            {
                "row_id": index,
                "input_position": position,
                "serving_token_index": serving_index,
                "serving_label": label,
                "native_argmax": native,
                "serving_label_native_margin": float(logits[index, label])
                - float(logits[index, native]),
                "supervision_eligible": agrees,
                "protection_numeric_eligible": agrees,
                "parent_success_protection_proven": False,
                "exclusion_reason": None if agrees else "native_argmax_differs_from_serving_label",
            }
        )
    # Borrowed target embedding rows must be token identities, not position- or
    # history-dependent hidden states accidentally supplied as tap0.
    seen = {}
    for index, token in enumerate(inputs):
        if token in seen:
            require(
                np.array_equal(
                    matrices["layer-0.f32"][index], matrices["layer-0.f32"][seen[token]]
                ),
                "native token embedding differs across repeated token IDs",
            )
        else:
            seen[token] = index
    auth.unchanged()
    return {
        "schema": "w1a8_modern_capture_admission_v1",
        "status": "NUMERIC_ROWS_ADMITTED",
        "target_id": TARGET_ID,
        "evidence": packet,
        "rows": records,
        "eligible_row_ids": [x["row_id"] for x in records if x["supervision_eligible"]],
        "excluded_row_ids": [x["row_id"] for x in records if not x["supervision_eligible"]],
        "native_embedding_first_row_by_token": {str(k): v for k, v in seen.items()},
        "scope": (
            "Development data numeric admission only; no parent-success, actor/native, "
            "quality, generalization or serving qualification"
        ),
        "parent_success_protection_proven": False,
        "cost": {
            "authentication_wall_seconds": time.perf_counter() - started,
            "authenticated_files": len(auth.files),
            "authenticated_bytes": auth.bytes,
            "training_updates": 0,
            "gpu_seconds": 0,
        },
    }


def require_eligible_rows(receipt, row_ids, *, purpose="supervision"):
    """Reject excluded rows at the consumer boundary; never grant parent proof."""
    require(purpose in ("supervision", "protection_numeric"), "unsupported admission purpose")
    field = purpose + "_eligible"
    require(
        receipt.get("schema") == "w1a8_modern_capture_admission_v1", "admission receipt required"
    )
    rows = {x["row_id"]: x for x in receipt["rows"]}
    require(
        len(rows) == len(receipt["rows"]) and len(set(row_ids)) == len(row_ids),
        "distinct row IDs required",
    )
    for row_id in row_ids:
        require(type(row_id) is int and row_id in rows, "unknown row ID")
        row = rows[row_id]
        require(
            row.get(field) is True
            and row["native_argmax"] == row["serving_label"]
            and row_id in receipt["eligible_row_ids"]
            and row_id not in receipt["excluded_row_ids"],
            "excluded row cannot be used for " + purpose,
        )
    return [rows[x] for x in row_ids]


def build_actor_steps(receipt):
    """Map adjacent teacher rows to shifted, teacher-forced EAGLE actor steps.

    A native teacher row at input position t labels token[t+1]. The actor
    pairs feature[t] with that token at decoder position t, then predicts
    token[t+2], supplied by the next teacher row. Both teacher rows must pass
    the selected numeric gate. The final teacher row has no actor label.

    The caller must authenticate the receipt and its evidence before use.
    This helper does not grant draft-vocabulary membership, native actor
    qualification, or parent-success protection. In recurrent free-running
    training, these labels remain valid only while proposals follow the
    recorded serving prefix; this mapping alone does not verify that prefix.
    """
    require(
        receipt.get("schema") == "w1a8_modern_capture_admission_v1",
        "admission receipt required",
    )
    rows = receipt.get("rows")
    require(isinstance(rows, list) and rows, "teacher rows required")
    start = rows[0].get("input_position")
    serving_start = rows[0].get("serving_token_index")
    require(type(start) is int and start >= 0, "nonnegative teacher input position required")
    require(
        type(serving_start) is int and 0 <= serving_start <= start,
        "nonnegative teacher serving offset required",
    )
    for index, row in enumerate(rows):
        require(
            row.get("row_id") == index
            and type(row.get("row_id")) is int
            and type(row.get("input_position")) is int
            and row["input_position"] == start + index
            and type(row.get("serving_token_index")) is int
            and row["serving_token_index"] == serving_start + index,
            "teacher row IDs, positions and serving indices must be contiguous",
        )
        require(
            all(
                type(row.get(key)) is int and 0 <= row[key] < VOCAB
                for key in ("serving_label", "native_argmax")
            ),
            "teacher labels must be target-vocabulary token IDs",
        )
    steps = []
    for index, (source, label) in enumerate(zip(rows, rows[1:])):
        teacher_ids = [source["row_id"], label["row_id"]]
        blocked = {}
        for purpose in ("supervision", "protection_numeric"):
            blocked[purpose] = []
            for row_id in teacher_ids:
                try:
                    require_eligible_rows(receipt, [row_id], purpose=purpose)
                except ValueError:
                    blocked[purpose].append(row_id)
        position = source["input_position"]
        steps.append(
            {
                "actor_step_id": index,
                "source_teacher_row_id": source["row_id"],
                "label_teacher_row_id": label["row_id"],
                "feature_input_position": position,
                "decoder_position": position,
                "input_position": position + 1,
                "label_position": position + 2,
                "input_token_id": source["serving_label"],
                "label_token_id": label["serving_label"],
                "supervision_eligible": not blocked["supervision"],
                "protection_numeric_eligible": not blocked["protection_numeric"],
                "excluded_teacher_row_ids": blocked,
                "parent_success_protection_proven": False,
                "draft_vocab_membership_proven": False,
                "actor_native_qualification_proven": False,
                "alignment": "feature_t_input_t_plus_1_label_t_plus_2",
                "interpretation": "teacher_forced_adjacent_transition",
            }
        )
    return steps


def require_actor_steps(receipt, step_ids, *, purpose="supervision"):
    """Recompute shifted actor admission; caller-supplied step flags are unused."""
    require(purpose in ("supervision", "protection_numeric"), "unsupported admission purpose")
    require(
        isinstance(step_ids, (list, tuple))
        and all(type(value) is int for value in step_ids)
        and len(set(step_ids)) == len(step_ids),
        "distinct actor step IDs required",
    )
    steps = {step["actor_step_id"]: step for step in build_actor_steps(receipt)}
    selected = []
    for step_id in step_ids:
        require(step_id in steps, "unknown actor step ID or final teacher boundary")
        step = steps[step_id]
        require_eligible_rows(
            receipt,
            [step["source_teacher_row_id"], step["label_teacher_row_id"]],
            purpose=purpose,
        )
        selected.append(step)
    return selected


def prepare(packet_path, output):
    packet_path, output = Path(packet_path), Path(output)
    require(not output.exists(), "refuse to overwrite admission receipt")
    result = admit_packet(json.loads(packet_path.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic, exclusive publication prevents a partial or another worker's
    # receipt from being mistaken for this completed admission transaction.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=".admission-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--packet", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.packet, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "eligible_rows": len(result["eligible_row_ids"]),
                "excluded_rows": len(result["excluded_row_ids"]),
                "cost": result["cost"],
            }
        )
    )


if __name__ == "__main__":
    main()

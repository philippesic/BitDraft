"""Versioned FIT-only expansion: freeze 24 unused TRAIN groups, then collect.

The collect path is an isolated derivative of the original actual-parent collector,
whose full source SHA256 is ORIGINAL_COLLECTOR_SHA256 below. Existing collectors,
original36, DEVELOPMENT and consumed fresh inputs are read-only. GPU owner must
launch collect through remote_job with --allow-cuda and the shared CUDA cache.
Preparation reads source metadata only; it does not load a model or native code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import unicodedata
from collections import Counter

import modern_collect as lifecycle

P = Path("/home/philip/binary-eagle-decoding")
ORIGINAL_COLLECTOR = P / "runs/w1a8-modern-actual-parent-capture-20261103-01/collect.py"
ORIGINAL_COLLECTOR_SHA256 = "d6f652229ddc5f558e49909e07aa163652fb193e5940aca133a2acc5ff17c97b"
C_ROOT = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts")
EXPANSION_SCHEMA = "w1a8_modern_phase1_fit_expansion_roster_v1"
SEED = "w1a8-modern4b-fit-expansion-v1"
PARENT_SHA256 = "6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048"
TARGET_SHA256 = "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"
ROSTER = P / "runs/w1a8-modern-generalization-data-20261103-01/roster02.json"
FRESH = P / "runs/w1a8-modern-protected-20261103-01/fresh-quarter-confirmation01/inputs01"
AUDIT = P / "runs/w1a8-modern-protected-20261103-01/fit-expansion-source-audit.json"
SPLIT = C_ROOT / "5080-broader-training-20261031/split.json"
BINDINGS = [
    P / ("runs/w1a8-algorithms-20261020/" + name + "/bindings.json")
    for name in ("loop2-answer-expand01", "loop2-mode-capture05")
]
PLANNED_BYTES = 512 * 1024**2
FUTURE_STATE_BYTES = 80 * 1024**2
METADATA_BYTES = 8 * 1024**2
TRACE_BUILD = P / "runs/w1a8-modern-trace-build-20261103-02/build.json"
CACHE = P / "runs/local-w1a8-modern-serving-trace-20261103-01/cuda"
MAP = P / "runs/w1a8-modern-native-replay-20261103-01/draft-map.ids"
KNOWN_EMBEDDING_CAPTURE = P / "runs/w1a8-modern-generalization-capture-20261103-02"
PER_GROUP = 16 * 1024**2
TRACE_CAP = 64 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response_signature(value):
    keys = (
        "tokens",
        "content",
        "stop",
        "stop_type",
        "stopping_word",
        "truncated",
        "tokens_predicted",
        "has_new_line",
    )
    require(all(key in value for key in keys), "raw token/text/stop response fields missing")
    require(
        isinstance(value["tokens"], list)
        and value["tokens"]
        and all(type(token) is int and 0 <= token < 248320 for token in value["tokens"])
        and len(value["tokens"]) == value["tokens_predicted"],
        "invalid raw completion token IDs",
    )
    return {key: value[key] for key in keys}


def trace_audit(cell, process_id, trace_build, map_ref, parser, utils, response):
    trace = cell / "trace"
    manifest = read(trace / "manifest.json")
    require(
        manifest["schema"] == "w1a8_actual_native_actor_trace_v2"
        and manifest["pid"] == process_id
        and manifest["cap_bytes"] == TRACE_CAP
        and manifest["required_llama"] == trace_build["environment"]["W1A8_TRACE_LLAMA"]
        and manifest["draft_map"] == map_ref["path"]
        and manifest["operation_configuration_changed"] is False
        and manifest["model_opened_by_interposer"] is False,
        "corrected v2 trace manifest differs",
    )
    files = sorted(path for path in trace.iterdir() if path.is_file())
    require(
        all(not path.is_symlink() and not path.name.startswith(".") for path in files),
        "trace contains a temporary or substituted file",
    )
    actual_bytes = sum(path.stat().st_size for path in files)
    require(actual_bytes <= TRACE_CAP, "per-process trace exceeded64MiB cap")
    events = [json.loads(line) for line in (trace / "calls.jsonl").read_text().splitlines() if line]
    require(
        events and [event["event"] for event in events] == list(range(1, len(events) + 1)),
        "missing/reordered native trace events",
    )
    before = {}
    after = set()
    native_proposals = []
    variants = {}
    for event in events:
        call = event["call"]
        if event["phase"] == "before":
            require(call not in before, "duplicate native before call")
            before[call] = event
        else:
            require(
                event["phase"] == "after"
                and call in before
                and call not in after
                and event["result"] == 0,
                "failed/unpaired native call",
            )
            after.add(call)
            source = before[call]
            if source["architecture"] != "eagle3" or source["process_type_name"] != "decode":
                continue
            for selected in event.get("selected", []):
                batch_row = source["batch"]["rows"][selected["batch_row"]]
                # Model initialization can decode unconditioned placeholder tokens.
                # Actual EAGLE proposals have the supplied feature embedding.
                if not batch_row["has_embd"]:
                    continue
                variant = selected["logits_mode"]
                require(
                    variant
                    in ("backend_explicit_candidate_pairs", "dense_target_vocab_mapped_subset"),
                    "ambiguous/old decoder-logit trace schema",
                )
                variants[variant] = variants.get(variant, 0) + 1
                native_proposals.append(selected["argmax_target_token"])
    require(set(before) == after, "native call trace ended before completion")
    text = (cell / "server.log").read_text()
    audit = parser.audit_trace(text, response, expected_depth=2)
    lines, _, _ = parser._request_lines(text, None, None)
    logged = []
    for _, line in lines:
        candidate = parser.CANDIDATE.search(line)
        if candidate and int(candidate[2]) == 0:
            logged.append(int(candidate[4]))
    require(
        native_proposals == logged,
        "actual native output proposals differ from serving debug decisions",
    )
    audit["inputs"] = {
        "log": utils.pin(cell / "server.log"),
        "response": utils.pin(cell / "response.json"),
        "parser": utils.pin(Path(parser.__file__)),
    }
    audit["native_output_pair_audit"] = {
        "matched_debug_proposals": len(logged),
        "logits_modes": variants,
        "trace_bytes": actual_bytes,
        "trace_files": [utils.pin(path) for path in files],
    }
    utils.atomic(cell / "protection.json", audit)
    return audit


def collect(args):
    require(args.allow_cuda, "sole GPU owner must explicitly pass --allow-cuda")
    require(not args.output.exists(), "actual-parent output directory already exists")
    require(
        os.environ.get("CUDA_CACHE_PATH")
        and Path(os.environ["CUDA_CACHE_PATH"]).resolve() == CACHE.resolve()
        and CACHE.is_dir(),
        "root COMMAND must override CUDA_CACHE_PATH to the existing shared cache",
    )
    roster = read(args.roster)
    validate_roster(roster)
    check_destination(args.output)
    storage_guard(args.output.parent, args.planned_bytes, args.future_state_bytes)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    source_root = args.output / "source"
    source_root.mkdir()
    created = True
    originals = [
        Path(__file__),
        Path(lifecycle.__file__),
        Path(__file__).with_name("modern_serving_trace.py"),
        ORIGINAL_COLLECTOR,
    ]
    require(
        lifecycle.pin(ORIGINAL_COLLECTOR)["sha256"] == ORIGINAL_COLLECTOR_SHA256,
        "original collector ancestry changed",
    )
    for original in originals:
        shutil.copyfile(original, source_root / original.name)
    utils = load_module("actual_parent_collect_utils", source_root / "modern_collect.py")
    parser = load_module("actual_parent_trace_parser", source_root / "modern_serving_trace.py")
    for original in originals:
        require(
            utils.pin(original)["sha256"] == utils.pin(source_root / original.name)["sha256"],
            "source changed while snapshotting",
        )
    worker = ExpansionCollector(args)
    worker.created_output = created
    source_pins = [utils.pin(source_root / path.name) for path in originals]
    cleaning = False

    def stop(signum, frame):
        if not cleaning:
            raise utils.StopRequested(f"received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    try:
        worker.stage = "authenticate_actual_parent_inputs"
        worker.permission()
        worker.budget(PLANNED_BYTES)
        for ref in roster["source_pins"]:
            worker.checked(ref)
        # Recompute selection from pinned source metadata, never from model outcomes.
        expected_rows, _ = select_rows(args.current_roster, args.split, args.bindings, args.fresh)
        require(
            expected_rows
            == [
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("rendered_text_path", "rendered_prompt_sha256")
                }
                for row in roster["rows"]
            ],
            "frozen expansion selection differs",
        )
        initial = read(P / "runs/w1a8-modern-native-admission-20261103-01/freeze.json")
        trace_build = read(args.build)
        require(trace_build["output_cap_bytes"] == TRACE_CAP, "trace cap differs from64MiB")
        require(
            utils.pin(trace_build["source"]["path"])["sha256"]
            == "1c09194a1f54240f31e51e06c7647d52878214c2f25336b1fcfbf8075cb38f12",
            "corrected interposer v2 source required",
        )
        references = [
            initial["baseline"],
            initial["target"],
            initial["w1a8_initialization"],
            trace_build["candidate_build"],
            trace_build["candidate_source_pins"],
            trace_build["source"],
            trace_build["builder"],
            trace_build["interposer"],
            trace_build["compiler"],
        ]
        references += trace_build["candidate_binaries"] + trace_build["headers"]
        references += [
            utils.pin(args.roster),
            utils.pin(args.build),
            utils.pin(MAP),
            utils.pin(P / "runs/w1a8-modern-native-admission-20261103-01/freeze.json"),
            utils.pin(ORIGINAL_COLLECTOR),
        ]
        for ref in references:
            worker.checked(ref)
        require(
            initial["w1a8_initialization"]["sha256"] == PARENT_SHA256
            and initial["target"]["sha256"] == TARGET_SHA256,
            "expansion requires unchanged 6dac parent and target",
        )
        require(
            initial["candidate_build"] == trace_build["candidate_build"],
            "native parent runtime pins differ",
        )
        candidate = read(trace_build["candidate_build"]["path"])
        for ref in candidate["binaries"]:
            worker.checked(ref)
        server = next(
            ref for ref in candidate["binaries"] if Path(ref["path"]).name == "llama-server"
        )
        map_ref = utils.pin(MAP)
        map_values = [int(value) for value in MAP.read_text().split()]
        require(
            len(map_values) == len(set(map_values)) == 50000
            and min(map_values) >= 0
            and max(map_values) < 248320,
            "absolute native draft map invalid",
        )
        map_raw = b"".join(value.to_bytes(8, "little", signed=True) for value in map_values)
        native_export = read(P / "runs/w1a8-modern-packed-init-20261103-01/export.json")
        require(
            hashlib.sha256(map_raw).hexdigest()
            == native_export["preserved_tensors"]["d2t"]["raw_sha256"]
            and native_export["output"]["sha256"] == initial["w1a8_initialization"]["sha256"],
            "trace map or parent export differs",
        )
        worker.checked(utils.pin(P / "runs/w1a8-modern-packed-init-20261103-01/export.json"))
        groups = roster["rows"]
        require(
            len({row["id"] for row in groups}) == 24
            and len({row["group_id"] for row in groups}) == 24,
            "unique24 whole FIT groups required",
        )
        clean = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("GGML_", "DSPARK_", "DFLASH_", "EAGLE_", "W1AX_", "LLAMA_", "W1A8_")
            )
            and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "NVIDIA_TF32_OVERRIDE")
        }
        target_environment = {
            **clean,
            "GGML_CUDA_DECODE_CONSISTENT": "32",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "4",
            "CUDA_CACHE_PATH": str(CACHE.resolve()),
            "LD_LIBRARY_PATH": trace_build["environment"]["LD_LIBRARY_PATH"],
        }
        requests = target_references(worker, groups, server, initial, candidate, target_environment)
        worker.budget(PLANNED_BYTES)
        utils.atomic(
            args.output / "freeze.json",
            {
                "source_pins": source_pins,
                "input_pins": list(worker.authenticated.values()),
                "roster": utils.pin(args.roster),
                "conditional": str(KNOWN_EMBEDDING_CAPTURE),
                "conditional_usage": "Existing exact native target layer0 embedding bank only; no conditional hidden features enter actual-serving views",
                "target_references": str((args.output / "target-only").resolve()),
                "expansion_schema": EXPANSION_SCHEMA,
                "selection": roster["selection"],
                "trace_build": utils.pin(args.build),
                "parent": initial["w1a8_initialization"],
                "target": initial["target"],
                "depth": 2,
                "groups": [row["id"] for row in groups],
                "cache_reused": str(CACHE.resolve()),
                "per_group_predicted_bytes": PER_GROUP,
                "planned_bytes": args.planned_bytes,
                "future_state_reserve_bytes": args.future_state_bytes,
                "both_filesystems_guarded": True,
                "trace_per_process_cap_bytes": TRACE_CAP,
                "storage_floor_bytes": utils.FLOOR,
                "extra_reserve_bytes": utils.RESERVE,
                "scope": "Actual native-parent emitted successes across all80returned tokens; no CPU actor equivalence or performance qualification",
            },
        )
        clean = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("GGML_", "DSPARK_", "DFLASH_", "EAGLE_", "W1AX_", "LLAMA_", "W1A8_")
            )
            and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "NVIDIA_TF32_OVERRIDE")
        }
        environment = {
            **clean,
            **trace_build["environment"],
            "GGML_CUDA_DECODE_CONSISTENT": "32",
            "GGML_W1AX_ACT_BITS": "8",
            "LLAMA_TRACE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "4",
            "CUDA_CACHE_PATH": str(CACHE.resolve()),
            "W1A8_TRACE_MAP": map_ref["path"],
        }
        failures = 0
        for index, row in enumerate(groups):
            worker.stage = "actual_parent_" + row["id"]
            worker.permission()
            worker.budget((len(groups) - index) * PER_GROUP)
            cell = args.output / row["id"]
            cell.mkdir()
            trace = cell / "trace"
            trace.mkdir()
            env = {**environment, "W1A8_TRACE_DIR": str(trace)}
            request, expected = requests[row["id"]]
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            argv = [
                server["path"],
                "-m",
                initial["target"]["path"],
                "-ngl",
                "99",
                "-c",
                "4096",
                "-np",
                "1",
                "-b",
                "512",
                "-ub",
                "512",
                "-t",
                "4",
                "-fa",
                "on",
                "--cache-type-k",
                "f16",
                "--cache-type-v",
                "f16",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--metrics",
                "--no-webui",
                "--no-warmup",
                "--spec-type",
                "draft-eagle3",
                "-md",
                initial["w1a8_initialization"]["path"],
                "-ngld",
                "99",
                "--cache-type-k-draft",
                "f16",
                "--cache-type-v-draft",
                "f16",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-n-min",
                "0",
                "--spec-draft-p-min",
                "0",
                "--spec-draft-sampling",
                "greedy",
                "-lv",
                "5",
            ]
            utils.atomic(
                cell / "command.json",
                {
                    "argv": argv,
                    "environment": {
                        key: value
                        for key, value in env.items()
                        if key.startswith(("LD_", "W1A8_", "LLAMA_", "GGML_", "CUDA_", "OMP_"))
                    },
                },
            )
            utils.atomic(cell / "request.json", request)
            utils.atomic(
                cell / "target-reference.json",
                {
                    "request": utils.pin(args.output / "target-only" / row["id"] / "request.json"),
                    "response": utils.pin(
                        args.output / "target-only" / row["id"] / "response.json"
                    ),
                },
            )
            began = time.perf_counter()
            result = None
            try:
                with (cell / "server.log").open("x") as log:
                    worker.active = subprocess.Popen(
                        argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    worker.active_identity = utils.process_identity(worker.active)
                    process_id = worker.active.pid
                    utils.atomic(cell / "identity.json", worker.active_identity)
                    worker.progress()
                    for _ in range(240):
                        worker.permission()
                        require(
                            worker.active.poll() is None,
                            "native parent server exited during startup",
                        )
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/health", timeout=1
                            ):
                                break
                        except (urllib.error.URLError, TimeoutError):
                            time.sleep(0.1)
                    else:
                        raise TimeoutError("native parent startup timed out")
                    maps = (Path("/proc") / str(process_id) / "maps").read_text()
                    (cell / "loaded-library-maps.txt").write_text(maps)
                    mapped = {
                        line.split()[-1]
                        for line in maps.splitlines()
                        if any(
                            name in line
                            for name in ("libggml", "libllama", "libmtmd", "libw1a8-modern-trace")
                        )
                    }
                    allowed = {
                        str(Path(ref["path"]).resolve()) for ref in candidate["binaries"]
                    } | {trace_build["interposer"]["path"]}
                    require(
                        mapped <= allowed
                        and trace_build["interposer"]["path"] in mapped
                        and trace_build["environment"]["W1A8_TRACE_LLAMA"] in mapped
                        and any("libggml-cuda.so" in path for path in mapped),
                        "actual native library maps differ",
                    )
                    for path in mapped:
                        worker.checked(
                            next(
                                ref
                                for ref in candidate["binaries"] + [trace_build["interposer"]]
                                if str(Path(ref["path"]).resolve()) == path
                            )
                        )
                    native_request = urllib.request.Request(
                        f"http://127.0.0.1:{port}/completion",
                        data=json.dumps(request).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    worker.permission()
                    with urllib.request.urlopen(native_request, timeout=120) as response:
                        observed = json.load(response)
                    utils.atomic(cell / "response.json", observed)
                    require(
                        response_signature(observed) == response_signature(expected),
                        "actual parent differs from frozen target tokens/text/stop/truncation",
                    )
                    worker.reap(cell / "release.json")
                audit = trace_audit(cell, process_id, trace_build, map_ref, parser, utils, observed)
                result = {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "role": row["role"],
                    "domain": row["domain"],
                    "status": "ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS",
                    "token_text_stop_parity": True,
                    "response_tokens": len(observed["tokens"]),
                    "draft_proposed": audit["draft_proposed"],
                    "draft_verified_accepted": audit["draft_verified_accepted"],
                    "emitted_draft_successes": audit["emitted_draft_successes"],
                    "protection": utils.pin(cell / "protection.json"),
                    "trace_bytes": audit["native_output_pair_audit"]["trace_bytes"],
                    "source_identity": row["source_ancestry"],
                    "source_group": row,
                    "cpu_native_actor_equivalence_proven": False,
                }
                failures = 0
            except (ValueError, OSError, TimeoutError, subprocess.TimeoutExpired) as error:
                result = {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "role": row["role"],
                    "domain": row["domain"],
                    "status": "ACTUAL_PARENT_TRACE_FAILED_RETAINED",
                    "error": repr(error),
                    "cpu_native_actor_equivalence_proven": False,
                }
                failures += 1
            finally:
                worker.reap(cell / "release.json")
            result["wall_seconds"] = time.perf_counter() - began
            utils.atomic(cell / "result.json", result)
            worker.groups.append(result)
            worker.costs.append(
                {
                    "stage": "actual_parent_" + row["id"],
                    "wall_seconds": result["wall_seconds"],
                    "status": result["status"],
                }
            )
            worker.progress()
            print(json.dumps(result), flush=True)
            require(
                failures < 3,
                "three consecutive actual-parent failures; inspect preserved inputs and trace",
            )
        require(
            directory_bytes(args.output) <= args.planned_bytes,
            "total capture exceeded planned byte bound",
        )
        storage_guard(
            args.output,
            args.planned_bytes,
            args.future_state_bytes,
            used=directory_bytes(args.output),
        )
        worker.stage = "complete"
        utils.atomic(
            args.output / "summary.json",
            {
                "status": "ACTUAL_PARENT_COLLECTION_COMPLETE_WITH_ACCOUNTED_FAILURES",
                "groups": worker.groups,
                "costs": worker.costs,
                "wall_seconds": time.perf_counter() - worker.started,
                "actual_native_protections": sum(
                    group.get("emitted_draft_successes", 0) for group in worker.groups
                ),
                "free_bytes_after": shutil.disk_usage(args.output).free,
                "training_updates": 0,
                "cpu_native_actor_equivalence_proven": False,
                "scope": "Actual emitted native-parent successes retained across entire response; no speed or generalization claim",
            },
        )
        worker.progress()
    except BaseException as error:
        cleaning = True
        worker.reap(args.output / "interrupted-release.json")
        utils.atomic(
            args.output / "failure.json",
            {
                "stage": worker.stage,
                "error": repr(error),
                "groups": worker.groups,
                "costs": worker.costs,
                "wall_seconds": time.perf_counter() - worker.started,
            },
        )
        raise
    finally:
        cleaning = True
        worker.reap(args.output / "final-release.json")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def identity(messages, normalize=False):
    require(
        isinstance(messages, list)
        and messages
        and all(
            isinstance(m, dict)
            and isinstance(m.get("content"), str)
            and m.get("role") in ("user", "system", "assistant")
            for m in messages
        ),
        "invalid source messages",
    )
    value = [(m["role"], normalized(m["content"]) if normalize else m["content"]) for m in messages]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def render(messages, rendering):
    require(
        len(messages) == 1 and messages[0]["role"] == "user",
        "preserve source chat; unsupported wrapper",
    )
    return (
        rendering["prefix"]
        + "<|im_start|>user\n"
        + messages[0]["content"]
        + "<|im_end|>\n"
        + rendering["assistant_suffix"]
    )


def messages_from_prompt(prompt):
    matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    require(len(matches) == 1, "fresh request user wrapper ambiguous")
    return [{"role": "user", "content": matches[0]}]


def select_rows(current_path, split_path, binding_paths, fresh):
    """Metadata-only deterministic selection. No response/score or label files read."""
    pins = {}

    def source(path):
        ref = lifecycle.pin(path)
        value = read(path)
        require(lifecycle.pin(path) == ref, "source changed while reading")
        pins[ref["path"]] = ref
        return value

    def receipt(ref):
        actual = lifecycle.pin(ref["path"])
        require(
            actual["sha256"] == ref["sha256"]
            and ("bytes" not in ref or actual["bytes"] == ref["bytes"]),
            "source receipt pin changed",
        )
        return source(ref["path"])

    current = source(current_path)
    require(
        current["schema"] == "w1a8_modern_phase1_generalization_roster_v2"
        and len(current["rows"]) == 36
        and current["rows"] == current["roster"]
        and Counter(r["role"] for r in current["rows"]) == {"fit": 24, "development": 12},
        "immutable original36 roster required",
    )
    original = current["rows"]
    fresh = Path(fresh)
    fresh_manifest, fresh_receipt = source(fresh / "manifest.json"), source(fresh / "receipt.json")
    require(
        fresh_receipt["manifest_sha256"]
        == pins[str((fresh / "manifest.json").resolve())]["sha256"],
        "fresh manifest pin differs",
    )
    fresh_requests = source(fresh / "requests.json")
    require(
        fresh_receipt["files"]["requests.json"]
        == pins[str((fresh / "requests.json").resolve())]["sha256"],
        "fresh request pin differs",
    )
    require(
        isinstance(fresh_requests, list)
        and fresh_requests
        and fresh_manifest.get("admitted") is True,
        "consumed fresh input packet missing",
    )
    fresh_messages = [messages_from_prompt(r["prompt"]) for r in fresh_requests]
    # The manifest carries all performance AND quality raw/rendered identities.
    # Do not read quality answer/test payloads merely to exclude their inputs.
    fresh_raw = {g["identities"]["raw_sha256"] for g in fresh_manifest["groups"]}
    fresh_normal = {g["identities"]["normalized_raw_sha256"] for g in fresh_manifest["groups"]}
    require(
        len(fresh_manifest["groups"]) == 48 and len(fresh_raw) == 48 and len(fresh_normal) == 48,
        "complete consumed fresh48 identities required",
    )
    excluded_exact = {identity(r["messages"]) for r in original} | {
        identity(m) for m in fresh_messages
    }
    excluded_normal = {identity(r["messages"], True) for r in original} | {
        identity(m, True) for m in fresh_messages
    }
    excluded_groups = {r["group_id"] for r in original} | {
        r["group_id"] for r in fresh_manifest["groups"]
    }
    cases = source(split_path)["cases"]
    require(len({c["group_id"] for c in cases}) == len(cases), "duplicate historical split group")
    cases = {c["group_id"]: c for c in cases}
    raw = {}
    for path in binding_paths:
        for index, row in enumerate(source(path)["rows"]):
            require(row["group_id"] not in raw, "duplicate whole source group")
            raw[row["group_id"]] = {
                **row,
                "binding_path": str(Path(path).resolve()),
                "binding_row": index,
            }
    require(set(cases) <= set(raw), "historical split has missing binding ancestry")
    selected, seen = [], set(excluded_normal)
    for domain in ("code", "reasoning", "prose"):
        pool = [
            r
            for gid, r in raw.items()
            if gid in cases
            and cases[gid]["role"] == "fit"
            and r["domain"] == domain
            and r["source_ancestry"]["source_split"] == "TRAIN"
            and gid not in excluded_groups
            and identity(r["messages"]) not in excluded_exact
            and identity(r["messages"], True) not in excluded_normal
            and len(r["messages"]) == 1
            and r["messages"][0]["role"] == "user"
            and hashlib.sha256(r["messages"][0]["content"].encode()).hexdigest() not in fresh_raw
            and hashlib.sha256(normalized(r["messages"][0]["content"]).encode()).hexdigest()
            not in fresh_normal
        ]
        pool.sort(key=lambda r: hashlib.sha256((SEED + "\0" + r["group_id"]).encode()).hexdigest())
        chosen = 0
        for row in pool:
            message_id = identity(row["messages"], True)
            if message_id in seen:
                continue
            case, ancestry = cases[row["group_id"]], row["source_ancestry"]
            require(
                case["source_split"] == "TRAIN"
                and case["domain"] == ancestry["domain"] == domain
                and row["prompt_id"] == ancestry["prompt_id"],
                "TRAIN/domain/prompt ancestry differs",
            )
            old_receipt = receipt(row["original_receipt"])
            require(
                old_receipt["prompt"]["messages"] == row["messages"]
                and old_receipt["prompt_source_sha256"] == ancestry["prompt_sha256"]
                and all(
                    old_receipt["chain_ancestry"][key] == ancestry[key]
                    for key in ("domain", "prompt_id", "source_split", "prompt_sha256")
                ),
                "original source messages or chain ancestry differs",
            )
            split_receipt = receipt(case["ancestry"])
            require(
                all(
                    split_receipt["chain_ancestry"][key] == ancestry[key]
                    for key in ("domain", "prompt_id", "source_split", "prompt_sha256")
                ),
                "split receipt source ancestry differs",
            )
            expected_old_prompt = (
                "<|im_start|>user\n"
                + row["messages"][0]["content"]
                + "<|im_end|>\n"
                + current["rendering"]["assistant_suffix"]
            )
            require(
                split_receipt["prompt"].get("text") == expected_old_prompt
                and split_receipt["tokens"][: case["prompt_length"]] == case["prompt_tokens"],
                "historical split prompt/token ancestry differs",
            )
            selected.append(
                {
                    "id": f"fit-expand-v1-{domain}-{chosen:02d}",
                    "role": "fit",
                    "domain": domain,
                    "group_id": row["group_id"],
                    "historical_split_role": case["role"],
                    "historical_binding_role": row["role"],
                    "historical_source_split": "TRAIN",
                    "binding_path": row["binding_path"],
                    "binding_row": row["binding_row"],
                    "prompt_id": row["prompt_id"],
                    "source_ancestry": ancestry,
                    "original_receipt": row["original_receipt"],
                    "historical_split_receipt": case["ancestry"],
                    "messages": row["messages"],
                    "message_identity_sha256": identity(row["messages"]),
                    "normalized_message_identity_sha256": message_id,
                    "historical_prompt_tokens": case["prompt_length"],
                }
            )
            seen.add(message_id)
            chosen += 1
            if chosen == 8:
                break
        require(chosen == 8, "not enough independent unused TRAIN FIT groups: " + domain)
    for ref in pins.values():
        require(lifecycle.pin(ref["path"]) == ref, "source changed during selection")
    return selected, list(pins.values())


def validate_roster(roster):
    require(
        roster.get("schema") == EXPANSION_SCHEMA
        and roster.get("status") == "FROZEN_INPUTS_NO_LABELS"
        and len(roster["rows"]) == 24
        and Counter((r["role"], r["domain"]) for r in roster["rows"])
        == {("fit", d): 8 for d in ("code", "reasoning", "prose")}
        and len({r["id"] for r in roster["rows"]})
        == len({r["group_id"] for r in roster["rows"]})
        == 24
        and roster["protocol"] == protocol()
        and roster["parent_sha256"] == PARENT_SHA256
        and roster["target_sha256"] == TARGET_SHA256,
        "distinct balanced expansion roster/protocol required",
    )
    for row in roster["rows"]:
        require(
            re.fullmatch(r"fit-expand-v1-(code|reasoning|prose)-[0-9]{2}", row["id"])
            and row["historical_source_split"] == row["source_ancestry"]["source_split"] == "TRAIN"
            and row["message_identity_sha256"] == identity(row["messages"])
            and row["normalized_message_identity_sha256"] == identity(row["messages"], True),
            "expansion alias or source identity differs",
        )
        require(
            lifecycle.pin(row["rendered_text_path"])["sha256"] == row["rendered_prompt_sha256"],
            "frozen rendered prompt changed",
        )
        require(
            Path(row["rendered_text_path"]).read_text()
            == render(row["messages"], roster["rendering"]),
            "rendered prompt differs from source messages",
        )


def protocol():
    return {
        "n_predict": 80,
        "temperature": 0,
        "seed": 1234,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": False,
    }


def check_destination(path):
    path = Path(path).resolve()
    require(
        path.is_relative_to(C_ROOT.resolve()) and path != C_ROOT.resolve(),
        "raw capture must write directly to a unique C artifact directory",
    )
    require(not path.exists(), "capture destination already exists")


def directory_bytes(path):
    return (
        sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        if Path(path).exists()
        else 0
    )


def storage_guard(destination, planned, future, *, used=0):
    require(
        type(planned) is int
        and TRACE_CAP <= planned <= PLANNED_BYTES
        and type(future) is int
        and future >= FUTURE_STATE_BYTES
        and 0 <= used <= planned,
        "invalid planned capture bound/future state reserve",
    )
    for path, additional in ((P, METADATA_BYTES), (destination, planned - used + future)):
        path = Path(path)
        while not path.exists():
            path = path.parent
        free = shutil.disk_usage(path).free
        require(
            free >= lifecycle.FLOOR + lifecycle.RESERVE + additional,
            f"insufficient storage at {path}: free={free}, additional={additional}",
        )


class ExpansionCollector(lifecycle.Collector):
    """Reuse owned-process identity/stop protocol without changing shared classes."""

    def budget(self, remaining):
        used = directory_bytes(self.output)
        require(
            used + TRACE_CAP + METADATA_BYTES <= self.args.planned_bytes,
            "remaining total capture budget cannot cover next64MiB native trace; stop without replacement",
        )
        storage_guard(self.output, self.args.planned_bytes, self.args.future_state_bytes, used=used)
        return shutil.disk_usage(self.output).free


def target_references(worker, groups, server, initial, candidate, environment):
    """Frozen target-only runtime; exact80-token protocol, no dense teacher export."""
    root = worker.output / "target-only"
    root.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    argv = [
        server["path"],
        "-m",
        initial["target"]["path"],
        "-ngl",
        "99",
        "-c",
        "4096",
        "-np",
        "1",
        "-b",
        "512",
        "-ub",
        "512",
        "-t",
        "4",
        "-fa",
        "on",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        "--no-webui",
        "--no-warmup",
        "--spec-type",
        "none",
    ]
    lifecycle.atomic(
        root / "command.json",
        {
            "argv": argv,
            "environment": {
                key: value
                for key, value in environment.items()
                if key.startswith(("LD_", "W1A8_", "LLAMA_", "GGML_", "CUDA_", "OMP_"))
            },
        },
    )
    requests = {}
    try:
        worker.permission()
        worker.budget(PLANNED_BYTES)
        with (root / "server.log").open("x") as log:
            worker.active = subprocess.Popen(
                argv, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            worker.active_identity = lifecycle.process_identity(worker.active)
            lifecycle.atomic(root / "identity.json", worker.active_identity)
            for _ in range(240):
                worker.permission()
                require(worker.active.poll() is None, "target-only server exited during startup")
                try:
                    worker.request(port, "health", timeout=1)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise TimeoutError("target-only startup timed out")
            maps = (Path("/proc") / str(worker.active.pid) / "maps").read_text()
            (root / "loaded-library-maps.txt").write_text(maps)
            mapped = {
                line.split()[-1]
                for line in maps.splitlines()
                if any(
                    name in line
                    for name in ("libggml", "libllama", "libmtmd", "libw1a8-modern-trace")
                )
            }
            allowed = {str(Path(ref["path"]).resolve()) for ref in candidate["binaries"]}
            require(
                mapped <= allowed and any("libggml-cuda.so" in x for x in mapped),
                "target-only runtime libraries differ",
            )
            for path in mapped:
                worker.checked(
                    next(
                        ref
                        for ref in candidate["binaries"]
                        if str(Path(ref["path"]).resolve()) == path
                    )
                )
            for row in groups:
                worker.permission()
                worker.budget(PLANNED_BYTES)
                cell = root / row["id"]
                cell.mkdir()
                prompt = Path(row["rendered_text_path"]).read_text()
                tokens = worker.request(
                    port,
                    "tokenize",
                    {"content": prompt, "add_special": True, "parse_special": True},
                )
                lifecycle.atomic(cell / "tokenization.json", tokens)
                token_ids = tokens["tokens"]
                require(
                    token_ids
                    and all(type(t) is int and 0 <= t < 248320 for t in token_ids)
                    and len(token_ids) + 80 <= 384,
                    "selected whole group exceeds native geometry; no replacement",
                )
                request = {"prompt": prompt, **protocol()}
                lifecycle.atomic(cell / "request.json", request)
                response = worker.request(port, "completion", request)
                lifecycle.atomic(cell / "response.json", response)
                response_signature(response)
                require(
                    response["truncated"] is False
                    and response["tokens_evaluated"] == len(token_ids)
                    and 1 <= response["tokens_predicted"] <= 80,
                    "target response geometry/truncation differs",
                )
                for name in ("tokenization.json", "request.json", "response.json"):
                    worker.checked(lifecycle.pin(cell / name))
                requests[row["id"]] = request, response
        return requests
    finally:
        worker.reap(root / "release.json")


def prepare(args):
    require(not args.output.exists(), "expansion preparation already exists")
    storage_guard(C_ROOT, PLANNED_BYTES, FUTURE_STATE_BYTES)
    rows, pins = select_rows(args.current_roster, args.split, args.bindings, args.fresh)
    current = read(args.current_roster)
    # Inventory is evidence, never used to substitute for source verification.
    pins.append(lifecycle.pin(args.audit))
    pins.append(lifecycle.pin(Path(__file__)))
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / Path(__file__).name)
    for row in rows:
        text = render(row["messages"], current["rendering"])
        path = args.output / (row["id"] + "-prompt.txt")
        path.write_text(text, encoding="utf-8")
        row["rendered_text_path"] = str(path.resolve())
        row["rendered_prompt_sha256"] = lifecycle.pin(path)["sha256"]
    roster = {
        "schema": EXPANSION_SCHEMA,
        "status": "FROZEN_INPUTS_NO_LABELS",
        "rows": rows,
        "source_pins": pins,
        "rendering": current["rendering"],
        "protocol": protocol(),
        "parent_sha256": PARENT_SHA256,
        "target_sha256": TARGET_SHA256,
        "selection": {
            "seed": SEED,
            "groups": 24,
            "per_domain": 8,
            "rule": "SHA256(seed+NUL+group_id) ascending, historical split FIT TRAIN only; whole groups",
            "excluded": "Original36 group IDs and exact/NFKC whitespace-normalized identities; consumed fresh input identities",
            "outcome_based_selection_or_replacement": False,
            "native_geometry": "Freeze selection first; native prompt+80<=384 enforced at collect; failure aborts without replacement",
            "fresh_inputs_used_only_for_exclusion": True,
            "warning": "Historical TRAIN exposure remains; exact/normalized exclusion does not establish semantic disjointness",
        },
    }
    validate_roster(roster)
    lifecycle.atomic(args.output / "roster.json", roster)
    print(
        json.dumps(
            {
                "status": roster["status"],
                "roster": lifecycle.pin(args.output / "roster.json"),
                "groups": 24,
                "counts": dict(Counter(r["domain"] for r in rows)),
                "labels_generated": False,
                "native_backend_initialized": False,
            }
        ),
        flush=True,
    )


def main():
    cli = argparse.ArgumentParser(__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    for action in ("prepare", "collect"):
        parser = sub.add_parser(action)
        parser.add_argument("--current-roster", type=Path, default=ROSTER)
        parser.add_argument("--split", type=Path, default=SPLIT)
        parser.add_argument("--bindings", type=Path, nargs="+", default=BINDINGS)
        parser.add_argument("--fresh", type=Path, default=FRESH)
        parser.add_argument("--output", type=Path, required=True)
        if action == "prepare":
            parser.add_argument("--audit", type=Path, default=AUDIT)
        else:
            parser.add_argument("--roster", type=Path, required=True)
            parser.add_argument("--build", type=Path, default=TRACE_BUILD)
            parser.add_argument("--planned-bytes", type=int, default=PLANNED_BYTES)
            parser.add_argument("--future-state-bytes", type=int, default=FUTURE_STATE_BYTES)
            parser.add_argument("--allow-cuda", action="store_true")
    args = cli.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()

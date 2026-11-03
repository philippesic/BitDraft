"""Collect frozen modern target labels and bounded native teacher features.

The sole GPU owner must run this through the existing job supervisor with
--allow-cuda. This is phase-one teacher-data collection, not feature screening,
serving-verifier acceptance evidence, parent protection or throughput evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path("/home/philip/binary-eagle-decoding")
FLOOR = 2 * 1024**3
RESERVE = 64 * 1024**2
FEATURE_ROW_BYTES = 4 * 2560 * 4
LOGIT_ROW_BYTES = 248320 * 4
ANCESTRY_COMMAND = PROJECT / (
    "runs/qwen35-correctness-matrix-20261101-01/pilot-01/"
    "4B-Q4_K_M/rep-0-target_only-none-d0/command.json"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def atomic(path, value, *, replace=False):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=".collect-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
            temporary = None
        else:
            os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_identity(process):
    proc = Path("/proc") / str(process.pid)
    return {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ticks": int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19]),
        "cmdline": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
    }


def owns_live(process, identity):
    if process.poll() is not None:
        return False
    require(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot_id"],
        "boot identity changed",
    )
    proc = Path("/proc") / str(identity["pid"])
    require(
        proc.exists()
        and int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
        == identity["start_ticks"],
        "owned process birth identity differs",
    )
    return True


class StopRequested(Exception):
    pass


class Collector:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.started = time.perf_counter()
        self.stage = "authenticate"
        self.active = None
        self.active_identity = None
        self.cleaning = False
        self.created_output = False
        self.groups = []
        self.costs = []
        self.authenticated = {}

    def checked(self, ref):
        path = str(Path(ref["path"]).resolve(strict=True))
        if path not in self.authenticated:
            actual = pin(path)
            require(
                actual["sha256"] == ref["sha256"]
                and ("bytes" not in ref or actual["bytes"] == ref["bytes"]),
                "input pin changed: " + path,
            )
            self.authenticated[path] = actual
        else:
            require(self.authenticated[path]["sha256"] == ref["sha256"], "conflicting source pin")
        return Path(path)

    def permission(self):
        control = read(Path.home() / ".config/binary-eagle-decoding/gpu-control.json")["rtx5080"]
        require(
            not control.get("pause_requested")
            and not control.get("stop_all_requested")
            and not (self.output / "STOP").exists(),
            "GPU pause/stop requested",
        )

    def budget(self, remaining):
        free = shutil.disk_usage(self.output).free
        require(
            free >= FLOOR + RESERVE + remaining,
            f"insufficient storage: free={free}, floor={FLOOR}, remaining_arrays={remaining}",
        )
        return free

    def progress(self):
        atomic(
            self.output / "progress.json",
            {
                "stage": self.stage,
                "groups": self.groups,
                "costs": self.costs,
                "elapsed_seconds": time.perf_counter() - self.started,
                "active_process": self.active_identity,
            },
            replace=True,
        )

    def reap(self, receipt_path):
        if self.active is None:
            return
        process, identity = self.active, self.active_identity
        if owns_live(process, identity):
            os.killpg(identity["pgid"], signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            require(owns_live(process, identity), "cannot authenticate process before force stop")
            os.killpg(identity["pgid"], signal.SIGKILL)
            process.wait(timeout=10)
        try:
            os.killpg(identity["pgid"], 0)
            group_absent = False
        except ProcessLookupError:
            group_absent = True
        atomic(
            receipt_path,
            {
                "owned_process_reaped": process.poll() is not None,
                "owned_process_group_absent": group_absent,
                "returncode": process.returncode,
                "identity": identity,
            },
        )
        self.active = self.active_identity = None
        require(group_absent, "owned process group still has members after process reaping")

    def request(self, port, route, payload=None, *, timeout=120):
        self.permission()
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/{route}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def authenticate(self):
        require(self.args.allow_cuda, "sole GPU owner must explicitly pass --allow-cuda")
        require(not self.output.exists(), "collection output directory already exists")
        self.roster = read(self.args.roster)
        require(
            self.roster["schema"] == "w1a8_modern_phase1_generalization_roster_v2"
            and self.roster["status"] == "FROZEN_INPUTS_AND_BOUNDED_GEOMETRY_NO_LABELS",
            "frozen roster02 is required",
        )
        self.rows = self.roster["rows"]
        require(
            len(self.rows) == 36 and self.rows == self.roster["roster"],
            "exact 36-row frozen aliases required",
        )
        require(
            len({row["group_id"] for row in self.rows}) == 36
            and len({row["id"] for row in self.rows}) == 36
            and all(re.fullmatch("[A-Za-z0-9_-]+", row["id"]) for row in self.rows),
            "unique whole-group IDs required",
        )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.mkdir()
        self.created_output = True
        source = self.output / "source"
        source.mkdir()
        original_admission = Path(__file__).with_name("modern_admission.py")
        for original in (Path(__file__), original_admission):
            copied = source / original.name
            shutil.copyfile(original, copied)
            require(
                pin(original)["sha256"] == pin(copied)["sha256"], "source changed during snapshot"
            )
        self.driver_source = pin(source / Path(__file__).name)
        self.admission_source = pin(source / "modern_admission.py")
        spec = importlib.util.spec_from_file_location(
            "collection_native_admission", self.admission_source["path"]
        )
        self.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gate)
        self.baseline_ref = dict(self.gate.TRUSTED_BASELINE)
        self.baseline = read(self.checked(self.baseline_ref))
        target = next(
            row["target_pin"] for row in self.baseline["rows"] if row["target"] == "4B-Q4_K_M"
        )
        require(
            self.gate.same_pin(target, self.roster["target_pin"]),
            "roster target differs from frozen baseline",
        )
        self.target = target
        self.checked(target)
        self.build_ref = pin(self.args.build)
        self.build = read(self.args.build)
        self.tokenizer_build_ref = pin(self.roster["tokenizer_build"])
        self.tokenizer_build = read(self.roster["tokenizer_build"])
        frozen_ref = next(
            ref for ref in self.baseline["sources"] if ref["path"].endswith("/final-freeze.json")
        )
        frozen_ref = {**frozen_ref, "path": str(PROJECT / frozen_ref["path"])}
        self.frozen = read(self.checked(frozen_ref))
        self.trusted_libraries = {
            ref["path"]: ref for ref in self.frozen["pins"] if ".so" in ref["path"]
        }
        for build in (self.build, self.tokenizer_build):
            require(
                self.gate.same_pin(build["baseline"], self.baseline_ref)
                and self.gate.same_pin(build["cuda_overlay"], self.baseline["cuda"]),
                "capture/tokenizer build target runtime differs",
            )
            for name in ("source", "builder", "compiler", "executable"):
                self.checked(build[name])
            for ref in build["runtime_sources"] + build["included_local_headers"]:
                self.checked(ref)
            for ref in build["runtime_libraries"] + [build["cuda_overlay"]]:
                require(
                    ref["path"] in self.trusted_libraries
                    and self.gate.same_pin(ref, self.trusted_libraries[ref["path"]]),
                    "unfrozen helper library",
                )
                self.checked(ref)
        self.checked(self.frozen["server"])
        self.checked(pin(ANCESTRY_COMMAND))
        prior_freezes = []
        for ref in self.roster["prior_pins"]:
            value = read(self.checked(ref))
            if Path(ref["path"]).name.startswith("freeze"):
                prior_freezes.append(value)
        source_pins = {}
        for prior in prior_freezes:
            for ref in prior["pins"]:
                self.checked(ref)
                source_pins[str(Path(ref["path"]).resolve())] = ref
        extension_freeze = self.args.roster.with_name(
            self.args.roster.name.replace("roster", "freeze", 1)
        )
        extension = read(extension_freeze)
        roster_ref = pin(self.args.roster)
        require(
            any(self.gate.same_pin(roster_ref, ref) for ref in extension["pins"]),
            "extension freeze does not pin selected roster",
        )
        self.checked(pin(extension_freeze))
        require(
            extension["prior_pins"] == self.roster["prior_pins"],
            "roster extension ancestry differs from freeze",
        )
        for row in self.rows:
            require(
                row["role"] in ("fit", "development")
                and row["domain"] in ("code", "prose", "reasoning")
                and row["historical_split_role"] == row["role"]
                and row["historical_source_split"] == "TRAIN",
                "group role/ancestry differs",
            )
            binding_path = str(Path(row["binding_path"]).resolve())
            require(binding_path in source_pins, "raw binding is not frozen")
            raw = read(self.checked(source_pins[binding_path]))["rows"][row["binding_row"]]
            require(
                all(
                    raw[key] == row[key]
                    for key in ("group_id", "prompt_id", "domain", "messages", "source_ancestry")
                )
                and row["raw_messages"] == row["messages"],
                "raw whole-group message ancestry differs",
            )
            self.checked(row["original_receipt"])
            for ref in row["tokenizer_pins"].values():
                self.checked(ref)
            text = Path(row["rendered_text_path"])
            require(
                hashlib.sha256(text.read_bytes()).hexdigest() == row["rendered_prompt_sha256"],
                "rendered original prompt changed",
            )
            self.checked(pin(text))
            token_ids = self.gate.token_ids(Path(row["tokenizer_pins"]["tokens"]["path"]))
            require(
                len(token_ids) == row["native_prompt_tokens"] and len(token_ids) + 80 <= 384,
                "frozen prompt exceeds untruncated total-row budget",
            )
            self.expected_window_hash(row)
        for role, expected in (("fit", 8), ("development", 4)):
            for domain in ("code", "prose", "reasoning"):
                require(
                    sum(row["role"] == role and row["domain"] == domain for row in self.rows)
                    == expected,
                    "frozen whole-group balance differs",
                )
        self.permission()
        self.budget(sum(self.estimate(row, 80) for row in self.rows))
        clean = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("GGML_", "DSPARK_", "DFLASH_", "EAGLE_", "W1AX_", "LLAMA_ARG_"))
            and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "NVIDIA_TF32_OVERRIDE")
        }
        self.environment = {
            **clean,
            **self.build["environment"],
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "4",
        }
        atomic(
            self.output / "freeze.json",
            {
                "schema": "w1a8_modern_collection_v1",
                "roster": roster_ref,
                "roster_extension_freeze": pin(extension_freeze),
                "build": self.build_ref,
                "tokenizer_build": self.tokenizer_build_ref,
                "baseline": self.baseline_ref,
                "target": self.target,
                "driver_source": self.driver_source,
                "admission_source": self.admission_source,
                "pins": list(self.authenticated.values()),
                "maximum_capture_bytes": sum(self.estimate(row, 80) for row in self.rows),
                "storage_floor_bytes": FLOOR,
                "reserve_bytes": RESERVE,
                "protocol": (
                    "one frozen target server, all36 requests, reap before fixed-continuation "
                    "captures; no replacements"
                ),
                "scope": (
                    "Phase-one conditional teacher data; no serving-verifier/parent-protection, "
                    "screening or performance proof"
                ),
                "training_updates": 0,
            },
        )
        self.costs.append(
            {
                "stage": "source_roster_and_native_dependency_authentication",
                "wall_seconds": time.perf_counter() - self.started,
            }
        )

    def expected_window_hash(self, row):
        digest = hashlib.sha256(
            (self.roster["seed"] + "\0" + row["group_id"] + "\0serving-window-v1").encode()
        ).hexdigest()[:16]
        require(
            row["window_sampling"]["u64_hex"] == digest
            and row["window_sampling"]["teacher_rows"] == 17,
            "predeclared window seed differs",
        )
        return int(digest, 16)

    @staticmethod
    def estimate(row, length):
        return (
            0
            if length < 2
            else (
                (row["native_prompt_tokens"] + length - 1) * FEATURE_ROW_BYTES
                + min(17, length) * LOGIT_ROW_BYTES
            )
        )

    def serving(self):
        self.stage = "frozen_target_server"
        root = self.output / "serving"
        root.mkdir()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        argv = [
            self.frozen["server"]["path"],
            "-m",
            self.target["path"],
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
        atomic(
            root / "freeze.json",
            {
                "source": self.driver_source,
                "server": self.frozen["server"],
                "target": self.target,
                "cuda": self.baseline["cuda"],
                "command_ancestry": pin(ANCESTRY_COMMAND),
                "command": argv,
                "environment": self.build["environment"],
                "scope": (
                    "Frozen target-only label collection; no throughput or actual "
                    "speculative-verifier proof"
                ),
            },
        )
        stage_started = time.perf_counter()
        try:
            self.permission()
            self.budget(sum(self.estimate(row, 80) for row in self.rows))
            with (root / "server.log").open("x") as log:
                self.active = subprocess.Popen(
                    argv,
                    env=self.environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.active_identity = process_identity(self.active)
                atomic(root / "identity.json", self.active_identity)
                self.progress()
                for _ in range(180):
                    require(self.active.poll() is None, "target server exited during startup")
                    try:
                        self.request(port, "health", timeout=1)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.25)
                else:
                    raise TimeoutError("target server did not become healthy")
                maps = (Path("/proc") / str(self.active.pid) / "maps").read_text()
                (root / "loaded-library-maps.txt").write_text(maps)
                self.gate.runtime_maps(
                    self.gate.Authentication(),
                    pin(root / "loaded-library-maps.txt"),
                    self.trusted_libraries,
                    required=[ref["path"] for ref in self.build["runtime_libraries"]]
                    + [self.baseline["cuda"]["path"]],
                )
                atomic(root / "properties.json", self.request(port, "props"))
                for row in self.rows:
                    self.permission()
                    self.budget(sum(self.estimate(item, 80) for item in self.rows))
                    group = root / row["id"]
                    group.mkdir()
                    raw = Path(row["rendered_text_path"]).read_bytes().decode("utf-8")
                    request = {
                        "prompt": raw,
                        "n_predict": 80,
                        "temperature": 0,
                        "seed": 1234,
                        "cache_prompt": False,
                        "return_tokens": True,
                        "stream": False,
                    }
                    atomic(group / "request.json", request)
                    began = time.perf_counter()
                    try:
                        response = self.request(port, "completion", request)
                        atomic(group / "response.json", response)
                        tokens = response.get("tokens")
                        valid = (
                            isinstance(tokens, list)
                            and all(type(t) is int and 0 <= t < 248320 for t in tokens)
                            and len(tokens) == response.get("tokens_predicted")
                            and len(tokens) <= 80
                        )
                        reason = (
                            "invalid_returned_tokens"
                            if not valid
                            else "truncated_source"
                            if response.get("truncated") is not False
                            else "short_continuation"
                            if len(tokens) < 2
                            else None
                        )
                        info = {
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "role": row["role"],
                            "domain": row["domain"],
                            "status": "SERVING_LABELS_RETAINED"
                            if reason is None
                            else "EXCLUDED_RETAINED",
                            "exclusion_reason": reason,
                            "actual_committed_tokens": len(tokens) if valid else None,
                            "stop_type": response.get("stop_type"),
                            "stopping_word": response.get("stopping_word"),
                            "truncated": response.get("truncated"),
                            "request": pin(group / "request.json"),
                            "response": pin(group / "response.json"),
                            "request_wall_seconds": time.perf_counter() - began,
                        }
                    except urllib.error.HTTPError as error:
                        (group / "http-error.body").write_bytes(error.read())
                        info = {
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "role": row["role"],
                            "domain": row["domain"],
                            "status": "EXCLUDED_RETAINED",
                            "exclusion_reason": "HTTP_error",
                            "error": repr(error),
                            "actual_committed_tokens": None,
                            "request_wall_seconds": time.perf_counter() - began,
                        }
                    atomic(group / "collection.json", info)
                    self.groups.append(info)
                    self.progress()
                    print(
                        json.dumps(
                            {
                                "stage": "serving",
                                "group": row["id"],
                                "status": info["status"],
                                "tokens": info["actual_committed_tokens"],
                            }
                        ),
                        flush=True,
                    )
        finally:
            self.reap(root / "release.json")
            self.costs.append(
                {
                    "stage": "all_target_requests_and_server_lifecycle",
                    "wall_seconds": time.perf_counter() - stage_started,
                }
            )
            self.progress()

    def captures(self):
        require(self.active is None, "server must be reaped before any native capture")
        self.stage = "fixed_native_captures"
        plans = [
            (row, info)
            for row, info in zip(self.rows, self.groups)
            if info["status"] == "SERVING_LABELS_RETAINED"
        ]
        remaining = sum(self.estimate(row, info["actual_committed_tokens"]) for row, info in plans)
        failures = 0
        for row, info in plans:
            self.permission()
            self.budget(remaining)
            stage_started = time.perf_counter()
            group = self.output / row["id"]
            group.mkdir()
            predicted = self.estimate(row, info["actual_committed_tokens"])
            response = read(info["response"]["path"])
            labels = response["tokens"]
            length, count = len(labels), min(17, len(labels))
            offset = self.expected_window_hash(row) % (length - count + 1)
            positions = list(
                range(
                    row["native_prompt_tokens"] - 1 + offset,
                    row["native_prompt_tokens"] - 1 + offset + count,
                )
            )
            fixed = group / "continuation.ids"
            fixed.write_text("\n".join(map(str, labels[:-1])) + "\n")
            out = group / "capture"
            argv = [
                self.build["executable"]["path"],
                "--model",
                self.target["path"],
                "--tokens",
                row["tokenizer_pins"]["tokens"]["path"],
                "--out",
                str(out),
                *self.build["required_backend_argv"],
                "--decode-tokens",
                str(fixed),
                "--prefill-chunk",
                "512",
                "--decode-chunk",
                "1",
                "--output-cap",
                "17",
                "--threads",
                "4",
                "--taps",
                "0,2,16,29",
                "--logit-positions",
                ",".join(map(str, positions)),
                "--max-output-mib",
                "128",
            ]
            atomic(
                group / "capture-command.json",
                {"argv": argv, "environment": self.build["environment"]},
            )
            status = None
            try:
                with (group / "capture.log").open("x") as log:
                    self.active = subprocess.Popen(
                        argv,
                        env=self.environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self.active_identity = process_identity(self.active)
                    atomic(group / "identity.json", self.active_identity)
                    self.progress()
                    code = self.active.wait(timeout=240)
                    require(code == 0, f"native capture exited {code}")
                self.reap(group / "release.json")
                meta = read(out / "capture.json")
                require(
                    meta["logit_positions"] == positions
                    and meta["logits_shape"] == [count, 248320]
                    and meta["feature_shape"] == [row["native_prompt_tokens"] + length - 1, 2560]
                    and meta["array_bytes"] == predicted,
                    "captured selected window/geometry differs",
                )
                packet = {
                    "schema": "w1a8_modern_capture_packet_v1",
                    "baseline": self.baseline_ref,
                    "target": self.target,
                    "build": self.build_ref,
                    "capture": pin(out / "capture.json"),
                    "capture_command": pin(group / "capture-command.json"),
                    "capture_library_maps": pin(out / "loaded-library-maps.txt"),
                    "continuation_tokens": pin(fixed),
                    "arrays": {
                        name: pin(out / name)
                        for name in (
                            "logits.f32",
                            "layer-0.f32",
                            "layer-2.f32",
                            "layer-16.f32",
                            "layer-29.f32",
                        )
                    },
                    "tokenizer": {
                        "build": self.tokenizer_build_ref,
                        **row["tokenizer_pins"],
                        "text": pin(row["rendered_text_path"]),
                    },
                    "serving": {
                        "freeze": pin(self.output / "serving/freeze.json"),
                        "request": info["request"],
                        "response": info["response"],
                        "library_maps": pin(self.output / "serving/loaded-library-maps.txt"),
                    },
                }
                atomic(group / "packet.json", packet)
                admission = group / "admission.json"
                admission_started = time.perf_counter()
                self.gate.prepare(group / "packet.json", admission)
                receipt = read(admission)
                steps = self.gate.build_actor_steps(receipt)
                eligible = [step["actor_step_id"] for step in steps if step["supervision_eligible"]]
                self.gate.require_actor_steps(receipt, eligible)
                self.gate.require_actor_steps(receipt, eligible, purpose="protection_numeric")
                status = {
                    "status": "CAPTURE_AND_NUMERIC_ADMISSION_COMPLETE",
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "role": row["role"],
                    "domain": row["domain"],
                    "actual_committed_tokens": length,
                    "teacher_window_count": count,
                    "teacher_window_offset": offset,
                    "teacher_logit_positions": positions,
                    "window_sampling": row["window_sampling"],
                    "eligible_teacher_rows": receipt["eligible_row_ids"],
                    "excluded_teacher_rows": receipt["excluded_row_ids"],
                    "actor_steps": steps,
                    "eligible_actor_steps": eligible,
                    "admission": pin(admission),
                    "parent_success_protection_proven": False,
                    "actual_serving_verifier_proven": False,
                    "admission_wall_seconds": time.perf_counter() - admission_started,
                }
                failures = 0
            except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                self.reap(group / "release.json")
                failures += 1
                status = {
                    "status": "CAPTURE_OR_ADMISSION_FAILED_RETAINED",
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "role": row["role"],
                    "domain": row["domain"],
                    "error": repr(error),
                    "actual_committed_tokens": length,
                    "teacher_window_count": count,
                    "teacher_window_offset": offset,
                    "parent_success_protection_proven": False,
                }
            finally:
                self.reap(group / "release.json")
            remaining -= predicted
            status["stage_wall_seconds"] = time.perf_counter() - stage_started
            self.costs.append(
                {
                    "stage": "capture_and_admission",
                    "group": row["id"],
                    "wall_seconds": status["stage_wall_seconds"],
                    "status": status["status"],
                }
            )
            atomic(group / "result.json", status)
            info["native_collection"] = status
            self.progress()
            print(
                json.dumps(
                    {
                        "stage": "capture",
                        "group": row["id"],
                        "status": status["status"],
                        "eligible_actor_steps": len(status.get("eligible_actor_steps", [])),
                    }
                ),
                flush=True,
            )
            require(
                failures < 3,
                "three consecutive capture/admission failures; preserved evidence needs inspection",
            )

    def run(self):
        self.authenticate()
        self.serving()
        self.captures()
        self.stage = "complete"
        summary = {
            "status": "COLLECTION_COMPLETE_WITH_ACCOUNTED_EXCLUSIONS",
            "groups": self.groups,
            "costs": self.costs,
            "wall_seconds": time.perf_counter() - self.started,
            "source": self.driver_source,
            "admission_source": self.admission_source,
            "roster": pin(self.args.roster),
            "free_bytes_after": shutil.disk_usage(self.output).free,
            "training_updates": 0,
            "parent_success_protection_proven": False,
            "actual_serving_verifier_proven": False,
            "scope": (
                "Conditional native teacher data only; no accepted-parent, screening, "
                "quality or throughput qualification"
            ),
        }
        atomic(self.output / "summary.json", summary)
        self.progress()
        return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    collector = Collector(args)

    def stopped(signum, frame):
        if not collector.cleaning:
            raise StopRequested(f"received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stopped)
    try:
        result = collector.run()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "groups": len(result["groups"]),
                    "wall_seconds": result["wall_seconds"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        collector.cleaning = True
        if collector.created_output:
            collector.reap(collector.output / "interrupted-release.json")
            atomic(
                collector.output / "failure.json",
                {
                    "stage": collector.stage,
                    "error": repr(error),
                    "groups": collector.groups,
                    "costs": collector.costs,
                    "wall_seconds": time.perf_counter() - collector.started,
                },
            )
        raise
    finally:
        collector.cleaning = True
        if collector.active is not None:
            collector.reap(collector.output / "final-release.json")


if __name__ == "__main__":
    main()

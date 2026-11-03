"""Five-repeat modern native development requests; sole GPU operator only.

--models JSON: {baseline:pin, target:pin, source_pins:[pin], arms:[{id,
kind:target|dflash|w1a8|eagle_q4, model:pin|null, runtime:{server:pin, libraries:[pin],
environment:{...}, source_pins:[pin]}}]}. Include every executable loaded image
(including system/CUDA libraries) in runtime.libraries. Supply existing explicit
CUDA_CACHE_PATH and LD_LIBRARY_PATH per runtime; caches are shared, never copied.
Four to eight arms must include one frozen target and one frozen DFlash. The
default W1A8 runtime is original. Its sole optional variant is
runtime.variant="zero_output_headskip", with runtime.variant_build pin to the
approved headskip build.json (also include that pin in runtime.source_pins).

--inputs JSON: {roster:pin, source_pins:[pin], rows:[{id, request:pin,
reference:pin}], warmups:[two entries with the same fields]}. rows must be all
12 development roster groups. Use the existing conditional target requests and
responses unchanged. A pin has path, sha256 and optionally bytes.

This 80-token pilot measures native correctness and complete-request throughput
on reused development inputs. It makes no untouched quality/predictor claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

from modern_collect import Collector, atomic, pin, process_identity, read, require

PROJECT = Path("/home/philip/binary-eagle-decoding")
TARGET_SHA = "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"
INITIAL_SHA = "6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048"
BASELINE = PROJECT / "experiments/w1a8-standing-20261102/baseline.json"
NATIVE = PROJECT / "runs/w1a8-modern-native-admission-20261103-01/freeze.json"
FROZEN = PROJECT / "runs/qwen35-correctness-matrix-20261101-01/final-freeze.json"
HEADSKIP = PROJECT / "runs/w1a8-modern-headskip-20261103-01/build.json"
HEADSKIP_LIBRARY_SHA = "3c55f4f0b63104b9859184fffda1eee8fa2d87ef45c0e791677f2ab076247e97"
PAYLOAD_KEYS = {
    "prompt",
    "n_predict",
    "temperature",
    "seed",
    "cache_prompt",
    "return_tokens",
    "stream",
}


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def signature(response):
    require(
        isinstance(response.get("tokens"), list)
        and all(type(t) is int and t >= 0 for t in response["tokens"]),
        "raw token IDs missing",
    )
    require(
        isinstance(response.get("content"), str)
        and isinstance(response.get("stop_type"), str)
        and isinstance(response.get("stopping_word"), str)
        and type(response.get("truncated")) is bool,
        "raw text/stop/truncation missing",
    )
    return {
        k: response[k] for k in ["tokens", "content", "stop_type", "stopping_word", "truncated"]
    }


class Requests(Collector):
    def guard(self):
        self.permission()
        require(not self.stopping, "signal stop requested")
        for path in [self.output, Path("/mnt/c")]:
            require(
                shutil.disk_usage(path).free >= 2 * 1024**3 + 64 * 1024**2,
                "WSL/C storage floor: " + str(path),
            )

    def authenticate(self):
        require(self.args.allow_cuda, "sole GPU owner must pass --allow-cuda")
        require(self.args.repetitions == 5, "five matched repetitions required")
        self.output.mkdir(parents=True, exist_ok=False)
        self.created_output = True
        self.models = read(self.args.models)
        self.inputs = read(self.args.inputs)
        for ref in references([self.models, self.inputs]):
            self.checked(ref)
        self.baseline = read(self.checked(self.models["baseline"]))
        require(
            pin(BASELINE)["sha256"] == self.models["baseline"]["sha256"],
            "standing baseline differs",
        )
        row = next(x for x in self.baseline["rows"] if x["target"] == "4B-Q4_K_M")
        require(
            self.models["target"]["sha256"] == TARGET_SHA == row["target_pin"]["sha256"],
            "target must be frozen 4B Q4_K_M",
        )
        native = read(NATIVE)
        frozen = read(FROZEN)
        original_build = read(self.checked(native["candidate_build"]))
        candidate_server = next(
            x for x in original_build["binaries"] if Path(x["path"]).name == "llama-server"
        )
        self.arms = self.models["arms"]
        require(
            4 <= len(self.arms) <= 8 and len({a["id"] for a in self.arms}) == len(self.arms),
            "four to eight unique arms required",
        )
        require(
            sum(a["kind"] == "target" for a in self.arms) == 1
            and sum(a["kind"] == "dflash" for a in self.arms) == 1
            and all(a["kind"] in ("target", "dflash", "w1a8", "eagle_q4") for a in self.arms),
            "one frozen target,one frozen DFlash,and W1A8 candidate arms required",
        )
        self.variant_sources = []
        for arm in self.arms:
            require(arm["id"].replace("-", "").replace("_", "").isalnum(), "unsafe arm ID")
            runtime = arm["runtime"]
            env = runtime["environment"]
            require(runtime.get("source_pins"), "runtime immutable source pins required")
            require(
                env.get("GGML_CUDA_DECODE_CONSISTENT") == "32"
                and env.get("CUDA_VISIBLE_DEVICES") == "0"
                and env.get("OMP_NUM_THREADS") == "4",
                "explicit correctness CUDA/thread environment required",
            )
            require(
                env.get("LD_LIBRARY_PATH")
                and env.get("CUDA_CACHE_PATH")
                and Path(env["CUDA_CACHE_PATH"]).is_dir(),
                "existing runtime library/cache paths required",
            )
            require(
                not env.get("LD_PRELOAD")
                and not any(
                    k.startswith(("W1A8_", "LLAMA_", "DSPARK_", "DFLASH_", "EAGLE_")) for k in env
                ),
                "trace/algorithm environment overrides forbidden",
            )
            permitted = {
                "LD_LIBRARY_PATH",
                "CUDA_CACHE_PATH",
                "CUDA_VISIBLE_DEVICES",
                "GGML_CUDA_DECODE_CONSISTENT",
                "GGML_W1AX_ACT_BITS",
                "OMP_NUM_THREADS",
                "CUDA_CACHE_MAXSIZE",
                "CUDA_MODULE_LOADING",
            }
            require(set(env) <= permitted, "unrecognized runtime environment key")
            candidate_runtime = arm["kind"] in ("w1a8", "eagle_q4")
            expected_server = candidate_server if candidate_runtime else frozen["server"]
            require(
                runtime["server"]["sha256"] == expected_server["sha256"],
                "frozen/original candidate server differs",
            )
            original_libraries = original_build["binaries"] if candidate_runtime else frozen["pins"]
            native_library_hashes = {
                Path(r["path"]).resolve().name: r["sha256"]
                for r in original_libraries
                if ".so" in Path(r["path"]).name
            }
            variant = runtime.get("variant", "original")
            require(variant in ("original", "zero_output_headskip"), "unknown runtime variant")
            if variant == "zero_output_headskip":
                require(candidate_runtime, "headskip variant requires an EAGLE candidate")
                build_ref = runtime.get("variant_build")
                require(isinstance(build_ref, dict), "headskip variant_build pin required")
                require(
                    self.checked(build_ref).resolve() == HEADSKIP.resolve()
                    and build_ref["sha256"] == pin(HEADSKIP)["sha256"],
                    "only root-approved headskip build is permitted",
                )
                require(
                    any(
                        Path(ref["path"]).resolve() == HEADSKIP.resolve()
                        and ref["sha256"] == build_ref["sha256"]
                        for ref in runtime["source_pins"]
                    ),
                    "headskip variant build must be included in runtime source_pins",
                )
                variant_build = read(self.checked(build_ref))
                for ref in references(variant_build):
                    self.checked(ref)
                require(
                    variant_build["library"]["sha256"] == HEADSKIP_LIBRARY_SHA
                    and Path(variant_build["library"]["path"]).resolve().name
                    == "libllama.so.0.8.0",
                    "approved headskip core hash differs",
                )
                native_library_hashes["libllama.so.0.8.0"] = HEADSKIP_LIBRARY_SHA
                self.variant_sources.extend(
                    variant_build[key] for key in ("source", "unity", "original", "patch")
                )
            else:
                require(
                    "variant_build" not in runtime,
                    "original runtime cannot substitute a variant build",
                )
            for library in runtime["libraries"]:
                name = Path(library["path"]).resolve().name
                if name.startswith(("libllama", "libggml", "libmtmd")):
                    require(
                        name in native_library_hashes
                        and library["sha256"] == native_library_hashes[name],
                        "original native library differs: " + name,
                    )
            expected_cuda = (
                next(
                    x
                    for x in original_build["binaries"]
                    if "libggml-cuda.so" in Path(x["path"]).name
                )
                if candidate_runtime
                else self.baseline["cuda"]
            )
            cuda = [x for x in runtime["libraries"] if "libggml-cuda.so" in Path(x["path"]).name]
            require(
                len(cuda) == 1 and cuda[0]["sha256"] == expected_cuda["sha256"],
                "frozen/original candidate CUDA differs",
            )
            if arm["kind"] == "target":
                require(arm.get("model") is None, "target arm cannot have drafter")
            elif arm["kind"] == "dflash":
                require(
                    arm["model"]["sha256"] == row["draft_pin"]["sha256"],
                    "primary DFlash Q4_K_M differs",
                )
            elif arm["kind"] == "eagle_q4":
                require(
                    arm["model"]["sha256"] == native["ordinary_eagle"]["sha256"],
                    "ordinary EAGLE Q4_0 diagnostic differs",
                )
            else:
                require(env.get("GGML_W1AX_ACT_BITS") == "8", "W1A8 activation mode required")
            if arm["kind"] != "w1a8":
                require("GGML_W1AX_ACT_BITS" not in env, "W1A8 mode leaked into baseline")
        initializers = [
            a for a in self.arms if a.get("model") and a["model"]["sha256"] == INITIAL_SHA
        ]
        require(
            len(initializers) in (1, 2),
            "one initializer or matched original/headskip initializer pair required",
        )
        if len(initializers) == 2:
            require(
                {a["runtime"].get("variant", "original") for a in initializers}
                == {"original", "zero_output_headskip"},
                "two initializer arms must isolate original versus headskip runtime",
            )
        self.authenticate_inputs()
        self.guard()
        hardware = subprocess.check_output(
            [
                "/usr/lib/wsl/lib/nvidia-smi",
                "--id=0",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        )
        require(
            hardware.split(",")[0].strip() == "NVIDIA GeForce RTX 5080", "RTX5080 GPU0 required"
        )
        (self.output / "hardware.txt").write_text(hardware)
        sources = self.output / "source"
        sources.mkdir()
        snapshot_refs = [pin(__file__), pin(Path(__file__).with_name("modern_collect.py"))]
        snapshot_refs += self.models.get("source_pins", []) + self.inputs.get("source_pins", [])
        snapshot_refs += self.variant_sources + self.input_source_refs
        for arm in self.arms:
            snapshot_refs += arm["runtime"]["source_pins"]
        snapshots = []
        for i, ref in enumerate(snapshot_refs):
            source = self.checked(ref)
            require(source.stat().st_size < 32 * 1024**2, "source snapshot unexpectedly large")
            dest = sources / f"{i:03d}-{source.name}"
            shutil.copyfile(source, dest)
            snapshots.append({"original": ref, "snapshot": pin(dest)})
        atomic(self.output / "models.json", self.models)
        atomic(self.output / "inputs.json", self.inputs)
        atomic(
            self.output / "freeze.json",
            {
                "schema": self.request_schema,
                "models": pin(self.args.models),
                "inputs": pin(self.args.inputs),
                "baseline": self.models["baseline"],
                "original_native_admission": pin(NATIVE),
                "original_frozen_runtime": pin(FROZEN),
                "hardware": pin(self.output / "hardware.txt"),
                "source_snapshots": snapshots,
                "repetitions": self.args.repetitions,
                "expected_startups": len(self.arms) * self.args.repetitions,
                "expected_measured_requests": len(self.rows)
                * len(self.arms)
                * self.args.repetitions,
                "expected_excluded_warmups": len(self.warmups)
                * len(self.arms)
                * self.args.repetitions,
                "runtime_variants": [
                    {
                        "arm": arm["id"],
                        "variant": arm["runtime"].get("variant", "original"),
                        "variant_build": arm["runtime"].get("variant_build"),
                    }
                    for arm in self.arms
                ],
                "arm_order": "forward on even repetition,reverse on odd",
                "warmups_per_startup": 2,
                "output_limit": self.output_limit,
                "metric": "returned token IDs / summed full HTTP wall, startup and warmups excluded",
                "scope": self.request_scope,
            },
        )

    def authenticate_inputs(self):
        self.output_limit = 80
        self.request_schema = "w1a8_modern_development_requests_v1"
        self.request_scope = "reused development pilot; no untouched quality, quick predictor, backlog or optimum claim"
        self.input_source_refs = []
        roster = read(self.checked(self.inputs["roster"]))
        roster_rows = {r["id"]: r for r in roster["rows"]}
        require(
            roster.get("schema") == "w1a8_modern_phase1_generalization_roster_v2"
            and len(roster_rows) == 36,
            "frozen36 roster02 required",
        )
        self.rows = self.inputs["rows"]
        self.warmups = self.inputs["warmups"]
        require(
            len(self.rows) == 12
            and {r["id"] for r in self.rows}
            == {r["id"] for r in roster["rows"] if r["role"] == "development"},
            "all12 development groups required",
        )
        require(
            len(self.warmups) == 2 and len({r["id"] for r in self.warmups}) == 2,
            "two unique excluded warmup prompts required",
        )
        self.payloads = {}
        self.gold = {}
        for item in self.rows + self.warmups:
            require(item["id"] in roster_rows, "input not in frozen roster")
            payload = read(self.checked(item["request"]))
            gold = read(self.checked(item["reference"]))
            require(
                set(payload) == PAYLOAD_KEYS
                and payload
                == {
                    "prompt": payload["prompt"],
                    "n_predict": 80,
                    "temperature": 0,
                    "seed": 1234,
                    "cache_prompt": False,
                    "return_tokens": True,
                    "stream": False,
                },
                "request settings differ",
            )
            require(
                payload["prompt"]
                == Path(roster_rows[item["id"]]["rendered_text_path"]).read_text(),
                "raw prompt differs from roster",
            )
            self.payloads[item["id"]] = payload
            self.gold[item["id"]] = signature(gold)

    def prepare_cell(self, port, cell, arm, rep):
        """Optional non-timing setup for explicit input adapters."""

    def audit_maps(self, cell, tag, arm):
        maps = (Path("/proc") / str(self.active.pid) / "maps").read_text()
        path = cell / f"loaded-library-maps-{tag}.txt"
        path.write_text(maps)
        expected = {
            str(Path(r["path"]).resolve()): r
            for r in [arm["runtime"]["server"]] + arm["runtime"]["libraries"]
        }
        images = set()
        for line in maps.splitlines():
            parts = line.split(maxsplit=5)
            if (
                len(parts) == 6
                and parts[5].startswith("/")
                and ("x" in parts[1] or ".so" in parts[5])
            ):
                require(not parts[5].endswith(" (deleted)"), "deleted loaded image")
                images.add(str(Path(parts[5]).resolve()))
        require(
            str(Path(arm["runtime"]["server"]["path"]).resolve()) in images,
            "server not in loaded images",
        )
        audit = []
        for image in sorted(images):
            require(image in expected, "unfrozen loaded image: " + image)
            self.checked(expected[image])
            audit.append(self.authenticated[image])
        require(any("libggml-cuda.so" in Path(p).name for p in images), "CUDA backend not loaded")
        atomic(cell / f"loaded-images-{tag}.json", {"maps": pin(path), "images": audit})

    def completion(self, port, item, cell, index, warmup, arm, rep):
        self.guard()
        payload = self.payloads[item["id"]]
        prefix = f"{index:02d}-{item['id']}"
        request_path = cell / (prefix + "-request.json")
        response_path = cell / (prefix + "-response.json")
        atomic(request_path, payload)
        begin = time.perf_counter()
        response = self.request(port, "completion", payload, timeout=180)
        seconds = time.perf_counter() - begin
        atomic(response_path, response)
        require(
            signature(response) == self.gold[item["id"]],
            "token/text/stop parity failed: " + arm["id"] + "/" + item["id"],
        )
        timing = response.get("timings", {})
        require(math.isfinite(seconds) and seconds > 0, "invalid request wall")
        result = {
            "arm": arm["id"],
            "runtime_variant": arm["runtime"].get("variant", "original"),
            "repetition": rep,
            "id": item["id"],
            "warmup": warmup,
            "request": pin(request_path),
            "response": pin(response_path),
            "full_http_wall_seconds": seconds,
            "completion_tokens": len(response["tokens"]),
            "timings": timing,
            "draft_proposed": timing.get("draft_n"),
            "draft_accepted": timing.get("draft_n_accepted"),
            "token_text_stop_parity": True,
        }
        atomic(cell / (prefix + "-receipt.json"), result)
        self.records.append(result)
        return result

    def cell(self, rep, arm):
        self.guard()
        cell = self.output / f"rep-{rep:02d}-{arm['id']}"
        cell.mkdir()
        begin = time.perf_counter()
        runtime = arm["runtime"]
        process = None
        failure = None
        clean = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(
                (
                    "GGML_",
                    "DSPARK_",
                    "DFLASH_",
                    "EAGLE_",
                    "W1AX_",
                    "W1A8_",
                    "LLAMA_",
                    "CUDA_",
                    "LD_",
                    "OMP_",
                )
            )
            and k != "NVIDIA_TF32_OVERRIDE"
        }
        env = {**clean, **runtime["environment"]}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        spec = {
            "target": "none",
            "dflash": "draft-dflash",
            "w1a8": "draft-eagle3",
            "eagle_q4": "draft-eagle3",
        }[arm["kind"]]
        argv = [
            runtime["server"]["path"],
            "-m",
            self.models["target"]["path"],
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
            spec,
        ]
        if arm.get("model"):
            argv += [
                "-md",
                arm["model"]["path"],
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
            ]
        if arm["kind"] == "w1a8":
            argv += ["-lv", "4"]
        atomic(cell / "command.json", {"argv": argv, "environment": runtime["environment"]})
        try:
            with (cell / "server.log").open("x") as log:
                process = subprocess.Popen(
                    argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                self.active = process
                self.active_identity = process_identity(process)
                atomic(cell / "identity.json", self.active_identity)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    self.guard()
                    require(process.poll() is None, "server exited during startup")
                    try:
                        self.request(port, "health", timeout=1)
                        break
                    except Exception:
                        self.guard()
                        time.sleep(0.2)
                else:
                    raise RuntimeError("server startup timeout")
                atomic(cell / "startup.json", {"seconds": time.perf_counter() - begin})
                self.prepare_cell(port, cell, arm, rep)
                for index, item in enumerate(self.warmups):
                    self.completion(port, item, cell, index, True, arm, rep)
                self.audit_maps(cell, "warm", arm)
                for index, item in enumerate(self.rows, start=2):
                    self.completion(port, item, cell, index, False, arm, rep)
                self.audit_maps(cell, "final", arm)
                if arm["kind"] == "w1a8":
                    content = (cell / "server.log").read_text()
                    require(
                        "complete nine-linear fixed row W1A8" in content
                        and content.count("W1A8 packed ") == 9,
                        "packed nine-linear loader receipts missing",
                    )
        except BaseException as error:
            failure = repr(error)
            atomic(
                cell / "failure.json",
                {"error": failure, "elapsed_seconds": time.perf_counter() - begin},
            )
            raise
        finally:
            if process is not None:
                self.reap(cell / "release.json")
            atomic(
                cell / "cost.json",
                {"elapsed_seconds": time.perf_counter() - begin, "failure": failure},
            )
        print(
            json.dumps(
                {
                    "stage": "cell_complete",
                    "repetition": rep,
                    "arm": arm["id"],
                    "measured_requests": len(self.rows),
                }
            ),
            flush=True,
        )

    def run(self):
        self.records = []
        self.stopping = False

        def stop(signum, frame):
            self.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            self.authenticate()
            for rep in range(self.args.repetitions):
                for arm in self.arms if rep % 2 == 0 else list(reversed(self.arms)):
                    self.cell(rep, arm)
            summaries = []
            for arm in self.arms:
                reps = []
                for rep in range(self.args.repetitions):
                    rows = [
                        r
                        for r in self.records
                        if r["arm"] == arm["id"] and r["repetition"] == rep and not r["warmup"]
                    ]
                    require(len(rows) == len(self.rows), "incomplete repetition")
                    tokens = sum(r["completion_tokens"] for r in rows)
                    wall = sum(r["full_http_wall_seconds"] for r in rows)
                    reps.append(
                        {
                            "repetition": rep,
                            "tokens": tokens,
                            "full_http_wall_seconds": wall,
                            "tokens_per_second": tokens / wall,
                            "draft_proposed": sum(r["draft_proposed"] or 0 for r in rows),
                            "draft_accepted": sum(r["draft_accepted"] or 0 for r in rows),
                        }
                    )
                summaries.append(
                    {
                        "arm": arm["id"],
                        "kind": arm["kind"],
                        "runtime_variant": arm["runtime"].get("variant", "original"),
                        "variant_build": arm["runtime"].get("variant_build"),
                        "repetitions": reps,
                        "tokens_per_second": sum(r["tokens"] for r in reps)
                        / sum(r["full_http_wall_seconds"] for r in reps),
                    }
                )
            baseline = next(s for s in summaries if s["kind"] == "dflash")
            for summary in summaries:
                ratios = [
                    r["tokens_per_second"] / b["tokens_per_second"]
                    for r, b in zip(summary["repetitions"], baseline["repetitions"])
                ]
                summary["matched_DFlash_speedup_repetitions"] = ratios
                summary["median_matched_DFlash_speedup"] = statistics.median(ratios)
            atomic(
                self.output / "result.json",
                {
                    "status": "DEVELOPMENT_NATIVE_REQUESTS_PASS",
                    "token_text_stop_parity": True,
                    "measured_requests": sum(not row["warmup"] for row in self.records),
                    "excluded_warmups": sum(row["warmup"] for row in self.records),
                    "completed_startups": len(self.arms) * self.args.repetitions,
                    "arms": summaries,
                    "records": self.records,
                    "wall_seconds": time.perf_counter() - self.started,
                    "scope": "80-token reused development pilot, original candidate bridge runtime; historical235.9TPS is separate256-token panel, not this denominator",
                    "untouched_quality_qualified": False,
                    "predictor_validated": False,
                    "backlog_started": False,
                },
            )
        except BaseException as error:
            if self.created_output:
                atomic(
                    self.output / "failure.json",
                    {
                        "error": repr(error),
                        "completed_records": self.records,
                        "wall_seconds": time.perf_counter() - self.started,
                    },
                )
            raise


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--allow-cuda", action="store_true")
    Requests(parser.parse_args()).run()


if __name__ == "__main__":
    main()

"""Build/check the bounded native actor trace interposer without opening any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/philip/binary-eagle-decoding"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    project, output = args.project.resolve(), args.output.resolve()
    runtime_run = project / "runs/w1a8-modern-packed-runtime-20261103-01"
    candidate = runtime_run / "build03-result.json"
    result = json.loads(candidate.read_text())
    if result["status"] != "COMPILED_NOT_GPU_QUALIFIED":
        raise ValueError("pinned candidate build receipt required")
    for ref in result["binaries"]:
        if pin(ref["path"]) != ref:
            raise ValueError("candidate image changed: " + ref["path"])
    candidate_sources = runtime_run / "modified-source-pins03.json"
    for ref in json.loads(candidate_sources.read_text()):
        if pin(ref["path"]) != ref:
            raise ValueError("candidate source changed: " + ref["path"])
    output.mkdir(exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    source = Path(__file__).with_name("modern_trace.cpp")
    compiled, builder = snapshot / source.name, snapshot / Path(__file__).name
    shutil.copyfile(source, compiled)
    shutil.copyfile(__file__, builder)
    runtime = runtime_run / "source"
    libraries = runtime_run / "build02/bin"
    library = libraries / "libllama.so.0.8.0"
    interposer = output / "libw1a8-modern-trace.so"
    compiler = Path(shutil.which("c++")).resolve()
    dependency = output / "trace.d"
    command = [
        str(compiler),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        "-MMD",
        "-MF",
        str(dependency),
        "-I",
        str(runtime / "include"),
        "-I",
        str(runtime / "src"),
        "-I",
        str(runtime / "ggml/include"),
        "-I",
        str(runtime / "vendor/nlohmann"),
        str(compiled),
        "-L",
        str(libraries),
        "-Wl,-rpath," + str(libraries),
        "-Wl,-z,defs",
        "-lllama",
        "-lggml-base",
        "-ldl",
        "-pthread",
        "-o",
        str(interposer),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": str(libraries)}
    # Never inherit a preload or trace destination into build/ABI subprocesses.
    for key in ("LD_PRELOAD", "W1A8_TRACE_DIR", "W1A8_TRACE_LLAMA", "W1A8_TRACE_MAP"):
        env.pop(key, None)
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    (output / "compile.stdout").write_text(completed.stdout)
    (output / "compile.stderr").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("trace compile failed: " + str(output / "compile.stderr"))
    probe = (
        "import ctypes; p=ctypes.CDLL(None).w1a8_trace_abi_check; "
        "p.restype=ctypes.c_char_p; print(p().decode())"
    )
    base_env = {**env, "LD_PRELOAD": str(interposer)}
    disabled = subprocess.run(
        [sys.executable, "-c", probe], env=base_env, capture_output=True, text=True, check=True
    )
    checks = {"disabled_abi": json.loads(disabled.stdout)}
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        root = Path(temporary)
        mapping = root / "map.ids"
        mapping.write_text("\n".join(str(x) for x in range(50000)) + "\n")
        good = root / "good"
        good.mkdir()
        configured = {
            **base_env,
            "W1A8_TRACE_DIR": str(good),
            "W1A8_TRACE_LLAMA": str(library),
            "W1A8_TRACE_MAP": str(mapping),
        }
        checked = subprocess.run(
            [sys.executable, "-c", probe],
            env=configured,
            capture_output=True,
            text=True,
            check=True,
        )
        checks["configured_abi"] = json.loads(checked.stdout)
        checks["configured_manifest"] = json.loads((good / "manifest.json").read_text())
        if (good / "calls.jsonl").read_bytes() != b"":
            raise RuntimeError("ABI check unexpectedly invoked native model processing")
        bad = root / "bad"
        bad.mkdir()
        badmap = root / "bad-map.ids"
        badmap.write_text("1\n")
        cases = {
            "missing_environment": {**base_env, "W1A8_TRACE_DIR": str(bad)},
            "wrong_library": {
                **configured,
                "W1A8_TRACE_DIR": str(bad),
                "W1A8_TRACE_LLAMA": str(interposer),
            },
            "invalid_map": {
                **configured,
                "W1A8_TRACE_DIR": str(bad),
                "W1A8_TRACE_MAP": str(badmap),
            },
            "nonempty_directory": configured,
            "relative_directory": {**configured, "W1A8_TRACE_DIR": "relative-path"},
        }
        for name, environment in cases.items():
            tested = subprocess.run(
                [sys.executable, "-c", probe], env=environment, capture_output=True, text=True
            )
            if tested.returncode != 86 or "W1A8_TRACE_FATAL" not in tested.stderr:
                raise RuntimeError("trace environment guard failed: " + name)
            checks[name] = {"exit": tested.returncode, "reason": tested.stderr.strip()}
        if list(bad.iterdir()) or (good / "failure.json").exists():
            raise RuntimeError("rejected configuration wrote into an unclaimed destination")
    dependencies = dependency.read_text().replace("\\\n", " ").split(":", 1)[1].split()
    headers = [pin(p) for p in dependencies if Path(p) != compiled]
    if not any(Path(ref["path"]).name == "llama-batch.h" for ref in headers):
        raise RuntimeError("batch ABI header absent from compiler dependency pins")
    linked = subprocess.run(
        ["ldd", str(interposer)], env=env, capture_output=True, text=True, check=True
    ).stdout
    if "not found" in linked:
        raise RuntimeError("unresolved interposer dependency")
    receipt = {
        "status": "CPU_BUILD_ABI_GUARDS_PASS_NOT_SERVER_TESTED",
        "source": pin(compiled),
        "builder": pin(builder),
        "interposer": pin(interposer),
        "compiler": pin(compiler),
        "command": command,
        "candidate_build": pin(candidate),
        "candidate_source_pins": pin(candidate_sources),
        "candidate_binaries": result["binaries"],
        "headers": headers,
        "checks": checks,
        "linked_libraries": linked,
        "environment": {
            "LD_PRELOAD": str(interposer),
            "W1A8_TRACE_LLAMA": str(library),
            "W1A8_TRACE_DIR": "REQUIRED_ABSOLUTE_EXISTING_EMPTY_DIRECTORY",
            "W1A8_TRACE_MAP": "REQUIRED_ABSOLUTE_50000_NATIVE_D2T_IDS",
            "LD_LIBRARY_PATH": str(libraries),
        },
        "output_cap_bytes": 64 * 1024 * 1024,
        "failure_exit": 86,
        "model_opened": False,
        "backend_initialized": False,
        "gpu_used": False,
        "wall_seconds": time.perf_counter() - started,
        "native_trace_qualified": False,
        "scope": "CPU compile and no-model ABI/environment guards only",
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "build": str(output / "build.json")}))


if __name__ == "__main__":
    main()

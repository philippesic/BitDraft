"""Root-operated CPU build for the original-runtime actual-state diagnostic.

No target/draft is opened and no backend is initialized by build checks. Invoke:
python modern_actual_native_replay_build.py --output NEWDIR --loaded-maps OLD_MAPS
Then prepare cases with modern_actual_native_cases.py. Root alone runs replay:
GGML_CUDA_DECODE_CONSISTENT=32 GGML_W1AX_ACT_BITS=8 CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=ORIGINAL_BIN EXEC --cases cases.json --model parent --out NEWDIR --allow-cuda
Candidate uses --model candidate --calibration PARENT_OUT/replay.json with identical cases.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from modern_collect import atomic, pin, read, require

PROJECT = Path("/home/philip/binary-eagle-decoding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--loaded-maps",
        type=Path,
        required=True,
        help="previous real original-runtime maps, for dynamic CUDA/system image pins",
    )
    args = parser.parse_args()
    begin = time.perf_counter()
    require(not args.output.exists(), "build output exists")
    for path in (args.output.parent, Path("/mnt/c")):
        require(
            shutil.disk_usage(path).free >= 2 * 1024**3 + 64 * 1024**2,
            "storage floor: " + str(path),
        )
    runtime = PROJECT / "runs/w1a8-modern-packed-runtime-20261103-01"
    build_ref = pin(runtime / "build03-result.json")
    original = read(build_ref["path"])
    require(original["status"] == "COMPILED_NOT_GPU_QUALIFIED", "original packed build required")
    expected = {Path(r["path"]).resolve().name: r for r in original["binaries"]}
    for ref in original["binaries"]:
        require(pin(ref["path"]) == ref, "original binary changed: " + ref["path"])
    sources_ref = pin(runtime / "modified-source-pins03.json")
    for ref in read(sources_ref["path"]):
        require(pin(ref["path"]) == ref, "original source changed: " + ref["path"])
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source"
    snapshot.mkdir()
    source = Path(__file__).with_name("modern_actual_native_replay.cpp")
    cpp, builder = snapshot / source.name, snapshot / Path(__file__).name
    shutil.copyfile(source, cpp)
    shutil.copyfile(__file__, builder)
    binary = args.output / "modern_actual_native_replay"
    dependency = args.output / "replay.d"
    libraries = runtime / "build02/bin"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GGML_", "LLAMA_", "W1A8_", "W1AX_", "CUDA_"))
        and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")
    }
    env.update(LD_LIBRARY_PATH=str(libraries), CUDA_VISIBLE_DEVICES="")
    compiler = Path(shutil.which("c++")).resolve()
    argv = [
        str(compiler),
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-MMD",
        "-MF",
        str(dependency),
        "-I",
        str(runtime / "source/include"),
        "-I",
        str(runtime / "source/ggml/include"),
        "-I",
        str(runtime / "source/vendor/nlohmann"),
        str(cpp),
        "-L",
        str(libraries),
        "-Wl,-rpath," + str(libraries),
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-ldl",
        "-lcrypto",
        "-o",
        str(binary),
    ]
    completed = subprocess.run(argv, env=env, capture_output=True, text=True)
    (args.output / "compile.stdout").write_text(completed.stdout)
    (args.output / "compile.stderr").write_text(completed.stderr)
    require(completed.returncode == 0, "compile failed; retained stderr")
    checks = {}
    for flag in ("--help", "--abi-check"):
        result = subprocess.run(
            [str(binary), flag], env=env, capture_output=True, text=True, check=True
        )
        checks[flag] = result.stdout
    images = {
        str(Path(r["path"]).resolve())
        for r in original["binaries"]
        if ".so" in Path(r["path"]).name
    }
    # Include runtime images known to load dynamically during actual CUDA work.
    for line in args.loaded_maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
            require(not fields[5].endswith(" (deleted)"), "deleted prior loaded image")
            images.add(str(Path(fields[5]).resolve()))
    linked = {}
    for image in [str(binary)] + sorted(images):
        result = subprocess.run(["ldd", image], env=env, capture_output=True, text=True, check=True)
        require("not found" not in result.stdout, "unresolved helper/runtime dependency")
        linked[image] = result.stdout
        images.update(str(Path(p).resolve()) for p in re.findall(r"(/[^\s()]+)", result.stdout))
    refs = [pin(image) for image in sorted(images)]
    for ref in refs:
        name = Path(ref["path"]).name
        if name.startswith(("libllama", "libggml", "libmtmd")):
            require(
                name in expected and ref["sha256"] == expected[name]["sha256"],
                "non-original native image: " + name,
            )
    headers = dependency.read_text().replace("\\\n", " ").split(":", 1)[1].split()
    receipt = {
        "status": "CPU_BUILD_CHECKS_ONLY",
        "original_runtime_only": True,
        "source": pin(cpp),
        "builder": pin(builder),
        "compiler": pin(compiler),
        "command": argv,
        "executable": pin(binary),
        "candidate_build": build_ref,
        "candidate_source_pins": sources_ref,
        "prior_loaded_maps": pin(args.loaded_maps),
        "runtime_libraries": refs,
        "cpu_library": expected["libggml-cpu.so.0.28.0"],
        "cuda_library": expected["libggml-cuda.so.0.28.0"],
        "headers": [pin(p) for p in headers if Path(p) != cpp],
        "checks": checks,
        "linked_libraries": linked,
        "wall_seconds": time.perf_counter() - begin,
        "environment": {
            "LD_LIBRARY_PATH": str(libraries),
            "GGML_CUDA_DECODE_CONSISTENT": "32",
            "GGML_W1AX_ACT_BITS": "8",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "model_opened": False,
        "backend_initialized": False,
        "native_qualified": False,
    }
    atomic(args.output / "build.json", receipt)
    print(str(args.output / "build.json"))


if __name__ == "__main__":
    main()

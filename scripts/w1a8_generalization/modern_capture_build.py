"""Build only the small native capture helper against the retained runtime.

No model is opened and no backend is initialized by this build or its checks.
The receipt supplies the exact LD_LIBRARY_PATH and explicit backend arguments
for a later GPU-owner launch. No retained source or library is modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def pin(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def command(argv, **kwargs):
    return subprocess.run(list(map(str, argv)), check=True, text=True, capture_output=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/philip/binary-eagle-decoding"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    source = Path(__file__).with_name("modern_capture.cpp").resolve(strict=True)
    runtime = project / "runs/spec-shortlist-20261030/runtime"
    libraries = project / "runs/spec-shortlist-20261030/build/bin"
    baseline = project / "experiments/w1a8-standing-20261102/baseline.json"
    frozen = json.loads(baseline.read_text())
    cuda = pin(frozen["cuda"]["path"])
    if (cuda["sha256"], cuda["bytes"]) != (frozen["cuda"]["sha256"], frozen["cuda"]["bytes"]):
        raise ValueError("immutable CUDA overlay does not match frozen baseline")
    compiler = Path(shutil.which(args.compiler) or args.compiler).resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite build directory: {output}")
    if shutil.disk_usage(output.parent).free < 32 * 1024**2:
        raise OSError("need 32 MiB free for helper build")
    symbols = command(["nm", "-D", "--defined-only", libraries / "libllama.so.0.8.0"]).stdout
    demangled = command(["c++filt"], input=symbols).stdout
    exports = ["llama_set_embeddings_layer_inp(llama_context*, unsigned int, bool)",
               "llama_get_embeddings_layer_inp(llama_context*, unsigned int)"]
    if not all(symbol in demangled for symbol in exports):
        raise RuntimeError("retained libllama does not export required native input tap API")
    output.mkdir()
    overlay = output / "runtime-overlay"
    overlay.mkdir()
    for name in ("libggml-cuda.so", "libggml-cuda.so.0", "libggml-cuda.so.0.28.0"):
        (overlay / name).symlink_to(cuda["path"])
    environment = {"LD_LIBRARY_PATH": f"{overlay}:{libraries}", "GGML_CUDA_DECODE_CONSISTENT": "32"}
    env = {**os.environ, **environment}
    executable = output / "modern_capture"
    dependency = output / "modern_capture.d"
    argv = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-MMD", "-MF", dependency,
            "-I", runtime / "include", "-I", runtime / "ggml/include", source,
            "-L", libraries, "-Wl,-rpath," + str(libraries), "-lllama", "-lggml", "-lggml-base", "-o", executable]
    compiled = command(argv, env=env)
    checks = {
        "help": command([executable, "--help"], env=env).stdout,
        "abi": json.loads(command([executable, "--abi-check"], env=env).stdout),
        "linked_libraries": command(["ldd", executable], env=env).stdout,
    }
    if "not found" in checks["linked_libraries"]:
        raise RuntimeError("unresolved helper runtime dependency")
    rejected = subprocess.run([str(executable), "--unexpected"], text=True, capture_output=True, env=env)
    if rejected.returncode != 1 or "unknown" not in rejected.stderr:
        raise RuntimeError("invalid argument was not rejected before backend initialization")
    checks["invalid_argument_rejected"] = True
    headers = shlex.split(dependency.read_text().replace("\\\n", " ").split(":", 1)[1])
    implementation_files = ["src/llama-context.cpp", "src/llama-context.h", "src/models/qwen35.cpp",
                            "src/llama-graph.cpp", "src/llama-graph.h", "src/llama.cpp", "ggml/src/ggml-backend-reg.cpp"]
    runtime_libraries = [pin(libraries / name) for name in
                         ("libllama.so.0.8.0", "libggml.so.0.28.0", "libggml-base.so.0.28.0", "libggml-cpu.so.0.28.0")]
    receipt = {
        "status": "compiled_linked_cpu_checks_only",
        "native_capture_validated": False,
        "model_opened": False,
        "backend_initialized": False,
        "source": pin(source), "builder": pin(__file__), "baseline": pin(baseline),
        "compiler": pin(compiler), "compiler_version": command([compiler, "--version"]).stdout,
        "runtime_revision": command(["git", "-C", runtime, "rev-parse", "HEAD"]).stdout.strip(),
        "runtime_sources": [pin(runtime / name) for name in implementation_files],
        "included_local_headers": [pin(path) for path in headers if Path(path).resolve() != source],
        "runtime_libraries": runtime_libraries, "cuda_overlay": cuda,
        "executable": pin(executable), "compile_argv": list(map(str, argv)),
        "compile_stderr": compiled.stderr, "environment": environment,
        "required_backend_argv": ["--cpu-library", str(libraries / "libggml-cpu.so.0.28.0"), "--cuda-library", cuda["path"]],
        "checks": checks,
        "remaining_checks": ["GPU-owner capture using immutable target IDs and target pin",
                             "verify tap dimensions, row order, finite F32 and selected full logits",
                             "compare target-only and tap-enabled native greedy trajectories",
                             "compare chunk512 and chunk1 replay with a practical native trajectory/numeric gate",
                             "pin target, token inputs, capture outputs and build receipt before training"],
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "build.json"),
                      "executable": str(executable), "native_capture_validated": False}))


if __name__ == "__main__":
    main()

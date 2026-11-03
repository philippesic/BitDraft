"""CPU build/checks for replay against the separately qualified packed runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/philip/binary-eagle-decoding"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    run = project / "runs/w1a8-modern-packed-runtime-20261103-01"
    result_path = run / "build03-result.json"
    result = json.loads(result_path.read_text())
    if result["status"] != "COMPILED_NOT_GPU_QUALIFIED":
        raise ValueError("qualified candidate build receipt required")
    for ref in result["binaries"]:
        if pin(ref["path"]) != ref:
            raise ValueError("candidate binary changed: " + ref["path"])
    refs = json.loads((run / "modified-source-pins03.json").read_text())
    for ref in refs:
        if pin(ref["path"]) != ref:
            raise ValueError("candidate source changed: " + ref["path"])
    output.mkdir(exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    source = Path(__file__).with_name("modern_replay.cpp")
    compiled = snapshot / source.name
    builder = snapshot / Path(__file__).name
    shutil.copyfile(source, compiled)
    shutil.copyfile(__file__, builder)
    runtime, libraries = run / "source", run / "build02/bin"
    compiler = Path(shutil.which("c++")).resolve()
    env = {**os.environ, "LD_LIBRARY_PATH": str(libraries), "CUDA_VISIBLE_DEVICES": ""}
    executable = output / "modern_replay"
    dependency = output / "replay.d"
    command = [str(compiler), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-MMD",
               "-MF", str(dependency), "-I", str(runtime / "include"), "-I",
               str(runtime / "ggml/include"), "-I", str(runtime / "vendor/nlohmann"),
               str(compiled), "-L", str(libraries), "-Wl,-rpath," + str(libraries),
               "-lllama", "-lggml", "-lggml-base", "-ldl", "-o", str(executable)]
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    (output / "compile.stdout").write_text(completed.stdout)
    (output / "compile.stderr").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("replay compile failed; see " + str(output / "compile.stderr"))
    checks = {}
    for flag in ("--help", "--abi-check"):
        completed = subprocess.run([str(executable), flag], env=env, check=True,
                                   capture_output=True, text=True)
        checks[flag] = completed.stdout
    invalid = subprocess.run([str(executable), "--invalid"], env=env,
                             capture_output=True, text=True)
    if invalid.returncode != 1 or "argument" not in invalid.stderr:
        raise RuntimeError("unknown replay argument guard failed")
    checks["invalid_argument_rejected_before_backend"] = True
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temporary = Path(temporary)
        packet = temporary / "cases.json"
        packet.write_text(json.dumps({"schema": "wrong", "cases": []}))
        destination = temporary / "uncreated"
        command2 = [str(executable), "--target", "/unopened.gguf", "--draft", "/unopened.gguf",
                    "--cases", str(packet), "--out", str(destination), "--cpu-library",
                    "/unopened.so", "--cuda-library", "/unopened.so"]
        tested = subprocess.run(command2, env={**env, "GGML_CUDA_DECODE_CONSISTENT": "32",
                                              "GGML_W1AX_ACT_BITS": "8"},
                                capture_output=True, text=True)
        if tested.returncode != 1 or "unsupported replay packet" not in tested.stderr or destination.exists():
            raise RuntimeError("case schema was not rejected before backend/model initialization")
        checks["invalid_packet_rejected_before_backend"] = True
    linked = subprocess.run(["ldd", str(executable)], env=env, check=True,
                            capture_output=True, text=True).stdout
    if "not found" in linked:
        raise RuntimeError("replay runtime dependency unresolved")
    headers = dependency.read_text().replace("\\\n", " ").split(":", 1)[1].split()
    receipt = {"status": "CPU_BUILD_CHECKS_ONLY", "source": pin(compiled), "builder": pin(builder),
               "executable": pin(executable), "compiler": pin(compiler), "command": command,
               "candidate_build": pin(result_path), "candidate_source_pins": pin(run / "modified-source-pins03.json"),
               "candidate_binaries": result["binaries"], "headers": [pin(p) for p in headers if Path(p) != compiled],
               "environment": {"LD_LIBRARY_PATH": str(libraries), "GGML_CUDA_DECODE_CONSISTENT": "32",
                               "GGML_W1AX_ACT_BITS": "8"}, "checks": checks, "linked_libraries": linked,
               "required_backend_argv": ["--cpu-library", str(libraries / "libggml-cpu.so.0.28.0"),
                                         "--cuda-library", str(libraries / "libggml-cuda.so.0.28.0")],
               "native_replay_qualified": False, "model_opened": False, "backend_initialized": False}
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "build.json")}))


if __name__ == "__main__":
    main()

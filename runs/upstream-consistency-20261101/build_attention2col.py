from pathlib import Path
import hashlib
import json
import subprocess
import time

P = Path(__file__).resolve().parent
start = time.perf_counter()
command = ["cmake", "--build", str(P / "attention2col-build"), "--target", "ggml-cuda", "-j", "2"]
result = subprocess.run(command)
libraries = {}
if result.returncode == 0:
    for path in (P / "attention2col-build/bin").glob("*.so.*"):
        if path.is_symlink():
            continue
        with path.open("rb") as stream:
            libraries[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
receipt = {"command": command, "exit_code": result.returncode,
           "compilation_seconds": time.perf_counter() - start, "training_seconds": 0,
           "gpu_evaluation_seconds": 0, "libraries": libraries,
           "source_receipt": str(P / "attention2col-source-receipt.json")}
(P / "attention2col-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(result.returncode)

"""Finite local development batch, serial GPU ownership, no training."""
from pathlib import Path
import json
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
tasks = [
    ["python3", str(P / "run_quality.py"), "original"],
    ["python3", str(P / "run_quality.py"), "conservative32"],
    ["python3", str(P / "run_quality.py"), "quant4group"],
    ["python3", str(P / "run_matched_requests.py"), "--output", "development-stock-01"],
    ["python3", str(P / "run_matched_requests.py"), "--output", "development-9b-mtp-01",
     "--models", "Qwen3.5-9B-Q4_K_M.gguf", "qwen35-9b-mtp-nvfp4.gguf", "--nine-b-mode", "mtp"],
]
records = []
for command in tasks:
    control = json.loads((Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text())["rtx5080"]
    assert not control.get("pause_requested") and not control.get("stop_all_requested")
    assert not (P / "DEVELOPMENT-STOP").exists()
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    records.append({"command": command, "exit_code": result.returncode,
                    "phase_total_seconds": time.perf_counter() - start})
    (P / "development-batch-progress.json").write_text(json.dumps(records, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
(P / "development-batch-receipt.json").write_text(json.dumps({"completed": True,
    "records": records, "training_seconds": 0, "untouched_confirmation": False}, indent=2) + "\n")

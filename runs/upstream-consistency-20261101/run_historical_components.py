from pathlib import Path
import fcntl
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
control = json.loads((Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text())["rtx5080"]
assert not control.get("pause_requested") and not control.get("stop_all_requested")
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
command = ["python3", "-B", str(ROOT / "runs/5080-broader-training-20261031/request-evaluator01.py"),
           "run", "--config", str(P / "historical-component-config.json"),
           "--output", str(P / "historical-component-01"), "--stop", str(P / "HISTORICAL-STOP"),
           "--diagnostic", "--allow-native"]
start = time.perf_counter()
result = subprocess.run(command, cwd=ROOT)
(P / "historical-component-receipt.json").write_text(json.dumps({
    "command": command, "exit_code": result.returncode,
    "diagnostic_total_wall_seconds": time.perf_counter() - start,
    "training_seconds": 0, "clean_performance_claim": False,
    "instrumentation": "synchronized CPU wall components, not CUDA kernel attribution",
    "concurrent_cpu_preparation": "isolated attention2col CUDA build, two compile jobs",
}, indent=2) + "\n")
raise SystemExit(result.returncode)

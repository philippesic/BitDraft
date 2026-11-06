from pathlib import Path
import fcntl
import json
import os
import re
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OLD = ROOT / "runs/spec-shortlist-20261030"
MODELS = ("Qwen3.5-4B-Q4_K_M.gguf", "qwen35-4b-mtp-nvfp4.gguf",
          "Qwen3.5-9B-Q4_K_M.gguf", "qwen35-9b-mtp-nvfp4.gguf")
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
control_path = Path.home() / ".config/binary-eagle-decoding/gpu-control.json"
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
OUT = P / "native-groups-01"
OUT.mkdir(exist_ok=False)
rows = []
tasks = [("original_negative", MODELS[0]),
         *[(variant, model) for variant in ("quant4group",) for model in MODELS]]
for variant, model in tasks:
    control = json.loads(control_path.read_text())["rtx5080"]
    assert not control.get("pause_requested") and not control.get("stop_all_requested")
    assert not (OUT / "STOP").exists()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_") and k not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    env["LD_LIBRARY_PATH"] = str(OLD / "build/bin")
    if variant != "original_negative":
        env.update(GGML_CUDA_DECODE_CONSISTENT="32", LD_LIBRARY_PATH=str(P / (variant + "-overlay")) + ":" + str(OLD / "build/bin"))
    stage = OUT / variant / model
    stage.mkdir(parents=True)
    command = [str(P / "test-grouped-parity"), "-m", str(OLD / "models" / model),
               "-ngl", "99", "-t", "4", "-fa", "on", "--test", "spec-parity", "--logs", str(stage / "logs")]
    (stage / "command.json").write_text(json.dumps({"command": command, "environment": {k: env[k] for k in env if k.startswith("GGML_") or k == "LD_LIBRARY_PATH"}}, indent=2) + "\n")
    start = time.perf_counter()
    with (stage / "stdout.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = "\n".join(p.read_text(errors="replace") for p in (stage / "logs").glob("*.log"))
    checks = re.findall(r"spec parity: checks=(\d+) greedy_flips=(\d+) max_abs=([^\s]+)", text)
    row = {"variant": variant, "model": model, "exit_code": result.returncode,
           "native_process_wall_seconds": time.perf_counter() - start,
           "checks": sum(int(x[0]) for x in checks), "greedy_flips": sum(int(x[1]) for x in checks),
           "max_abs": max((float(x[2]) for x in checks), default=None), "executed_summaries": len(checks)}
    rows.append(row)
    (OUT / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(row), flush=True)
assert rows[0]["exit_code"] != 0 and rows[0]["checks"] > 0, "negative control was not executed and rejected"
(OUT / "summary.json").write_text(json.dumps({"completed": True, "rows": rows,
    "invalid_control_rejected": True, "candidate_all_strict_gates_passed": all(r["exit_code"] == 0 for r in rows[1:]),
    "scope": "native numerical/state/rejection checks; widths2/3/4/5/7/8/16/17/32; no performance/quality claim"}, indent=2) + "\n")

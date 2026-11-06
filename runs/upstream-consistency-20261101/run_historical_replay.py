from pathlib import Path
import fcntl
import importlib.util
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
control_path = Path.home() / ".config/binary-eagle-decoding/gpu-control.json"
control = json.loads(control_path.read_text())["rtx5080"]
assert not control.get("pause_requested") and not control.get("stop_all_requested")
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
spec = importlib.util.spec_from_file_location("historical_evaluator", ROOT / "runs/5080-broader-training-20261031/request-evaluator01.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
config = json.loads((P / "historical-component-config.json").read_text())
old_ready = json.loads((ROOT / "runs/shortlist-throughput-20261027/native/ready.json").read_text())
out = P / "historical-replay-01"
out.mkdir(exist_ok=False)
records = []
for name in ("parent3504", "fresh_best", "q4_k_m"):
    control = json.loads(control_path.read_text())["rtx5080"]
    assert not control.get("pause_requested") and not control.get("stop_all_requested")
    assert not (P / "HISTORICAL-STOP").exists()
    model = config["models"][name]
    evaluator.checked(model)
    env = evaluator.environment(config, model)
    env["LD_LIBRARY_PATH"] = str(Path(old_ready["server"]["path"]).parent)
    command = [str(P / "historical-replay"), model["path"],
               str(P / "historical-replay-cases.json"), str(out / name), "on"]
    start = time.perf_counter()
    with (out / (name + ".log")).open("w") as log:
        result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    records.append({"arm": name, "command": command, "model": model, "exit_code": result.returncode,
                    "native_load_replay_capture_wall_seconds": time.perf_counter() - start})
    (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
    assert result.returncode == 0, name + " replay failed; log retained"
(out / "receipt.json").write_text(json.dumps({"completed": True, "records": records,
    "training_seconds": 0, "scope": "12 fixed development histories per arm; native diagnostic only",
    "runtime": old_ready["runtime_libraries"], "clean_performance_claim": False}, indent=2) + "\n")

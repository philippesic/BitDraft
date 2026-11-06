from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
manifest = json.loads((P / "frontier-replay-inputs/manifest.json").read_text())
assert manifest["completed"] and manifest["capture_preserved_original_decisions"]
spec = importlib.util.spec_from_file_location("historical_evaluator", ROOT / "runs/5080-broader-training-20261031/request-evaluator01.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
config = json.loads((P / "historical-component-config.json").read_text())
out = P / "frontier-counterfactual-01"
out.mkdir(exist_ok=False)
records = []
for arm in ("parent3504", "fresh_best"):
    model = config["models"][arm]
    evaluator.checked(model)
    (out / arm).mkdir()
    env = evaluator.environment(config, model)
    env["LD_LIBRARY_PATH"] = str(ROOT / "runs/shortlist-throughput-20261027/native/build/bin")
    for number, batch in enumerate(manifest["batches"]):
        control = json.loads((Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text())["rtx5080"]
        assert not control.get("pause_requested") and not control.get("stop_all_requested")
        assert not (P / "HISTORICAL-STOP").exists()
        assert hashlib.sha256(Path(batch["path"]).read_bytes()).hexdigest() == batch["sha256"]
        directory = out / arm / f"batch-{number:02d}"
        command = [str(P / "historical-replay"), model["path"], batch["path"], str(directory), "on"]
        start = time.perf_counter()
        with (out / arm / f"batch-{number:02d}.log").open("w") as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        records.append({"arm": arm, "batch": number, "command": command, "exit_code": result.returncode,
                        "native_process_wall_seconds": time.perf_counter() - start})
        (out / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
        assert result.returncode == 0, "native replay failed; original logs preserved"
(out / "receipt.json").write_text(json.dumps({"completed": True, "records": records,
    "training_seconds": 0, "scope": "same actual target features/chunks/seed/labels across two models; development diagnostic only"}, indent=2) + "\n")

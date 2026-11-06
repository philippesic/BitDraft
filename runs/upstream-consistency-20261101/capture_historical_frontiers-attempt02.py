"""Capture existing development requests without changing their numerical source."""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import shutil
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OUT = P / "historical-frontier-capture-02"
lock = (Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = json.loads((Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text())["rtx5080"]
assert not state.get("pause_requested") and not state.get("stop_all_requested")
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
assert shutil.disk_usage(P).free > 16 * 1024**3
OUT.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location("historical_evaluator", ROOT / "runs/5080-broader-training-20261031/request-evaluator01.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
config = json.loads((P / "historical-component-config.json").read_text())
config["models"] = {name: config["models"][name] for name in ("parent3504", "q4_k_m")}
config["port"] = 18940
configuration = OUT / "configuration.json"
configuration.write_text(json.dumps(config, indent=2) + "\n")
# The existing capture API requires these cache/context constraints. They do
# not trigger inside this fixed context, but full proposal identity is checked.
evaluator.load_config(configuration)
import evaluate_nine_model_native as command_module
original_command = command_module.native_command
command_module.native_command = lambda *a, **kw: original_command(*a, **kw) + ["--no-context-shift", "--ctx-checkpoints", "0"]
original_environment = evaluator.environment


def environment(binding, model, trace=None):
    env = original_environment(binding, model, trace)
    arm = next(name for name, record in binding["models"].items() if record["sha256"] == model["sha256"])
    destination = OUT / "capture" / arm
    destination.mkdir(parents=True, exist_ok=False)
    env.pop("DSPARK_QUALITY_CAPTURE", None)
    env.update(EAGLE_CAPTURE_PREFIX=str(destination / "capture"),
               EAGLE_CAPTURE_TARGET_FEATURES="1", EAGLE_CAPTURE_TARGET_FEATURES_LIMIT="40000",
               EAGLE_RECORD_ROUNDS_JSONL=str(destination / "rounds.jsonl"))
    return env


evaluator.environment = environment
args = argparse.Namespace(config=configuration, output=OUT / "requests", stop=P / "HISTORICAL-STOP",
                          diagnostic=True, allow_native=True, preflight=False)
start = time.perf_counter()
evaluator.run(args)
(OUT / "receipt.json").write_text(json.dumps({"completed": True, "training_seconds": 0,
    "capture_total_seconds": time.perf_counter() - start, "numerical_source_changed": False, "required_capture_flags": ["--no-context-shift", "--ctx-checkpoints", "0"],
    "scope": "existing development requests and actual verifier histories, not training or final heldout",
    "root_responsibility": "verify capture did not alter any request IDs, proposed tokens or accepted counts",
    "instrumentation": "target features plus disposition and round ledgers; full-vocabulary target logits disabled"}, indent=2) + "\n")

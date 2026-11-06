"""Execute the frozen confirmation once, without training or candidate selection."""
from pathlib import Path
import datetime
import hashlib
import json
import signal
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
SIGNAL = "local-confirmation-20261101-complete"
records = []
started = time.perf_counter()
completed = False
failure = None


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    freeze_path = P / "final-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    input_receipt = json.loads((P / "confirmation-inputs/receipt.json").read_text())
    assert input_receipt["freeze_sha256"] == hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    assert input_receipt["performance_prompts"] == input_receipt["quality_tasks"] == 24
    assert input_receipt["quality_known_wrong_controls_rejected"] == 24
    for ref in freeze["evaluator_sources"]:
        assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    for filename, expected in input_receipt["files"].items():
        assert hashlib.sha256((P / "confirmation-inputs" / filename).read_bytes()).hexdigest() == expected
    libraries = {
        "conservative32": freeze["control_library"],
        "quant4group": {"path": freeze["candidate_library"]["library"], "sha256": freeze["candidate_library"]["sha256"]},
    }
    for ref in [*freeze["targets"].values(), *libraries.values()]:
        with Path(ref["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == ref["sha256"]
    metadata = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "freeze_sha256": input_receipt["freeze_sha256"], "source": str(Path(__file__).resolve()),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "preparation_seconds": time.perf_counter() - started,
                "additional_unchanged_helpers": {str(ROOT / "scripts/benchmark_native_eagle.py"):
                    hashlib.sha256((ROOT / "scripts/benchmark_native_eagle.py").read_bytes()).hexdigest()}}
    (P / "confirmation-preflight.json").write_text(json.dumps(metadata, indent=2) + "\n")
    commands = [["python3", str(P / "run_final_quality.py"), variant,
                 str(P / "confirmation-inputs/quality-cases.json")]
                for variant in ("original", "conservative32", "quant4group")]
    commands.append(["python3", str(P / "run_final_requests.py"), "--output", "confirmation-requests-01",
                     "--panel", str(P / "confirmation-inputs/requests.json"), "--repetitions", "5",
                     "--prompt-count", "24", "--nine-b-mode", "mtp"])
    for command in commands:
        control = json.loads((Path.home() / ".config/binary-eagle-decoding/gpu-control.json").read_text())["rtx5080"]
        assert not control.get("pause_requested") and not control.get("stop_all_requested")
        assert not (P / "CONFIRMATION-STOP").exists()
        phase_start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT)
        records.append({"command": command, "exit_code": result.returncode,
                        "phase_total_wall_seconds": time.perf_counter() - phase_start})
        (P / "confirmation-progress.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode != 0:
            raise RuntimeError(f"confirmation stage failed: {command[1]}")
    completed = True
except BaseException as exc:
    failure = repr(exc)
    raise
finally:
    (P / "confirmation-receipt.json").write_text(json.dumps({
        "completed": completed, "failure": failure, "records": records,
        "training_seconds": 0, "total_wall_seconds": time.perf_counter() - started,
        "scope": "frozen source/settings; new confirmation panel; owner must review results and release",
    }, indent=2) + "\n")
    subprocess.run(["tmux", "-L", "binary-eagle-runtime", "wait-for", "-S", SIGNAL], check=False)

import json
import shutil
import sys
from pathlib import Path

from tools.policy_review import validate_report
from tools.repo_files import hooks_path, inventory, read_json
from w1a1_eagle.capacity import backing_available
from w1a1_eagle.config import numerical_python
from w1a1_eagle.identity import source_identity


def diagnose(root: Path) -> dict[str, object]:
    hook_ready = hooks_path(root) == ".githooks"
    receipt = root / ".venv/foundation-verification.json"
    files, escapes = inventory(root)
    verification = "missing"
    if receipt.is_file():
        verification = (
            "stale" if validate_report(files, read_json(receipt)) else "current CPU proof"
        )
    return {
        "schema": 1,
        "check_python": sys.executable,
        "numerical_python": numerical_python(root),
        "hook": "ready" if hook_ready else "disabled: run git config core.hooksPath .githooks",
        "wsl_available_bytes": shutil.disk_usage(root).free,
        "backing_available_bytes": backing_available(),
        "verification": verification,
        "source_sha256": source_identity(root),
        "symlink_escapes": escapes,
        "capabilities": {
            "CPU": "run full gate for current evidence",
            "GPU": "unqualified",
            "native": "unqualified",
            "benchmark": "deferred",
        },
        "authority": "diagnostics and hashes do not authorize a run",
    }


def main() -> int:
    report = diagnose(Path(__file__).resolve().parent.parent)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["hook"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

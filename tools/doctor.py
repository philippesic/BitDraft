import json
import os
import shutil
import sys
from pathlib import Path

from tools.policy_review import validate_report
from tools.repo_files import hooks_path, inventory, read_json
from w1a1_eagle.capacity import backing_available
from w1a1_eagle.config import numerical_python
from w1a1_eagle.identity import source_identity


def diagnose(root: Path) -> dict[str, object]:
    hook = root / ".githooks/pre-commit"
    if hooks_path(root) != ".githooks":
        hook_status = "disabled: run git config core.hooksPath .githooks"
    elif not hook.is_file():
        hook_status = "missing: restore .githooks/pre-commit from the working branch"
    elif not os.access(hook, os.X_OK):
        hook_status = "not executable: run chmod +x .githooks/pre-commit"
    else:
        hook_status = "ready"
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
        "hook": hook_status,
        "wsl_available_bytes": shutil.disk_usage(root).free,
        "backing_available_bytes": backing_available(),
        "verification": verification,
        "source_sha256": source_identity(root),
        "symlink_escapes": escapes,
        "capabilities": {
            "CPU": "run full gate for current evidence",
            "GPU": "not assessed by this CPU diagnostic; see docs/support.md",
            "native": "not assessed by this CPU diagnostic; see docs/support.md",
            "benchmark": "available through python -m benchmarks; execution not assessed",
        },
        "authority": "diagnostics and hashes do not authorize a run",
    }


def main() -> int:
    report = diagnose(Path(__file__).resolve().parent.parent)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["hook"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

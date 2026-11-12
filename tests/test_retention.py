import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.retention import active_clients, execute, inspect
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import read, reference


def plan_for(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "closed-run"
    raw.mkdir()
    payload = raw / "unused-checkpoint"
    payload.write_bytes(b"obsolete state")
    evidence = tmp_path / "measurements.json"
    evidence.write_text('{"acceptance":0.25,"promoted":false}')
    plan = tmp_path / "plan.json"
    document: Record = {
        "schema": 1,
        "owner": "test investigation",
        "authorization": "test fixture only",
        "result": "results/closed-question/result.md",
        "limitations": "exact resume retired",
        "root": str(raw),
        "protected": [],
        "retained": [reference(Artifact.capture(evidence))],
        "remove": [reference(Artifact.capture(payload))],
    }
    plan.write_text(json.dumps(document))
    return plan, payload, tmp_path / "retirement.jsonl"


def test_retirement_verifies_evidence_and_preserves_failure_before_deletion(tmp_path: Path) -> None:
    plan, payload, receipt = plan_for(tmp_path)
    assert execute(plan, receipt)["phase"] == "verified"
    assert payload.exists() and not receipt.exists()
    evidence = tmp_path / "measurements.json"
    evidence.write_text("corrupt")
    with pytest.raises(ValueError):
        execute(plan, receipt, hashlib.sha256(plan.read_bytes()).hexdigest())
    assert payload.exists() and not receipt.exists()


def test_retirement_is_exact_journaled_resumable_and_preserves_other_files(tmp_path: Path) -> None:
    plan, payload, receipt = plan_for(tmp_path)
    other = payload.parent / "needed"
    other.write_bytes(b"keep")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    assert execute(plan, receipt, digest)["phase"] == "complete"
    rows = [record(cast(object, json.loads(line))) for line in receipt.read_text().splitlines()]
    assert rows[0]["phase"] == "retirement-intent" and rows[-1]["phase"] == "complete"
    assert not payload.exists() and other.read_bytes() == b"keep"
    assert execute(plan, receipt, digest)["phase"] == "complete"
    receipt.unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        execute(plan, receipt, digest)


def test_retirement_rejects_protection_escape_links_and_changed_payload(tmp_path: Path) -> None:
    plan, payload, receipt = plan_for(tmp_path)
    document = read(plan)
    document["protected"] = [str(payload)]
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="protected"):
        inspect(plan)
    document["protected"] = []
    removals = sequence(document["remove"])
    removals.append(removals[0])
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unique"):
        inspect(plan)
    removals.pop()
    original_removal = removals[0]
    outside = tmp_path / "other-project"
    outside.write_bytes(b"unrelated")
    removals[0] = reference(Artifact.capture(outside))
    plan.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="escaped"):
        inspect(plan)
    removals[0] = original_removal
    plan.write_text(json.dumps(document))
    shared = tmp_path / "shared"
    os.link(payload, shared)
    with pytest.raises(ValueError, match="hardlinks"):
        execute(plan, receipt)
    shared.unlink()
    payload.write_bytes(b"changed state!")
    with pytest.raises(ValueError):
        execute(plan, receipt)
    payload.unlink()
    payload.symlink_to(tmp_path / "measurements.json")
    with pytest.raises(ValueError, match="symlink"):
        inspect(plan)
    assert not receipt.exists()


def test_retirement_detects_live_open_file_dependency(tmp_path: Path) -> None:
    plan, payload, receipt = plan_for(tmp_path)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; f=open(sys.argv[1]); print('ready',flush=True); sys.stdin.read()",
            str(payload),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == b"ready\n"
        assert child.pid in active_clients(payload.parent)
        with pytest.raises(ValueError, match="active process"):
            execute(plan, receipt)
        assert payload.exists()
    finally:
        child.communicate(timeout=10)


def test_retirement_recovers_after_a_journaled_filesystem_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, first, receipt = plan_for(tmp_path)
    second = first.parent / "second-unused-state"
    second.write_bytes(b"second obsolete state")
    document = read(plan)
    sequence(document["remove"]).append(reference(Artifact.capture(second)))
    plan.write_text(json.dumps(document))
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    original = Path.unlink

    def fail_second(path: Path, missing_ok: bool = False) -> None:
        if path == second:
            raise PermissionError("injected retired-directory failure")
        original(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_second)
        with pytest.raises(PermissionError):
            execute(plan, receipt, digest)
    assert not first.exists() and second.exists()
    assert execute(plan, receipt, digest)["phase"] == "complete"
    assert not second.exists()

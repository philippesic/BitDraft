import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from benchmarks.staged import train_stage, verified_state


def test_refused_resume_restarts_the_stage_from_its_original_intent() -> None:
    with tempfile.TemporaryDirectory(prefix="staged-") as directory:
        stage = Path(directory)
        train = stage / "train"
        (train / "states").mkdir(parents=True)
        (train / "states" / "latest.json").write_text("{}")
        log = stage / "stage.log"
        calls: list[list[str]] = []

        def runner(argv: list[str], sink: Path, root: Path) -> None:
            calls.append(argv)
            if len(calls) == 1:
                raise ValueError("cumulative wall budget exhausted")
            train.mkdir()

        train_stage(stage / "intent-resume-4125.json", stage, train, "scope", log, stage, runner)
        assert calls[0][calls[0].index("w1a1_eagle") + 2].endswith("intent-resume-4125.json")
        assert calls[1][calls[1].index("w1a1_eagle") + 2] == str(stage / "intent.json")
        assert (stage / "train-unstarted-0" / "states" / "latest.json").exists()
        assert "restarting the stage" in log.read_text()


def test_a_failed_first_attempt_is_not_retried() -> None:
    with tempfile.TemporaryDirectory(prefix="staged-") as directory:
        stage = Path(directory)
        calls: list[list[str]] = []

        def runner(argv: list[str], sink: Path, root: Path) -> None:
            calls.append(argv)
            raise ValueError("stage step failed")

        with pytest.raises(ValueError):
            train_stage(
                stage / "intent.json", stage, stage / "train", "scope", stage / "log", stage, runner
            )
        assert len(calls) == 1


def test_the_final_state_must_match_its_pointer() -> None:
    with tempfile.TemporaryDirectory(prefix="staged-") as directory:
        train = Path(directory) / "train"
        (train / "states").mkdir(parents=True)
        payload = bytes(range(256)) * 64
        sha = hashlib.sha256(payload).hexdigest()
        state = train / "states" / (sha + ".bin")
        state.write_bytes(payload)
        pointer = {"sha256": sha, "bytes": len(payload), "updates": 8250}
        (train / "states" / "latest.json").write_text(json.dumps(pointer))
        assert verified_state(train, 8250)["sha256"] == sha
        with pytest.raises(ValueError):
            verified_state(train, 16500)
        state.write_bytes(payload[:-1] + b"\x00")
        with pytest.raises(ValueError):
            verified_state(train, 8250)

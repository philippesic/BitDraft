from pathlib import Path

import pytest

from benchmarks.kernel_capture import capture_kernel_activations, dev_task
from w1a1_eagle.config import kernel_capture_environment


def test_capture_environment_is_separate() -> None:
    environment = kernel_capture_environment(
        Path("/runtime"), Path("/helper.so"), Path("/capture"), 8
    )
    assert environment["GGML_CUDA_DECODE_CONSISTENT"] == "32"
    assert environment["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert environment["LD_PRELOAD"] == "/helper.so"
    assert environment["BITDRAFT_KERNEL_SLOTS"] == "8"


def test_capture_rejects_unknown_schema(tmp_path: Path) -> None:
    specification = tmp_path / "spec.json"
    specification.write_text('{"schema":"unknown"}')
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="specification"):
        capture_kernel_activations(specification, output)
    assert not output.exists()


@pytest.mark.parametrize("role", ["train", "test", "unknown"])
def test_capture_rejects_non_dev_task(role: str) -> None:
    with pytest.raises(ValueError, match="DEV"):
        dev_task({"tasks": [{"id": "unit", "role": role}]}, "unit")


def test_capture_requires_unique_dev_task() -> None:
    task = {"id": "unit", "role": "dev"}
    assert dev_task({"tasks": [task]}, "unit") == task
    with pytest.raises(ValueError, match="DEV"):
        dev_task({"tasks": [task, task]}, "unit")
    with pytest.raises(ValueError, match="DEV"):
        dev_task({"tasks": [task]}, "missing")

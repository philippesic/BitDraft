import math
import struct
from pathlib import Path

import pytest

from benchmarks.kernel_benchmark import Weights, metrics, samples
from w1a1_eagle.contracts import number


def test_sample_failure_is_retained() -> None:
    times, exact = samples("sample,0,12.5,1\nsample,1,13.0,0\nsample,2,12.8,1\n")
    assert times == [12.5, 13.0, 12.8]
    assert exact is False


@pytest.mark.parametrize(
    "text",
    ["", "sample,1,12,1", "sample,0,0,1", "sample,0,nan,1", "sample,0,inf,1", "sample,0,12,2"],
)
def test_bad_samples(text: str) -> None:
    with pytest.raises(ValueError):
        samples(text)


def test_metrics_count_multiply_add_once() -> None:
    result = metrics([9.0, 10.0, 11.0], 128, 256, 4, 2, 8192, 800)
    assert math.isclose(number(result["multiply_adds_per_second"]), 26_214_400_000)
    assert math.isclose(number(result["weight_bytes_per_second"]), 819_200_000)
    assert math.isclose(number(result["memory_floor_microseconds"]), 0.01024)
    assert result["spread_percent"] == 20.0


def test_weights_f16_conversion_and_bounds(tmp_path: Path) -> None:
    name = b"scale"
    directory = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    directory += struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 2, 1, 0)
    directory += bytes((-len(directory)) % 32)
    source = tmp_path / "weights.gguf"
    source.write_bytes(directory + struct.pack("<ee", 0.5, -0.0))
    weights = Weights(source)
    assert weights.tensor("scale", {1}) == struct.pack("<ff", 0.5, -0.0)
    with pytest.raises(ValueError, match="unsupported"):
        weights.tensor("scale", {26})
    source.write_bytes(directory)
    with pytest.raises(ValueError, match="bounds"):
        weights.tensor("scale", {1})


@pytest.mark.parametrize(("kind", "size"), [(12, 144), (14, 210)])
def test_preserve_baseline_quantized_payload(tmp_path: Path, kind: int, size: int) -> None:
    name = b"weight"
    directory = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    directory += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 256, 1, kind, 0)
    directory += bytes((-len(directory)) % 32)
    payload = bytes(range(size))
    source = tmp_path / "baseline.gguf"
    source.write_bytes(directory + payload)
    weights = Weights(source)
    assert weights.tensors["weight"].kind == kind
    assert weights.tensors["weight"].size == size
    assert weights.tensor("weight", {12, 14}) == payload

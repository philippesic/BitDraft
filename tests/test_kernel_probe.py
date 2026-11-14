import struct
from pathlib import Path

import pytest

from benchmarks.kernel_probe import extract_matrix, runtime_kernels


def test_extract_matrix(tmp_path: Path) -> None:
    name = b"gate"
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    tensor = struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 2, 1, 26, 0)
    data = header + tensor
    data += bytes((-len(data)) % 32)
    payload = struct.pack("<II", 0x12345678, 0xFFFFFFFF)
    source = tmp_path / "weights.gguf"
    source.write_bytes(data + payload)
    output = tmp_path / "matrix.bin"
    assert extract_matrix(source, "gate", output) == (2, 1)
    assert output.read_bytes() == payload
    source.write_bytes(data + payload[:4])
    with pytest.raises(ValueError, match="bounds"):
        extract_matrix(source, "gate", output)


def test_runtime_kernels(tmp_path: Path) -> None:
    source = tmp_path / "runtime.cu"
    kernels = "static __device__ float gained_value() {}\n"
    source.write_text("prefix\n" + kernels + "void ggml_cuda_w1ax_pack() {}")
    assert runtime_kernels(source) == kernels

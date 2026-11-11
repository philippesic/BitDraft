import hashlib
import struct
from pathlib import Path

import pytest

from w1a1_eagle.numerical.gguf_readback import Expected, metadata, verify


def text(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture(path: Path, fault: str = "") -> Expected:
    rows = [
        ("f32", (2, 2), 0, struct.pack("<4f", 1, 2, 3, 4)),
        ("i32", (1,), 26, struct.pack("<i", -1)),
        ("q4", (32, 1), 2, bytes(18)),
        ("q4k", (256, 1), 12, bytes(144)),
    ]
    fields = [
        ("dflash.w1ax.config_hash", 8, text("recipe")),
        ("tokens", 9, struct.pack("<IQ", 8, 3) + text("") + text("é") + text("<EOS>")),
        ("scores", 9, struct.pack("<IQ2f", 6, 2, 0.25, 2)),
    ]
    body = b"".join(text(name) + struct.pack("<I", kind) + value for name, kind, value in fields)
    headers, data = b"", b""
    expected: Expected = {}
    for index, (name, shape, kind, payload) in enumerate(rows):
        offset = (len(data) + 31) // 32 * 32
        data += bytes(offset - len(data)) + payload
        expected[name] = (
            {0: "F32", 26: "I32", 2: "Q4_0", 12: "Q4_K"}[kind],
            shape,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        if index == 0:
            shape = (1, 4) if fault == "shape" else shape
            kind = 26 if fault == "kind" else kind
            offset = 1 if fault == "offset" else offset
        if index == 1 and fault == "duplicate":
            name = "f32"
        headers += (
            text(name)
            + struct.pack("<I", len(shape))
            + struct.pack("<" + "Q" * len(shape), *shape)
            + struct.pack("<IQ", kind, offset)
        )
    header = b"GGUF" + struct.pack("<IQQ", 3, len(rows), len(fields)) + body + headers
    blob = header + bytes((-len(header)) % 32) + data
    if fault == "truncated":
        blob = blob[:-1]
    elif fault == "payload":
        blob = blob[:-1] + b"\1"
    path.write_bytes(blob)
    return expected


def test_independent_gguf_fixture_preserves_strings_arrays_geometry_and_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model"
    expected = fixture(path)
    fields = {name: value for name, value, _, _ in metadata(path)}
    assert fields["tokens"] == ["", "é", "<EOS>"] and fields["scores"] == [0.25, 2.0]
    verify(path, expected, "recipe")
    with pytest.raises(ValueError, match="recipe"):
        verify(path, expected, "wrong")


def test_gguf_readback_refuses_corruption_and_ambiguous_tensor_layout(tmp_path: Path) -> None:
    path = tmp_path / "model"
    for fault in ("truncated", "payload", "shape", "kind", "offset", "duplicate"):
        expected = fixture(path, fault)
        with pytest.raises(ValueError):
            verify(path, expected, "recipe")
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + text("bad-array") + struct.pack("<IIQ", 9, 9, 2)
    )
    with pytest.raises(ValueError, match="array"):
        metadata(path)

import hashlib
import math
import struct
from pathlib import Path
from typing import BinaryIO, cast

from w1a1_eagle.contracts import integer

Metadata = tuple[str, object, int, int | None]
TensorInfo = tuple[str, tuple[int, ...], int, int]
Expected = dict[str, tuple[str, tuple[int, ...], int, str]]
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


class Reader:
    def __init__(self, source: BinaryIO, size: int) -> None:
        self.source = source
        self.size = size

    def read(self, count: int) -> bytes:
        if count < 0 or count > self.size - self.source.tell() or count > 64 * 1024 * 1024:
            raise ValueError("GGUF read exceeds file or allocation bounds")
        value = self.source.read(count)
        if len(value) != count:
            raise ValueError("truncated GGUF")
        return value

    def uint(self, width: int) -> int:
        return int.from_bytes(self.read(width), "little")

    def skip(self, count: int) -> None:
        if count < 0 or count > self.size - self.source.tell():
            raise ValueError("GGUF offset exceeds file bounds")
        self.source.seek(count, 1)

    def text(self, retain: bool = True) -> str:
        count = self.uint(8)
        if not retain:
            self.skip(count)
            return ""
        return self.read(count).decode("utf-8")

    def value(self, kind: int, retain: bool) -> tuple[object, int | None]:
        if kind == 8:
            return self.text(retain), None
        if kind == 9:
            subtype, count = self.uint(4), self.uint(8)
            if subtype == 8:
                if count > (self.size - self.source.tell()) // 8:
                    raise ValueError("GGUF string array exceeds file bounds")
                values = [self.text(retain) for _ in range(count)] if retain else []
                if not retain:
                    for _ in range(count):
                        self.text(False)
                return values, subtype
            if subtype not in SCALARS:
                raise ValueError("unsupported GGUF array type")
            fmt = SCALARS[subtype]
            size = count * struct.calcsize("<" + fmt)
            if not retain:
                self.skip(size)
                return [], subtype
            data = self.read(size)
            return list(
                cast(tuple[object, ...], struct.unpack("<" + str(count) + fmt, data))
            ), subtype
        if kind not in SCALARS:
            raise ValueError("unsupported GGUF metadata type")
        fmt = SCALARS[kind]
        return cast(
            tuple[object, ...], struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))
        )[0], None

    def header(self, retain: bool) -> tuple[list[Metadata], int, int]:
        if self.read(4) != b"GGUF" or self.uint(4) != 3:
            raise ValueError("readback requires little-endian GGUF3")
        tensors, fields = self.uint(8), self.uint(8)
        if not 1 <= tensors <= 65536 or fields > 65536:
            raise ValueError("GGUF header counts outside supported bounds")
        rows: list[Metadata] = []
        names: set[str] = set()
        alignment = 32
        for _ in range(fields):
            key, kind = self.text(), self.uint(4)
            if not key or key in names:
                raise ValueError("duplicate or empty GGUF metadata key")
            names.add(key)
            value, subtype = self.value(
                kind, retain or key in {"general.alignment", "dflash.w1ax.config_hash"}
            )
            if key == "general.alignment":
                alignment = integer(value)
            rows.append((key, value, kind, subtype))
        if not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError("unsupported GGUF alignment")
        return rows, tensors, alignment


def metadata(path: Path) -> list[Metadata]:
    with path.open("rb") as source:
        return Reader(source, path.stat().st_size).header(True)[0]


def verify(path: Path, expected: Expected, recipe: str) -> None:
    with path.open("rb") as source:
        reader = Reader(source, path.stat().st_size)
        fields, count, alignment = reader.header(False)
        if (
            count != len(expected)
            or dict((key, value) for key, value, _, _ in fields).get("dflash.w1ax.config_hash")
            != recipe
        ):
            raise ValueError("GGUF recipe or tensor count differs")
        tensors: list[TensorInfo] = []
        for _ in range(count):
            name, dimensions = reader.text(), reader.uint(4)
            if not 1 <= dimensions <= 4:
                raise ValueError("unsupported GGUF tensor dimensions")
            shape = tuple(reader.uint(8) for _ in range(dimensions))
            tensors.append((name, shape, reader.uint(4), reader.uint(8)))
        if len({name for name, _, _, _ in tensors}) != count or {
            name for name, _, _, _ in tensors
        } != set(expected):
            raise ValueError("GGUF tensor coverage differs")
        beginning = (source.tell() + alignment - 1) // alignment * alignment
        intervals: list[tuple[int, int]] = []
        for name, shape, kind, offset in tensors:
            kind_name, wanted_shape, size, digest = expected[name]
            if (
                shape != wanted_shape
                or kind != {"F32": 0, "Q4_0": 2, "Q4_K": 12, "I32": 26}[kind_name]
            ):
                raise ValueError("GGUF tensor type or geometry differs")
            if not all(shape) or offset % alignment or beginning + offset + size > reader.size:
                raise ValueError("GGUF tensor offset exceeds bounds or alignment")
            block, width = (32, 18) if kind == 2 else (256, 144) if kind == 12 else (1, 4)
            if math.prod(shape) % block or math.prod(shape) // block * width != size:
                raise ValueError("GGUF tensor size differs")
            intervals.append((beginning + offset, beginning + offset + size))
            source.seek(beginning + offset)
            actual = hashlib.sha256()
            for start in range(0, size, 1024 * 1024):
                actual.update(reader.read(min(size - start, 1024 * 1024)))
            if actual.hexdigest() != digest:
                raise ValueError("GGUF tensor payload differs")
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("GGUF tensor payloads overlap")

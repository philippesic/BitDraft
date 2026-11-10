import json
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO, cast

from w1a1_eagle.contracts import Record, record, string


def unique(pairs: list[tuple[str, object]]) -> Record:
    result: Record = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value: str) -> object:
    raise ValueError("nonfinite JSON constant: " + value)


class Tokens:
    def __init__(self, source: TextIO, maximum: int) -> None:
        self.source = source
        self.maximum = maximum
        self.buffer = ""
        self.offset = 0
        self.eof = False
        self.decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid_constant)
        self.peak_bytes = 0

    def fill(self) -> None:
        self.buffer = self.buffer[self.offset :]
        self.offset = 0
        available = self.maximum - len(self.buffer.encode())
        if available <= 0:
            raise ValueError("JSON record exceeds streaming byte limit")
        chunk = self.source.read(min(8192, max(1, available // 4)))
        self.buffer += chunk
        self.eof = not chunk
        size = len(self.buffer.encode())
        self.peak_bytes = max(self.peak_bytes, size)
        if size > self.maximum:
            raise ValueError("JSON record exceeds streaming byte limit")

    def peek(self) -> str:
        while True:
            while self.offset < len(self.buffer) and self.buffer[self.offset].isspace():
                self.offset += 1
            if self.offset < len(self.buffer):
                return self.buffer[self.offset]
            if self.eof:
                return ""
            self.fill()

    def take(self, expected: str) -> None:
        if self.peek() != expected:
            raise ValueError("JSON framing expected " + expected)
        self.offset += 1

    def value(self) -> object:
        if not self.peek():
            raise ValueError("truncated JSON value")
        self.buffer = self.buffer[self.offset :]
        self.offset = 0
        while True:
            try:
                value, end = cast(tuple[object, int], self.decoder.raw_decode(self.buffer))
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError("invalid or truncated JSON value") from error
                self.fill()
                continue
            if end == len(self.buffer) and not self.eof:
                self.fill()
                continue
            self.offset = end
            return value


class ArrayRows:
    def __init__(self, path: Path, key: str, record_bytes: int = 1024 * 1024) -> None:
        if type(record_bytes) is not int or record_bytes < 8192:
            raise ValueError("streaming record bound must cover one read chunk")
        self.path = path
        self.key = key
        self.record_bytes = record_bytes
        self.header: Record = {}
        self.peak_buffer_bytes = 0
        self.complete = False

    def __iter__(self) -> Iterator[Record]:
        self.header = {}
        self.complete = False
        found = False
        keys: set[str] = set()
        with self.path.open(encoding="utf-8", newline="") as source:
            tokens = Tokens(source, self.record_bytes)
            tokens.take("{")
            first_key = True
            while tokens.peek() != "}":
                if not first_key:
                    tokens.take(",")
                first_key = False
                key = string(tokens.value())
                if key in keys:
                    raise ValueError("duplicate top-level key")
                keys.add(key)
                tokens.take(":")
                if key != self.key:
                    self.header[key] = tokens.value()
                    continue
                found = True
                tokens.take("[")
                first_row = True
                while tokens.peek() != "]":
                    if not first_row:
                        tokens.take(",")
                    first_row = False
                    yield record(tokens.value())
                tokens.take("]")
            tokens.take("}")
            if tokens.peek() or not found:
                raise ValueError("trailing JSON or missing row array")
            self.peak_buffer_bytes = tokens.peak_bytes
            self.complete = True

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterator, Protocol, cast

import pyarrow.parquet

from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.records import canonical


class Table(Protocol):
    def to_pylist(self) -> list[object]: ...


class Parquet(Protocol):
    def read_table(self, source: str) -> Table: ...


def parquet_rows(path: Path) -> Iterator[Record]:
    api = cast(Parquet, pyarrow.parquet)
    for value in api.read_table(str(path)).to_pylist():
        yield record(value)


def json_rows(path: Path) -> Iterator[Record]:
    with path.open() as source:
        for line in source:
            yield record(cast(object, json.loads(line)))


class TemplateRefused(ValueError):
    pass


class Tokenizer:
    def __init__(self, binary: Path, target: Path, log: Path) -> None:
        self.last_prompt: str | None = None
        self.last_tokens: tuple[int, ...] = ()
        self.log = log.open("w")
        self.process = subprocess.Popen(
            [str(binary), str(target)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            env=native_environment(binary.parent) | {"CUDA_VISIBLE_DEVICES": ""},
        )
        if self.process.stdout is None:
            self.close()
            raise ValueError("native vocabulary helper has no response pipe")
        ready = record(cast(object, json.loads(self.process.stdout.readline())))
        if ready.get("ready") is not True or ready.get("native_chat_template") is not True:
            self.close()
            raise ValueError(
                "rebuild the tokenizer helper with the authoritative native chat renderer"
            )
        self.template_sha256 = hashlib.sha256(string(ready["template"]).encode()).hexdigest()

    def query(self, payload: Record) -> Record:
        if self.process.stdin is None or self.process.stdout is None:
            raise ValueError("tokenizer pipes unavailable")
        encoded = canonical(payload)
        if len(encoded) > 2 * 1024 * 1024:
            raise TemplateRefused("prompt exceeds bounded native request protocol")
        self.process.stdin.write(encoded.decode() + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("native prompt helper exited; inspect tokenizer.log")
        reply = record(cast(object, json.loads(line)))
        if "error" in reply:
            raise TemplateRefused(string(reply["error"]))
        return reply

    def render(self, messages: list[Record], tools: list[Record]) -> str:
        payload = record(
            cast(object, json.loads(canonical({"messages": messages, "tools": tools})))
        )
        reply = self.query(payload)
        self.last_prompt = string(reply["prompt"])
        self.last_tokens = self.identifiers(reply)
        return self.last_prompt

    @staticmethod
    def identifiers(reply: Record) -> tuple[int, ...]:
        tokens = tuple(integer(v) for v in sequence(reply["tokens"]))
        if len(tokens) != integer(reply["n_tokens"]):
            raise ValueError("native tokenizer count disagreement")
        return tokens

    def tokenize(self, rendered: str) -> tuple[int, ...]:
        if rendered == self.last_prompt:
            return self.last_tokens
        return self.identifiers(self.query({"text": rendered}))

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.log.close()

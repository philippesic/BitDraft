import json
import subprocess
from pathlib import Path
from typing import Iterator, Protocol, cast

import gguf
import jinja2.sandbox
import pyarrow.parquet

from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.numerical.weights import GGUF
from w1a1_eagle.records import canonical, identity


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


class Tokenizer:
    def __init__(self, binary: Path, target: Path, log: Path) -> None:
        reader = cast(GGUF, gguf).GGUFReader(str(target))
        template = string(reader.fields["tokenizer.chat_template"].contents())
        self.template_sha256 = identity({"template": template})
        environment = jinja2.sandbox.SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = self.refuse
        self.template = environment.from_string(template)
        self.log = log.open("w")
        self.process = subprocess.Popen(
            [str(binary), str(target)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            env=native_environment(binary.parent) | {"CUDA_VISIBLE_DEVICES": ""},
        )
        if self.process.stdout is None or not record(
            cast(object, json.loads(self.process.stdout.readline()))
        ).get("ready"):
            self.close()
            raise ValueError("native vocabulary helper did not initialize")

    @staticmethod
    def refuse(message: str) -> str:
        raise ValueError(message)

    def render(self, messages: list[Record], tools: list[Record]) -> str:
        return self.template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token="",
            eos_token="<|im_end|>",
        )

    def tokenize(self, rendered: str) -> tuple[int, ...]:
        if self.process.stdin is None or self.process.stdout is None:
            raise ValueError("tokenizer pipes unavailable")
        self.process.stdin.write(canonical({"text": rendered}).decode() + "\n")
        self.process.stdin.flush()
        reply = record(cast(object, json.loads(self.process.stdout.readline())))
        tokens = tuple(integer(v) for v in sequence(reply["tokens"]))
        if len(tokens) != integer(reply["n_tokens"]):
            raise ValueError("native tokenizer count disagreement")
        return tokens

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.log.close()

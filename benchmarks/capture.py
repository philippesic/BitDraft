import json
import os
import selectors
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import IO, cast

from benchmarks.measurement import now
from benchmarks.native import file_reference
from benchmarks.protection import source_names
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact, canonical, identity, read


class Replies:
    def __init__(self, stream: IO[bytes]) -> None:
        self.stream = stream
        self.buffer = b""

    def receive(self, timeout: float) -> Record:
        deadline = now() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.stream, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = deadline - now()
                if remaining <= 0 or not bool(cast(object, selector.select(remaining))):
                    raise ValueError("native capture timed out; preserve partial attempt")
                block = os.read(self.stream.fileno(), 65536)
                if not block:
                    raise ValueError("native capture exited; inspect attempt log")
                self.buffer += block
                if len(self.buffer) > 2 * 1024 * 1024:
                    raise ValueError("capture reply exceeds bounded protocol")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return record(cast(object, json.loads(line)))


def capture(specification: Path, output: Path) -> Record:
    from w1a1_eagle.numerical.prompting import Tokenizer, json_rows

    raw = read(specification)
    if (
        set(raw) - {"role", "heldout_exclusions"}
        != {
            "schema",
            "source",
            "binary",
            "libraries",
            "target",
            "tokenizer",
            "bank",
            "ids",
            "feature_dtype",
            "context",
            "capture_limit_bytes",
            "exclusions",
            "window_policy",
        }
        or raw["schema"] != "native-hard-capture-v1"
    ):
        raise ValueError("unsupported native capture specification")
    root = specification.resolve().parent
    refs = {
        key: artifact(raw[key], root)
        for key in ("source", "binary", "target", "tokenizer", "bank", "exclusions")
    }
    libraries = tuple(artifact(value, root) for value in sequence(raw["libraries"]))
    for ref in (*refs.values(), *libraries):
        ref.verify()
    names = {ref.path.name.split(".so")[0]: ref for ref in libraries}
    if not {"libggml-cpu", "libggml-cuda"} <= set(names):
        raise ValueError("explicit CPU/CUDA backend identities required")
    selected_role = string(raw.get("role", "train"))
    if selected_role not in {"train", "dev"}:
        raise ValueError("capture role must explicitly remain train or dev")
    heldout: set[str] = {"heldout24"}
    if "heldout_exclusions" in raw:
        excluded = artifact(raw["heldout_exclusions"], root)
        excluded.verify()
        heldout.update(string(value) for value in sequence(read(excluded.path)["identities"]))
    selected_ids = tuple(string(value) for value in sequence(raw["ids"]))
    if (
        not selected_ids
        or len(set(selected_ids)) != len(selected_ids)
        or any(not key.replace("-", "").replace("_", "").isalnum() for key in selected_ids)
    ):
        raise ValueError("unique nonempty capture selection required")
    bank = {string(row["id"]): row for row in json_rows(refs["bank"].path)}
    if not set(selected_ids) <= set(bank):
        raise ValueError("capture selection missing from authenticated bank")
    dtype, policy = string(raw["feature_dtype"]), string(raw["window_policy"])
    context, limit = integer(raw["context"]), integer(raw["capture_limit_bytes"])
    if (
        dtype not in {"float16", "float32"}
        or policy not in {"full15", "up-to-15"}
        or not 1024 <= context <= 32768
        or limit < 1
    ):
        raise ValueError("invalid capture storage/context/objective bounds")
    protected = {
        "heldout24",
        *(string(v) for v in sequence(read(refs["exclusions"].path)["identities"])),
    }
    for key in selected_ids:
        row = bank[key]
        aliases = {
            key,
            string(row["source_key"]),
            string(row["group"]),
            *(string(value) for value in sequence(row.get("aliases", []))),
        }
        if (
            row["role"] != selected_role
            or aliases & heldout
            or (selected_role == "train" and aliases & protected)
        ):
            raise ValueError("capture selection includes protected or non-training source")
    forecast = sum(
        (integer(bank[key]["input_tokens"]) + integer(bank[key]["output_cap"]))
        * 2560
        * 8
        * (2 if dtype == "float16" else 4)
        for key in selected_ids
    )
    if forecast > limit:
        raise ValueError(
            "capture maximum payload exceeds declared budget; revise selection explicitly"
        )
    store = Store(output)
    config = identity(raw)
    prior: list[Record] = []
    if (store.root / "specification.json").exists():
        if identity(read(store.root / "specification.json")) != config:
            raise ValueError("cannot resume capture with a changed specification")
        prior = (
            [
                record(v)
                for v in sequence(read(store.root / "progress.json", 64 * 1024 * 1024)["records"])
            ]
            if (store.root / "progress.json").exists()
            else []
        )
        for row in prior:
            for value in (row["metadata"], *sequence(row["files"])):
                artifact(value, store.root).verify()
    else:
        store.atomic(store.root / "specification.json", canonical(raw))
    complete = {string(row["id"]) for row in prior}
    if not complete <= set(selected_ids):
        raise ValueError("capture progress contains unrelated records")
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    attempt_number = 0
    while (store.root / ("attempt-" + str(attempt_number))).exists():
        attempt_number += 1
    attempt = Store(store.root / ("attempt-" + str(attempt_number)))
    tokenizer = Tokenizer(
        refs["tokenizer"].path, refs["target"].path, attempt.root / "tokenizer.log"
    )
    argv = [
        str(refs["binary"].path),
        str(refs["target"].path),
        str(names["libggml-cpu"].path),
        str(names["libggml-cuda"].path),
        str(context),
    ]
    attempt.atomic(attempt.root / "command.json", canonical({"argv": argv, "code_sha256": code}))
    try:
        with device_lease("cuda"), (attempt.root / "capture.log").open("wb") as log:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=native_environment(refs["binary"].path.parent),
                start_new_session=True,
            )
            try:
                if process.stdin is None or process.stdout is None:
                    raise ValueError("capture process pipes unavailable")
                replies = Replies(process.stdout)
                if replies.receive(180).get("ready") is not True:
                    raise ValueError("capture target did not initialize")
                owner = ProcessIdentity.read(process.pid)
                attempt.atomic(attempt.root / "owner.json", canonical(cast(Record, asdict(owner))))
                for key in selected_ids:
                    if key in complete:
                        continue
                    row = bank[key]
                    rendered = tokenizer.render(
                        [record(v) for v in sequence(row["messages"])],
                        [record(v) for v in sequence(row["tools"])],
                    )
                    if source_names(row, rendered) & heldout or (
                        selected_role == "train" and source_names(row, rendered) & protected
                    ):
                        raise ValueError("rendered source matches a protected identity")
                    tokens = tokenizer.tokenize(rendered)
                    if (
                        len(tokens) != integer(row["input_tokens"])
                        or identity({"tokens": list(tokens)}) != row["token_sha256"]
                    ):
                        raise ValueError("capture tokenizer/template differs from frozen bank")
                    if len(tokens) + integer(row["output_cap"]) > context:
                        raise ValueError("capture exceeds context; truncation is prohibited")
                    destination = attempt.root / key
                    request: Record = {
                        "tokens": list(tokens),
                        "output_cap": row["output_cap"],
                        "output": str(destination),
                        "feature_dtype": dtype,
                    }
                    process.stdin.write(canonical(request) + b"\n")
                    process.stdin.flush()
                    response = replies.receive(1800)
                    if response.get("status") != "complete" or response.get("output") != str(
                        destination
                    ):
                        raise ValueError("native capture response disagrees with requested output")
                    cell = Store(destination)
                    metadata = read(cell.root / "capture.json")
                    if (
                        tuple(integer(v) for v in sequence(metadata["input_tokens"])[: len(tokens)])
                        != tokens
                        or metadata.get("feature_dtype") != dtype
                    ):
                        raise ValueError("capture prefix or storage precision mismatch")
                    cell.atomic(cell.root / "source.json", canonical(row))
                    derived = metadata | {
                        "window_policy": policy,
                        "raw_capture": file_reference(cell.root / "capture.json"),
                        "target_sha256": refs["target"].sha256,
                    }
                    cell.atomic(cell.root / "training-metadata.json", canonical(derived))
                    files = [
                        file_reference(path)
                        for path in sorted(cell.root.iterdir())
                        if path.is_file() and path.name != "training-metadata.json"
                    ]
                    result_row: Record = {
                        "id": key,
                        "source": row["source"],
                        "group": row["group"],
                        "topic": row["category"],
                        "aliases": row.get("aliases", []),
                        "role": selected_role,
                        "ancestry": "verified",
                        "prior_use": "historical-training"
                        if row["prior_use"] == "historical_train"
                        else "unknown",
                        "metadata": file_reference(cell.root / "training-metadata.json"),
                        "files": files,
                    }
                    prior.append(result_row)
                    complete.add(key)
                    store.atomic(store.root / "progress.json", canonical({"records": prior}))
                process.stdin.close()
                if process.wait(timeout=30) != 0:
                    raise ValueError("native capture did not exit cleanly")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
        manifest: Record = {
            "schema": 1,
            "target_sha256": refs["target"].sha256,
            "protected": sorted(protected),
            "expected_train": len(prior) if selected_role == "train" else 0,
            "records": prior,
        }
        store.atomic(store.root / "corpus.json", canonical(manifest))
        result: Record = {
            "status": "complete",
            "captures": len(prior),
            "role": selected_role,
            "dtype": dtype,
            "window_policy": policy,
            "maximum_feature_bytes": forecast,
            "corpus": file_reference(store.root / "corpus.json"),
            "code_sha256": code,
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        attempt.atomic(
            attempt.root / "failure.json", canonical({"error": str(error), "completed": len(prior)})
        )
        raise
    finally:
        tokenizer.close()

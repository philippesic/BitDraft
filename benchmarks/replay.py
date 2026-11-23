import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from benchmarks.capture import Replies
from benchmarks.native import file_reference
from benchmarks.token_streams import Spans, replay_spans, token_stream, verify_tokenization
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact, canonical, identity, read, reference

FIELDS = {
    "schema",
    "source",
    "binary",
    "libraries",
    "target",
    "ids",
    "feature_dtype",
    "context",
    "chunk",
    "capture_limit_bytes",
}


@dataclass(frozen=True)
class Stream:
    row: Record
    metadata: Artifact
    tokens: tuple[int, ...]
    spans: Spans
    prompt_tokens: int
    policy: str
    source: Record | None
    target: str


def located(
    corpora: tuple[Artifact, ...], selected: tuple[str, ...]
) -> tuple[dict[str, Stream], set[str]]:
    wanted = set(selected)
    found: dict[str, Stream] = {}
    protected_all: set[str] = {"heldout24"}
    for ref in corpora:
        ref.verify()
        corpus = read(ref.path, 64 * 1024 * 1024)
        if corpus.get("schema") != 1:
            raise ValueError("token replay reads eight-tap corpus-v1 manifests")
        protected = {"heldout24", *(string(v) for v in sequence(corpus["protected"]))}
        protected_all.update(protected)
        for value in sequence(corpus["records"]):
            row = record(value)
            name = string(row["id"])
            if name not in wanted:
                continue
            if name in found:
                raise ValueError("replay selection is ambiguous across corpora")
            if (
                row["role"] != "train"
                or {
                    name,
                    string(row["group"]),
                    *(string(v) for v in sequence(row["aliases"])),
                }
                & protected
            ):
                raise ValueError("token replay admits unprotected TRAIN records only")
            found[name] = stream(row, ref.path.parent, string(corpus["target_sha256"]))
    return found, protected_all


def banked(
    banks: tuple[Artifact, ...], selected: tuple[str, ...]
) -> tuple[dict[str, Stream], set[str]]:
    wanted = set(selected)
    found: dict[str, Stream] = {}
    protected_all: set[str] = {"heldout24"}
    for ref in banks:
        ref.verify()
        bank = read(ref.path, 64 * 1024 * 1024)
        if bank.get("schema") != "token-stream-bank-v1":
            raise ValueError("token replay reads token-stream-bank-v1 manifests")
        protected = {"heldout24", *(string(v) for v in sequence(bank["protected"]))}
        protected_all.update(protected)
        for value in sequence(bank["records"]):
            row = record(value)
            name = string(row["id"])
            if name not in wanted:
                continue
            if name in found:
                raise ValueError("replay selection is ambiguous across inputs")
            if (
                row["role"] != "train"
                or {
                    name,
                    string(row["group"]),
                    *(string(v) for v in sequence(row["aliases"])),
                }
                & protected
            ):
                raise ValueError("token replay admits unprotected TRAIN records only")
            found[name] = banked_stream(row, ref)
    return found, protected_all


def banked_stream(row: Record, bank: Artifact) -> Stream:
    value = record(row["stream"])
    tokens = tuple(integer(v) for v in sequence(value["tokens"]))
    spans = tuple(
        (integer(sequence(span)[0]), integer(sequence(span)[1]))
        for span in sequence(value["supervised_spans"])
    )
    if (
        value.get("schema") != "token-stream-v1"
        or value.get("id") != row["id"]
        or identity({"tokens": list(tokens)}) != value.get("tokens_sha256")
        or not spans
        or spans[-1][1] >= len(tokens)
    ):
        raise ValueError("token stream record identity mismatch")
    source = record(row["source_row"]) if row.get("source_row") is not None else None
    verify_tokenization(tokens, source)
    return Stream(
        {
            key: item
            for key, item in row.items()
            if key not in {"stream", "source_row", "window_policy"}
        }
        | {"metadata": reference(bank), "files": []},
        bank,
        tokens,
        spans,
        integer(value["prompt_tokens"]),
        string(row.get("window_policy", "full15")),
        source,
        string(value["target_sha256"]),
    )


def stream(row: Record, root: Path, target: str) -> Stream:
    metadata_ref = artifact(row["metadata"], root)
    metadata_ref.verify()
    metadata = read(metadata_ref.path)
    tokens, spans = replay_spans(metadata)
    source: Record | None = None
    for value in sequence(row["files"]):
        ref = artifact(value, root)
        if ref.path.name == "source.json":
            ref.verify()
            source = read(ref.path)
    verify_tokenization(tokens, source)
    return Stream(
        row,
        metadata_ref,
        tokens,
        spans,
        spans[0][0] + 1,
        string(metadata.get("window_policy", "full15")),
        source,
        string(metadata.get("target_sha256", target)),
    )


def request(item: Stream, destination: Path, dtype: str, chunk: int, soft: int = 0) -> Record:
    return {
        "tokens": list(item.tokens),
        "output_cap": 0,
        "output": str(destination),
        "feature_dtype": dtype,
        "spans": [list(span) for span in item.spans],
        "replay": {"chunk": chunk, **({"soft": soft} if soft else {})},
    }


def checked_metadata(
    item: Stream,
    metadata: Record,
    dtype: str,
    chunk: int,
    target: str,
    raw: Record,
    soft: int = 0,
) -> Record:
    positions = tuple(position for first, last in item.spans for position in range(first, last + 1))
    if (
        tuple(integer(v) for v in sequence(metadata["input_tokens"])) != item.tokens
        or tuple(integer(v) for v in sequence(metadata["logit_positions"])) != positions
        or metadata.get("feature_dtype") != dtype
        or metadata.get("supervised_spans") != [list(span) for span in item.spans]
        or record(metadata.get("replay", {})).get("chunk") != chunk
        or len(sequence(metadata["greedy_ids"])) != len(positions)
        or ("soft_targets" in metadata) != bool(soft)
        or (
            bool(soft)
            and (
                record(metadata["soft_targets"]).get("k") != soft
                or record(metadata["soft_targets"]).get("positions") != len(positions)
            )
        )
    ):
        raise ValueError("replayed tokens, positions or storage differ from the request")
    return metadata | {
        "window_policy": item.policy,
        "raw_capture": raw,
        "target_sha256": target,
        "token_replay": {
            "source_metadata": reference(item.metadata),
            "tokens_sha256": identity({"tokens": list(item.tokens)}),
        },
    }


def replay(specification: Path, output: Path) -> Record:
    from w1a1_eagle.numerical.captures import CaptureSpec, validate_capture

    raw = read(specification)
    if (
        set(raw) - {"corpora", "streams", "soft_targets"} != FIELDS
        or not {"corpora", "streams"} & set(raw)
        or raw["schema"] != "token-replay-v1"
    ):
        raise ValueError("unsupported token replay specification")
    root = specification.resolve().parent
    refs = {key: artifact(raw[key], root) for key in ("source", "binary", "target")}
    libraries = tuple(artifact(value, root) for value in sequence(raw["libraries"]))
    for ref in (*refs.values(), *libraries):
        ref.verify()
    names = {ref.path.name.split(".so")[0]: ref for ref in libraries}
    if not {"libggml-cpu", "libggml-cuda"} <= set(names):
        raise ValueError("explicit CPU/CUDA backend identities required")
    selected = tuple(string(value) for value in sequence(raw["ids"]))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("unique nonempty replay selection required")
    dtype, context = string(raw["feature_dtype"]), integer(raw["context"])
    chunk, limit = integer(raw["chunk"]), integer(raw["capture_limit_bytes"])
    soft = integer(raw.get("soft_targets", 0))
    if not 0 <= soft <= 64:
        raise ValueError("soft targets take 0 to 64 entries")
    if dtype not in {"float16", "float32"} or not 1024 <= context <= 32768 or not 1 <= chunk <= 512:
        raise ValueError("invalid replay storage, context or chunk")
    streams, protected = located(
        tuple(artifact(v, root) for v in sequence(raw.get("corpora", []))), selected
    )
    from_banks, bank_protected = banked(
        tuple(artifact(v, root) for v in sequence(raw.get("streams", []))), selected
    )
    if set(streams) & set(from_banks):
        raise ValueError("replay selection is ambiguous across inputs")
    streams.update(from_banks)
    protected.update(bank_protected)
    if set(streams) != set(selected):
        raise ValueError("replay selection missing from the authenticated inputs")
    if any(item.target != refs["target"].sha256 for item in streams.values()):
        raise ValueError("replay target differs from the records' capture target")
    forecast = sum(
        len(streams[key].tokens) * 2560 * 8 * (2 if dtype == "float16" else 4) for key in selected
    )
    if forecast > limit:
        raise ValueError("replay payload exceeds the declared bound")
    if any(len(item.tokens) > context for item in streams.values()):
        raise ValueError("replay exceeds context; truncation is prohibited")
    store = Store(output)
    prior: list[Record] = []
    if (store.root / "specification.json").exists():
        if identity(read(store.root / "specification.json")) != identity(raw):
            raise ValueError("cannot resume replay with a changed specification")
        if (store.root / "progress.json").exists():
            prior = [
                record(v)
                for v in sequence(read(store.root / "progress.json", 64 * 1024 * 1024)["records"])
            ]
        for row in prior:
            for value in (row["metadata"], *sequence(row["files"])):
                artifact(value, store.root).verify()
    else:
        store.atomic(store.root / "specification.json", canonical(raw))
    complete = {string(row["id"]) for row in prior}
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    attempt_number = 0
    while (store.root / ("attempt-" + str(attempt_number))).exists():
        attempt_number += 1
    attempt = Store(store.root / ("attempt-" + str(attempt_number)))
    argv = [
        str(refs["binary"].path),
        str(refs["target"].path),
        str(names["libggml-cpu"].path),
        str(names["libggml-cuda"].path),
        str(context),
        "replay",
    ]
    attempt.atomic(attempt.root / "command.json", canonical({"argv": argv, "code_sha256": code}))
    try:
        with device_lease("cuda"), (attempt.root / "replay.log").open("wb") as log:
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
                    raise ValueError("replay process pipes unavailable")
                replies = Replies(process.stdout)
                if replies.receive(180).get("ready") is not True:
                    raise ValueError("replay target did not initialize")
                owner = ProcessIdentity.read(process.pid)
                attempt.atomic(attempt.root / "owner.json", canonical(cast(Record, asdict(owner))))
                pending = [key for key in selected if key not in complete]
                stdin = process.stdin

                def send(key: str) -> None:
                    stdin.write(
                        canonical(request(streams[key], attempt.root / key, dtype, chunk, soft))
                        + b"\n"
                    )
                    stdin.flush()

                if pending:
                    send(pending[0])
                for position, key in enumerate(pending):
                    item = streams[key]
                    destination = attempt.root / key
                    response = replies.receive(1800)
                    if response.get("status") != "complete" or response.get("output") != str(
                        destination
                    ):
                        raise ValueError("native replay response disagrees with requested output")
                    if position + 1 < len(pending):
                        send(pending[position + 1])
                    cell = Store(destination)
                    derived = checked_metadata(
                        item,
                        read(cell.root / "capture.json"),
                        dtype,
                        chunk,
                        refs["target"].sha256,
                        file_reference(cell.root / "capture.json"),
                        soft,
                    )
                    if item.source is not None:
                        cell.atomic(cell.root / "source.json", canonical(item.source))
                    cell.atomic(
                        cell.root / "token-stream.json",
                        canonical(
                            token_stream(
                                key,
                                item.tokens,
                                item.spans,
                                item.prompt_tokens,
                                refs["target"].sha256,
                                item.source,
                                {
                                    "route": "token-replay-v1",
                                    "chunk": chunk,
                                    "source_metadata": reference(item.metadata),
                                },
                            )
                        ),
                    )
                    cell.atomic(cell.root / "training-metadata.json", canonical(derived))
                    files = [
                        file_reference(path)
                        for path in sorted(cell.root.iterdir())
                        if path.is_file() and path.name != "training-metadata.json"
                    ]
                    checked = validate_capture(
                        CaptureSpec(
                            key,
                            artifact(
                                file_reference(cell.root / "training-metadata.json"), cell.root
                            ),
                            tuple(artifact(value, cell.root) for value in files),
                        ),
                        8 * 1024**3,
                    )
                    cell.atomic(
                        cell.root / "validation.json", canonical({**checked, "code_sha256": code})
                    )
                    files.append(file_reference(cell.root / "validation.json"))
                    row = {
                        key: value
                        for key, value in item.row.items()
                        if key not in {"metadata", "files"}
                    }
                    prior.append(
                        row
                        | {
                            "metadata": file_reference(cell.root / "training-metadata.json"),
                            "files": files,
                        }
                    )
                    complete.add(key)
                    store.atomic(store.root / "progress.json", canonical({"records": prior}))
                process.stdin.close()
                if process.wait(timeout=30) != 0:
                    raise ValueError("native replay did not exit cleanly")
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
            "expected_train": len(prior),
            "records": prior,
        }
        store.atomic(store.root / "corpus.json", canonical(manifest))
        result: Record = {
            "status": "complete",
            "route": "token-replay-v1",
            "records": len(prior),
            "chunk": chunk,
            "dtype": dtype,
            "maximum_feature_bytes": forecast,
            "retokenised": False,
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

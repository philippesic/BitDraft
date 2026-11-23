import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from benchmarks.measurement import now, request
from benchmarks.native import file_reference
from benchmarks.runtime_policy import payload
from benchmarks.serving import native_server
from benchmarks.token_streams import token_stream
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.prompting import TemplateRefused, Tokenizer
from w1a1_eagle.records import artifact, canonical, identity, read

FIELDS = {
    "schema",
    "binary",
    "tokenizer",
    "target",
    "bank",
    "eog_ids",
    "role",
    "first",
    "count",
    "seed",
    "concurrency",
    "slot_context",
    "shard_records",
}
OPTIONAL = {"decode_consistent", "unified_cache", "server_context", "order_seed", "batch"}
BANDS = (2048, 8192)
PROVENANCE = (
    "source_key",
    "category",
    "subtype",
    "convention",
    "thinking",
    "sampling",
    "license",
    "repository",
    "revision",
)


@dataclass(frozen=True)
class Item:
    row: Record
    prompt: tuple[int, ...]
    cap: int
    temperature: float
    top_p: float
    seed: int


def band(row: Record) -> int:
    length = integer(row["input_tokens"]) + integer(row["output_cap"])
    return sum(length > edge for edge in BANDS)


def balanced(rows: list[Record], seed: int) -> list[Record]:
    strata: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        key = (string(row["category"]), band(row))
        strata.setdefault(key, []).append((identity({"id": row["id"], "seed": seed}), index))
    placed: list[tuple[float, str, int]] = []
    for members in strata.values():
        members.sort()
        offset = int(members[0][0][:8], 16) / 16**8
        for rank, (digest, index) in enumerate(members):
            placed.append(((rank + offset) / len(members), digest, index))
    return [rows[index] for _, _, index in sorted(placed)]


def bank_rows(bank: Artifact, role: str, seed: int, order_seed: int | None = None) -> list[Record]:
    rows: list[Record] = []
    with bank.path.open() as source:
        for line in source:
            row = record(cast(object, json.loads(line)))
            if row["role"] == role:
                rows.append(row)
    if order_seed is not None:
        return balanced(rows, order_seed)
    keyed = [(identity({"id": row["id"], "seed": seed}), index) for index, row in enumerate(rows)]
    return [rows[index] for _, index in sorted(keyed)]


def prepared(row: Record, tokenizer: Tokenizer, default_seed: int) -> Item:
    messages = [record(v) for v in sequence(row["messages"])]
    tools = [record(v) for v in sequence(row["tools"])]
    rendered = tokenizer.render(messages, tools, thinking=row["thinking"] is True)
    tokens = tokenizer.tokenize(rendered)
    if (
        identity({"tokens": list(tokens)}) != row["token_sha256"]
        or tokenizer.template_sha256 != row["template_sha256"]
    ):
        raise ValueError("bank prompt no longer tokenizes to its recorded tokens")
    sampling = row.get("sampling")
    if sampling is None:
        return Item(row, tokens, integer(row["output_cap"]), 0.0, 1.0, default_seed)
    chosen = record(sampling)
    return Item(
        row,
        tokens,
        integer(row["output_cap"]),
        number(chosen["temperature"]),
        number(chosen["top_p"]),
        integer(chosen["seed"]),
    )


def answered(response: Record, eog: frozenset[int]) -> tuple[tuple[int, ...], bool]:
    answer = tuple(integer(v) for v in sequence(response["tokens"]))
    if answer and answer[-1] in eog:
        return answer[:-1], True
    return answer, string(response.get("stop_type", "")) == "eos"


def stream_row(item: Item, response: Record, eog: frozenset[int], target: str) -> Record:
    answer, stopped = answered(response, eog)
    tokens = item.prompt + answer
    row = item.row
    source: Record = {
        "input_tokens": len(item.prompt),
        "token_sha256": row["token_sha256"],
        "template_sha256": row["template_sha256"],
        "thinking": row["thinking"] is True,
    }
    return {
        "id": row["id"],
        "source": row["source"],
        "group": row["group"],
        "topic": row["category"],
        "aliases": [
            alias
            for alias in sequence(row.get("aliases", []))
            if not string(alias).startswith("repository:")
        ],
        "role": row["role"],
        "ancestry": "verified",
        "prior_use": row.get("prior_use", "unknown"),
        "window_policy": "full15",
        "source_row": source,
        "stream": token_stream(
            string(row["id"]),
            tokens,
            ((len(item.prompt) - 1, len(tokens) - 1),),
            len(item.prompt),
            target,
            source,
            {
                "route": "bank-writing-v1",
                "stop_type": "eos" if stopped else "limit",
                "temperature": item.temperature,
                "top_p": item.top_p,
                "seed": item.seed,
            },
        ),
    }


def shard_matches(path: Path, expected: list[str]) -> bool:
    shard = read(path, 1024 * 1024 * 1024)
    written = [string(record(row)["id"]) for row in sequence(shard["records"])]
    skipped = {string(value) for value in sequence(shard["skipped"])}
    return skipped <= set(expected) and written == [
        value for value in expected if value not in skipped
    ]


def provenance(row: Record) -> Record:
    return {key: row[key] for key in PROVENANCE if key in row}


def write_bank(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if not FIELDS <= set(raw) <= FIELDS | OPTIONAL or raw["schema"] != "bank-writing-v1":
        raise ValueError("unsupported bank writing specification")
    root = specification.resolve().parent
    binary, helper = artifact(raw["binary"], root), artifact(raw["tokenizer"], root)
    target, bank = artifact(raw["target"], root), artifact(raw["bank"], root)
    for ref in (binary, helper, target, bank):
        ref.verify()
    role, seed = string(raw["role"]), integer(raw["seed"])
    first, count = integer(raw["first"]), integer(raw["count"])
    concurrency, slot_context = integer(raw["concurrency"]), integer(raw["slot_context"])
    shard_records = integer(raw["shard_records"])
    eog = frozenset(integer(v) for v in sequence(raw["eog_ids"]))
    if (
        role != "train"
        or first < 0
        or count < 1
        or not 1 <= concurrency <= 64
        or not 1024 <= slot_context <= 65536
        or not 1 <= shard_records <= 4096
        or not eog
    ):
        raise ValueError("bank writing generates bounded TRAIN slices only")
    order_seed = integer(raw["order_seed"]) if "order_seed" in raw else None
    rows = bank_rows(bank, role, seed, order_seed)[first : first + count]
    if len(rows) != count:
        raise ValueError("bank slice extends past the TRAIN rows")
    if output.exists() and read(output / "specification.json") != raw:
        raise ValueError("bank writing output belongs to a different specification")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    if not (store.root / "specification.json").exists():
        freeze_source(code_root, store.root / "sources", code, 0)
        store.atomic(store.root / "specification.json", canonical(raw))
    shards = Store(store.root / "shards")
    tokenizer = Tokenizer(helper.path, target.path, store.root / "tokenizer.log")
    measured: list[Record] = []
    try:
        with (
            device_lease("cuda"),
            native_server(
                binary,
                target,
                None,
                integer(raw.get("server_context", slot_context * concurrency)),
                1,
                Store(store.root / "server"),
                consistent=raw.get("decode_consistent", True) is True,
                parallel=concurrency,
                unified_cache=raw.get("unified_cache", False) is True,
                batch=integer(raw.get("batch", 512)),
            ) as port,
        ):
            pending: list[tuple[int, Path, list[Item], list[str]]] = []
            for index, start in enumerate(range(0, count, shard_records)):
                path = shards.root / ("streams-" + str(index).zfill(5) + ".json")
                if path.exists():
                    if not shard_matches(
                        path, [string(row["id"]) for row in rows[start : start + shard_records]]
                    ):
                        raise ValueError("existing shard differs from its bank slice: " + str(path))
                    continue
                items: list[Item] = []
                skipped: list[str] = []
                for row in rows[start : start + shard_records]:
                    try:
                        item = prepared(row, tokenizer, seed)
                    except TemplateRefused:
                        skipped.append(string(row["id"]))
                        continue
                    if len(item.prompt) + item.cap > slot_context:
                        skipped.append(string(row["id"]))
                        continue
                    items.append(item)
                pending.append((index, path, items, skipped))

            def run(item: Item) -> Record:
                return request(
                    port,
                    "completion",
                    payload(
                        list(item.prompt),
                        item.cap,
                        item.temperature,
                        item.top_p,
                        0,
                        0.0,
                        item.seed,
                    ),
                )

            with ThreadPoolExecutor(concurrency) as pool:
                futures = [[pool.submit(run, item) for item in items] for _, _, items, _ in pending]
                began = now()
                for (index, path, items, skipped), submitted in zip(pending, futures, strict=True):
                    responses = [future.result() for future in submitted]
                    finished = now()
                    wall = finished - began
                    began = finished
                    records = [
                        stream_row(item, response, eog, target.sha256)
                        for item, response in zip(items, responses, strict=True)
                    ]
                    generated = sum(
                        len(sequence(record(r["stream"])["tokens"])) - len(item.prompt)
                        for r, item in zip(records, items, strict=True)
                    )
                    shards.atomic(
                        path,
                        canonical(
                            {
                                "schema": "token-stream-bank-v1",
                                "target_sha256": target.sha256,
                                "protected": ["heldout24"],
                                "records": records,
                                "provenance": {
                                    string(item.row["id"]): provenance(item.row) for item in items
                                },
                                "skipped": skipped,
                                "wall_seconds": wall,
                                "generated_tokens": generated,
                                "code_sha256": code,
                            }
                        ),
                    )
                    measured.append(
                        {
                            "shard": index,
                            "records": len(records),
                            "skipped": len(skipped),
                            "generated_tokens": generated,
                            "wall_seconds": wall,
                            "generated_tokens_per_second": generated / wall if wall > 0 else 0.0,
                        }
                    )
    finally:
        tokenizer.close()
    files = sorted(shards.root.glob("streams-*.json"))
    result: Record = {
        "schema": "bank-writing-result-v1",
        "status": "complete" if len(files) == -(-count // shard_records) else "partial",
        "shards": [file_reference(path) for path in files],
        "this_session": measured,
        "rows_sha256": identity([string(row["id"]) for row in rows]),
        "row_ids": [string(row["id"]) for row in rows],
        "rule": "batched and sampled answers are a different data recipe from one-at-a-time greedy capture",
        "code_sha256": code,
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

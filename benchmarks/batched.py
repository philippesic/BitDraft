from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from benchmarks.measurement import now, request
from benchmarks.runtime_policy import payload
from benchmarks.serving import native_server
from benchmarks.token_streams import token_stream, verify_tokenization
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import artifact, canonical, identity, read, reference

FIELDS = {"schema", "binary", "target", "corpora", "ids", "concurrency", "slot_context", "seed"}
ROW_FIELDS = ("id", "source", "group", "topic", "aliases", "role", "ancestry", "prior_use")


@dataclass(frozen=True)
class Prompt:
    row: Record
    metadata: Artifact
    prompt: tuple[int, ...]
    reference: tuple[int, ...]
    stopped: bool
    cap: int
    source: Record | None
    target: str
    eog: frozenset[int]
    policy: str


def prompts(
    corpora: tuple[Artifact, ...], selected: tuple[str, ...]
) -> tuple[dict[str, Prompt], set[str]]:
    wanted = set(selected)
    found: dict[str, Prompt] = {}
    protected_all: set[str] = {"heldout24"}
    for ref in corpora:
        ref.verify()
        corpus = read(ref.path, 64 * 1024 * 1024)
        if corpus.get("schema") != 1:
            raise ValueError("batched writing reads corpus-v1 reference captures")
        protected = {"heldout24", *(string(v) for v in sequence(corpus["protected"]))}
        protected_all.update(protected)
        for value in sequence(corpus["records"]):
            row = record(value)
            name = string(row["id"])
            if name not in wanted:
                continue
            if name in found:
                raise ValueError("batched selection is ambiguous across corpora")
            names = {name, string(row["group"]), *(string(v) for v in sequence(row["aliases"]))}
            if row["role"] != "train" or names & protected:
                raise ValueError("batched writing admits unprotected TRAIN records only")
            root = ref.path.parent
            metadata_ref = artifact(row["metadata"], root)
            metadata_ref.verify()
            metadata = read(metadata_ref.path)
            if "supervised_spans" in metadata:
                raise ValueError("batched writing measures single-answer records")
            tokens = tuple(integer(v) for v in sequence(metadata["input_tokens"]))
            count = integer(metadata["prompt_tokens"])
            answer = tuple(integer(v) for v in sequence(metadata["generated_tokens"]))
            if tokens[count:] != answer or not answer:
                raise ValueError("reference answer differs from its recorded token stream")
            source: Record | None = None
            for item in sequence(row["files"]):
                file = artifact(item, root)
                if file.path.name == "source.json":
                    file.verify()
                    source = read(file.path)
            verify_tokenization(tokens, source)
            found[name] = Prompt(
                row,
                metadata_ref,
                tokens[:count],
                answer,
                metadata.get("stop_type") == "eos",
                integer(metadata["output_cap"]),
                source,
                string(metadata.get("target_sha256", corpus["target_sha256"])),
                frozenset(integer(v) for v in sequence(metadata["eog_ids"])),
                string(metadata.get("window_policy", "full15")),
            )
    if set(found) != wanted:
        raise ValueError("batched selection missing from the authenticated corpora")
    return found, protected_all


def answered(item: Prompt, response: Record) -> tuple[tuple[int, ...], bool]:
    answer = tuple(integer(v) for v in sequence(response["tokens"]))
    if answer and answer[-1] in item.eog:
        return answer[:-1], True
    return answer, string(response.get("stop_type", "")) == "eos"


def divergence(answer: tuple[int, ...], expected: tuple[int, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(answer, expected, strict=False)):
        if left != right:
            return index
    return None if len(answer) == len(expected) else min(len(answer), len(expected))


def answer_row(name: str, item: Prompt, response: Record, seconds: float) -> Record:
    answer, stopped = answered(item, response)
    first = divergence(answer, item.reference)
    return {
        "id": name,
        "prompt_tokens": len(item.prompt),
        "generated": len(answer),
        "reference_generated": len(item.reference),
        "first_divergence": first,
        "identical": first is None and stopped == item.stopped,
        "share_before_divergence": 1.0 if first is None else first / max(1, len(item.reference)),
        "stop_type": response.get("stop_type"),
        "request_seconds": seconds,
    }


def level(
    raw: Record,
    store: Store,
    items: dict[str, Prompt],
    order: tuple[str, ...],
    protected: set[str],
    concurrency: int,
    binary: Artifact,
    target: Artifact,
    answers: dict[str, tuple[int, ...]],
    single: dict[str, tuple[int, ...]] | None,
) -> Record:
    cell = Store(store.root / f"concurrency-{concurrency}")
    slot_context = integer(raw["slot_context"])
    seed = integer(raw["seed"])
    with native_server(
        binary, target, None, slot_context * concurrency, 1, cell, parallel=concurrency
    ) as port:
        warm = items[order[0]]
        request(port, "completion", payload(list(warm.prompt), 8, 0.0, 1.0, 0, 0.0, seed))

        def run(name: str) -> tuple[str, Record, float]:
            item = items[name]
            began = now()
            response = request(
                port,
                "completion",
                payload(list(item.prompt), item.cap, 0.0, 1.0, 0, 0.0, seed),
            )
            return name, response, now() - began

        began = now()
        with ThreadPoolExecutor(concurrency) as pool:
            completed = list(pool.map(run, order))
        wall = now() - began
    rows = []
    streams = []
    for name, response, seconds in completed:
        item = items[name]
        answer, _ = answered(item, response)
        row = answer_row(name, item, response, seconds)
        if single is not None:
            first = divergence(answer, single[name])
            row["first_divergence_from_single_slot"] = first
            row["identical_to_single_slot"] = first is None
        rows.append(row)
        answers[name] = answer
        tokens = item.prompt + answer
        streams.append(
            {key: item.row[key] for key in ROW_FIELDS if key in item.row}
            | {
                "window_policy": item.policy,
                "source_row": item.source,
                "stream": token_stream(
                    name,
                    tokens,
                    ((len(item.prompt) - 1, len(tokens) - 1),),
                    len(item.prompt),
                    target.sha256,
                    item.source,
                    {
                        "route": "batched-writing-v1",
                        "concurrency": concurrency,
                        "stop_type": response.get("stop_type"),
                        "reference_metadata": reference(item.metadata),
                    },
                ),
            }
        )
    cell.atomic(cell.root / "answers.json", canonical({"records": rows}))
    cell.atomic(
        cell.root / "streams.json",
        canonical(
            {
                "schema": "token-stream-bank-v1",
                "target_sha256": target.sha256,
                "protected": sorted(protected),
                "records": streams,
            }
        ),
    )
    total = sum(integer(row["generated"]) for row in rows)
    identical = sum(row["identical"] is True for row in rows)
    shares = [float(cast(float, row["share_before_divergence"])) for row in rows]
    return {
        "concurrency": concurrency,
        "answers": len(rows),
        "generated_tokens": total,
        "wall_seconds": wall,
        "generated_tokens_per_second": total / wall if wall > 0 else 0.0,
        "identical_answers": identical,
        "diverged_answers": len(rows) - identical,
        "mean_share_before_divergence": sum(shares) / len(shares),
        "minimum_share_before_divergence": min(shares),
        "streams": reference(Artifact.capture(cell.root / "streams.json")),
        "identical_to_single_slot": None
        if single is None
        else sum(row.get("identical_to_single_slot") is True for row in rows),
    }


def write_batched(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != FIELDS or raw["schema"] != "batched-writing-v1":
        raise ValueError("unsupported batched writing specification")
    root = specification.resolve().parent
    binary, target = artifact(raw["binary"], root), artifact(raw["target"], root)
    binary.verify()
    target.verify()
    order = tuple(string(v) for v in sequence(raw["ids"]))
    levels = tuple(integer(v) for v in sequence(raw["concurrency"]))
    if (
        not order
        or len(set(order)) != len(order)
        or not levels
        or len(set(levels)) != len(levels)
        or any(not 1 <= value <= 64 for value in levels)
        or not 1024 <= integer(raw["slot_context"]) <= 32768
    ):
        raise ValueError("unique prompts, concurrency 1 to 64 and a bounded slot context required")
    items, protected = prompts(tuple(artifact(v, root) for v in sequence(raw["corpora"])), order)
    if any(item.target != target.sha256 for item in items.values()):
        raise ValueError("batched writing target differs from the reference captures")
    if any(len(item.prompt) + item.cap > integer(raw["slot_context"]) for item in items.values()):
        raise ValueError("a prompt and its cap exceed the slot context")
    if output.exists():
        raise ValueError("batched writing requires a new output root")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    with device_lease("cuda"):
        measured = []
        single: dict[str, tuple[int, ...]] | None = None
        for value in sorted(levels):
            answers: dict[str, tuple[int, ...]] = {}
            measured.append(
                level(raw, store, items, order, protected, value, binary, target, answers, single)
            )
            if value == 1:
                single = answers
    result: Record = {
        "status": "measured",
        "levels": measured,
        "reference": "one-at-a-time native capture answers of the same TRAIN prompts",
        "rule": "diverged text is a different data recipe, not a faster equivalent",
        "prompts_sha256": identity(list(order)),
        "code_sha256": code,
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

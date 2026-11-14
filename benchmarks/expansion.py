import heapq
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

from benchmarks.coverage import length_band
from benchmarks.native import file_reference
from benchmarks.protection import source_names
from benchmarks.sandbox import TOOLS
from benchmarks.sources import math_problems, oasst2
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.prompting import TemplateRefused, Tokenizer, json_rows
from w1a1_eagle.records import artifact, canonical, identity, read, reference

ADAPTERS: dict[str, Callable[[Path], Iterator[Record]]] = {
    "math": math_problems,
    "oasst2": oasst2,
}
MAXIMUM_INPUT = 16384


def quota_key(row: Record) -> str:
    return "|".join(
        (string(row["source"]), string(row["subtype"]), str(integer(row["length_band"])))
    )


def matching(key: str, quotas: dict[str, int]) -> str | None:
    source, subtype, band = key.split("|")
    for candidate in (key, source + "|" + subtype + "|*"):
        if candidate in quotas:
            return candidate
    return None


def names(row: Record) -> set[str]:
    values = source_names(row)
    for key in ("rendered_sha256", "token_sha256", "prompt_prefix_sha256"):
        if key in row:
            values.add(string(row[key]))
    return values


def session_row(
    task: Record, transcript: Record, tokenizer: Tokenizer, end_token: int
) -> tuple[Record | None, Counter[str], str]:
    stats: Counter[str] = Counter()
    messages = [record(value) for value in sequence(transcript["messages"])]
    responses = [record(value) for value in sequence(transcript["responses"])]
    rendered = tokenizer.render(messages, TOOLS)
    tokens = tokenizer.tokenize(rendered)
    spans: list[list[int]] = []
    turn = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        response = responses[turn]
        turn += 1
        stats["assistant_turns"] += 1
        prefix = tokenizer.render(messages[:index], TOOLS)
        prompt = tokenizer.tokenize(prefix)
        if prefix != response["prompt"]:
            stats["skipped_server_prompt_differs"] += 1
            continue
        if tokens[: len(prompt)] != prompt:
            stats["skipped_history_rerendered"] += 1
            continue
        if end_token not in tokens[len(prompt) :]:
            stats["skipped_unterminated"] += 1
            continue
        end = tokens.index(end_token, len(prompt))
        if end + 1 > MAXIMUM_INPUT:
            stats["skipped_context_limit"] += 1
            break
        if end == len(prompt):
            stats["skipped_empty"] += 1
            continue
        generated = tuple(integer(value) for value in sequence(response["tokens"]))
        stats[
            "identical_turn_tokens"
            if tokens[len(prompt) : end] == generated
            else "retokenized_turns"
        ] += 1
        stats["generated_tokens"] += len(generated)
        spans.append([len(prompt) - 1, end - 1])
    if turn != len(responses):
        raise ValueError("session transcript and responses disagree")
    if not spans:
        return None, stats, rendered
    length = spans[-1][1] + 2
    kept = tokens[:length]
    stats["spans"] = len(spans)
    stats["span_positions"] = sum(last - first + 1 for first, last in spans)
    row: Record = {
        key: task[key]
        for key in (
            "id",
            "source",
            "source_key",
            "group",
            "aliases",
            "role",
            "category",
            "subtype",
            "license",
            "prior_use",
        )
    } | {
        "messages": messages,
        "tools": TOOLS,
        "input_tokens": length,
        "length_band": length_band(length),
        "token_sha256": identity({"tokens": list(kept)}),
        "rendered_sha256": identity({"text": rendered}),
        "prompt_prefix_sha256": identity({"tokens": list(kept[: spans[0][0] + 1])}),
        "template_sha256": tokenizer.template_sha256,
        "supervised_spans": spans,
        "session": dict(stats),
    }
    return row, stats, rendered


def expand(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw)
        != {
            "schema",
            "pool",
            "pool_result",
            "base_banks",
            "sessions",
            "sources",
            "tokenizer",
            "exclusions",
            "heldout_exclusions",
            "quotas",
            "output_cap",
            "maximum_feature_bytes",
            "seed",
        }
        or raw["schema"] != 1
        or output.exists()
    ):
        raise ValueError("explicit TRAIN expansion selection and a new output required")
    root = specification.resolve().parent
    pool, pool_result, binary = (
        artifact(raw[key], root) for key in ("pool", "pool_result", "tokenizer")
    )
    for ref in (pool, pool_result, binary):
        ref.verify()
    previous = read(pool_result.path)
    target = artifact(previous["target"], pool_result.path.parent)
    target.verify()
    protected: set[str] = {"heldout24"}
    heldout: set[str] = {"heldout24"}
    for key, destination in (("exclusions", protected), ("heldout_exclusions", heldout)):
        ref = artifact(raw[key], root)
        ref.verify()
        destination.update(string(value) for value in sequence(read(ref.path)["identities"]))
    protected |= heldout
    excluded: set[str] = set()
    base_refs = [artifact(value, root) for value in sequence(raw["base_banks"])]
    for ref in base_refs:
        ref.verify()
        for row in json_rows(ref.path):
            excluded |= names(row)
    quotas = {string(key): integer(value) for key, value in record(raw["quotas"]).items()}
    if not quotas or any(value < 1 for value in quotas.values()):
        raise ValueError("positive explicit expansion quotas required")
    seed = integer(raw["seed"])
    cap = integer(raw["output_cap"])
    if not 1 <= cap <= 2048:
        raise ValueError("capture output cap must stay within the native bound")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    heaps: dict[str, list[tuple[str, int, Record]]] = {key: [] for key in quotas}
    rejected: Counter[str] = Counter()
    order = 0

    def offer(row: Record) -> None:
        nonlocal order
        if row["role"] != "train":
            rejected["not_train"] += 1
            return
        if names(row) & protected:
            rejected["protected_identity"] += 1
            return
        if names(row) & excluded:
            rejected["base_bank_identity"] += 1
            return
        key = matching(quota_key(row), quotas)
        if key is None:
            return
        rank = identity({"seed": seed, "id": row["id"]})
        heap = heaps[key]
        order += 1
        if len(heap) < 2 * quotas[key] + 64:
            heapq.heappush(heap, (rank_inverse(rank), -order, row))
        elif rank_inverse(rank) > heap[0][0]:
            heapq.heapreplace(heap, (rank_inverse(rank), -order, row))

    tokenizer = Tokenizer(binary.path, target.path, store.root / "tokenizer.log")
    try:
        if tokenizer.template_sha256 != previous["template_sha256"]:
            raise ValueError("expansion tokenizer template differs from the curated pool")
        for row in json_rows(pool.path):
            offer(row)
        source_records: list[Record] = []
        for value in sequence(raw["sources"]):
            source = record(value)
            if set(source) != {"kind", "artifact", "repository", "revision", "license"}:
                raise ValueError("source requires kind, artifact, repository, revision and license")
            ref = artifact(source["artifact"], root)
            ref.verify()
            kind = string(source["kind"])
            if kind not in ADAPTERS:
                raise ValueError("unsupported expansion source")
            count = 0
            for row in ADAPTERS[kind](ref.path):
                try:
                    rendered = tokenizer.render(
                        [record(v) for v in sequence(row["messages"])],
                        [record(v) for v in sequence(row["tools"])],
                    )
                except TemplateRefused:
                    rejected["native_template_refused"] += 1
                    continue
                tokens = tokenizer.tokenize(rendered)
                if not 1 <= len(tokens) <= MAXIMUM_INPUT:
                    rejected["length"] += 1
                    continue
                group = string(row["group"])
                row.update(
                    {
                        "role": "dev"
                        if int(identity({"group": group})[:8], 16) % 5 == 0
                        else "train",
                        "input_tokens": len(tokens),
                        "length_band": length_band(len(tokens)),
                        "token_sha256": identity({"tokens": list(tokens)}),
                        "rendered_sha256": identity({"text": rendered}),
                        "template_sha256": tokenizer.template_sha256,
                        "source_license": source["license"],
                        "source_revision": source["revision"],
                    }
                )
                if source_names(row, rendered) & protected:
                    rejected["protected_identity"] += 1
                    continue
                offer(row)
                count += 1
            source_records.append({**source, "artifact": reference(ref), "candidates": count})
        end_token = tokenizer.identifiers(tokenizer.query({"text": "<|im_end|>"}))
        if len(end_token) != 1:
            raise ValueError("chat end-of-turn marker must be one token")
        session_stats: Counter[str] = Counter()
        session_records: list[Record] = []
        for value in sequence(raw["sessions"]):
            ref = artifact(value, root)
            ref.verify()
            result = read(ref.path)
            if result.get("status") != "complete" or result.get("role") != "train":
                raise ValueError("expansion sessions must be complete TRAIN generations")
            if result["target_sha256"] != target.sha256:
                raise ValueError("session target differs from the curated pool target")
            tasks_ref = artifact(result["tasks"], ref.path.parent)
            progress_ref = artifact(result["progress"], ref.path.parent)
            tasks_ref.verify()
            progress_ref.verify()
            tasks = {
                string(record(task)["id"]): record(task)
                for task in sequence(read(tasks_ref.path, 256 * 1024 * 1024)["tasks"])
            }
            converted = 0
            for entry in sequence(read(progress_ref.path, 64 * 1024 * 1024)["sessions"]):
                transcript_ref = artifact(record(entry)["transcript"], ref.path.parent)
                transcript_ref.verify()
                transcript = read(transcript_ref.path, 64 * 1024 * 1024)
                task = tasks[string(transcript["id"])]
                converted_row, stats, rendered = session_row(
                    task, transcript, tokenizer, end_token[0]
                )
                session_stats.update(stats)
                if converted_row is None:
                    rejected["session_without_spans"] += 1
                    continue
                if source_names(converted_row, rendered) & protected:
                    rejected["protected_identity"] += 1
                    continue
                offer(converted_row)
                converted += 1
            session_records.append({"result": reference(ref), "converted": converted})
        selected: list[Record] = []
        groups: set[str] = set()
        fingerprints: set[str] = set()
        shortfall: dict[str, int] = {}
        for key in sorted(quotas):
            ranked = sorted(heaps[key], reverse=True)
            chosen = 0
            for _, _, row in ranked:
                if chosen == quotas[key]:
                    break
                group = string(row["group"])
                fingerprint = string(row.get("prompt_prefix_sha256", row["token_sha256"]))
                if group in groups or fingerprint in fingerprints:
                    rejected["duplicate_group_or_prompt"] += 1
                    continue
                if "supervised_spans" not in row:
                    rendered = tokenizer.render(
                        [record(v) for v in sequence(row["messages"])],
                        [record(v) for v in sequence(row["tools"])],
                    )
                    tokens = tokenizer.tokenize(rendered)
                    if (
                        len(tokens) != integer(row["input_tokens"])
                        or identity({"tokens": list(tokens)}) != row["token_sha256"]
                    ):
                        raise ValueError("candidate tokenization differs from the pinned target")
                    if source_names(row, rendered) & protected:
                        rejected["protected_content_identity"] += 1
                        continue
                row = dict(row)
                row["output_cap"] = 0 if "supervised_spans" in row else cap
                groups.add(group)
                fingerprints.add(fingerprint)
                selected.append(row)
                chosen += 1
            if chosen < quotas[key]:
                shortfall[key] = quotas[key] - chosen
        if shortfall:
            raise ValueError(
                "expansion candidates cannot fill quotas: " + canonical(shortfall).decode()
            )
    finally:
        tokenizer.close()
    selected.sort(key=lambda row: string(row["id"]))
    maximum = sum(
        (integer(row["input_tokens"]) + integer(row["output_cap"])) * 2560 * 8 * 2
        for row in selected
    )
    if maximum > integer(raw["maximum_feature_bytes"]):
        raise ValueError("expansion maximum FP16 features exceed the declared storage ceiling")
    store.atomic(store.root / "prompts.jsonl", b"".join(canonical(row) + b"\n" for row in selected))
    store.atomic(store.root / "ids.json", canonical({"ids": [row["id"] for row in selected]}))

    def tally(field: Callable[[Record], str]) -> dict[str, int]:
        return dict(sorted(Counter(field(row) for row in selected).items()))

    result_record: Record = {
        "status": "selected",
        "role": "train",
        "prompts": len(selected),
        "target_sha256": target.sha256,
        "template_sha256": tokenizer.template_sha256,
        "categories": tally(lambda row: string(row["category"])),
        "strata": tally(quota_key),
        "sources": tally(lambda row: string(row["source"])),
        "length_bands": tally(lambda row: str(integer(row["length_band"]))),
        "input_tokens": sum(integer(row["input_tokens"]) for row in selected),
        "session_span_positions": sum(
            integer(record(row["session"])["span_positions"])
            for row in selected
            if "session" in row
        ),
        "maximum_feature_bytes": maximum,
        "output_cap": cap,
        "rejected": dict(rejected),
        "session_conversion": dict(session_stats),
        "source_records": source_records,
        "session_records": session_records,
        "base_banks": [reference(ref) for ref in base_refs],
        "pool": reference(pool),
        "pool_result": reference(pool_result),
        "tokenizer": reference(binary),
        "prompts_file": file_reference(store.root / "prompts.jsonl"),
        "ids": file_reference(store.root / "ids.json"),
        "code_sha256": code,
        "scope": "TRAIN-only expansion; exact base-bank, protected and heldout identities and token prefixes excluded; semantic review remains required",
    }
    store.atomic(store.root / "result.json", canonical(result_record))
    return result_record


def rank_inverse(rank: str) -> str:
    return "".join("0123456789abcdef"[15 - int(char, 16)] for char in rank)

import hashlib
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

from benchmarks.conventions import CONVENTIONS, apply
from benchmarks.long_projects import ProtectedCode
from benchmarks.native import file_reference
from benchmarks.retrain_sources import ADAPTERS
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.prompting import TemplateRefused, Tokenizer, json_rows
from w1a1_eagle.records import artifact, canonical, identity, read

FIELDS = {
    "schema",
    "target",
    "tokenizer",
    "seed",
    "sources",
    "exclusions",
    "gate",
    "dev_permille",
    "quotas",
    "dev_quotas",
    "thinking",
    "sampled",
    "output_caps",
    "conventions",
    "maximum_input_tokens",
}
SOURCE_FIELDS = {"kind", "artifact", "repository", "revision", "license", "limit"}


def text_of(row: Record) -> str:
    parts: list[str] = []
    for value in sequence(row["messages"]):
        content = record(value).get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def gate_index(paths: list[Path]) -> tuple[ProtectedCode, set[str]]:
    texts: list[str] = []
    for path in paths:
        texts.extend(string(row["text"]) for row in json_rows(path))
    return ProtectedCode(texts), {normalized(text) for text in texts}


def text_fingerprints(row: Record) -> set[str]:
    texts = [
        string(message["content"])
        for message in (record(value) for value in sequence(row["messages"]))
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    return {
        hashlib.sha256(variant.encode()).hexdigest()
        for text in texts
        for variant in (text, " ".join(text.split()))
    }


def normalized(text: str) -> str:
    return identity({"text": " ".join(text.lower().split())})


def bucket(value: object, seed: int) -> int:
    return int(identity({"value": value, "seed": seed})[:8], 16) % 1000


def cell_of(row: Record) -> str:
    return string(row["category"]) + "/" + string(row["subtype"])


def quota_cells(quotas: Record) -> dict[str, int]:
    return {string(key): integer(value) for key, value in quotas.items()}


def matching_quota(cell: str, quotas: dict[str, int]) -> str | None:
    if cell in quotas:
        return cell
    wildcard = cell.split("/", 1)[0] + "/*"
    return wildcard if wildcard in quotas else None


def finalize(
    row: Record,
    raw: Record,
    seed: int,
    tokenizer: Tokenizer,
    maximum: int,
) -> Record | None:
    category = string(row["category"])
    thinking_share = number(record(raw["thinking"]).get(category, 0))
    thinking = bucket(["thinking", row["id"]], seed) < 1000 * thinking_share
    caps = record(raw["output_caps"])
    cap = integer(caps.get(category + ("/thinking" if thinking else ""), caps["default"]))
    messages = [record(v) for v in sequence(row["messages"])]
    tools = [record(v) for v in sequence(row["tools"])]
    convention = "native"
    if tools:
        choices = [string(v) for v in sequence(raw["conventions"])]
        if not choices or any(choice not in CONVENTIONS or choice == "react" for choice in choices):
            raise ValueError("TRAIN conventions exclude the held-out ReAct convention")
        convention = choices[bucket(["convention", row["id"]], seed) % len(choices)]
        messages, tools = apply(convention, messages, tools)
    try:
        rendered = tokenizer.render(messages, tools, thinking=thinking)
        tokens = tokenizer.tokenize(rendered)
    except TemplateRefused:
        return None
    if not 1 <= len(tokens) <= maximum:
        return None
    sampled = record(raw["sampled"])
    sampling: Record | None = None
    if bucket(["sampled", row["id"]], seed) < 1000 * number(sampled["share"]):
        sampling = {
            "temperature": sampled["temperature"],
            "top_p": sampled["top_p"],
            "seed": bucket(["sampling-seed", row["id"]], seed),
        }
    result = dict(row) | {
        "messages": messages,
        "tools": tools,
        "convention": convention,
        "thinking": thinking,
        "output_cap": cap,
        "input_tokens": len(tokens),
        "token_sha256": identity({"tokens": list(tokens)}),
        "template_sha256": tokenizer.template_sha256,
    }
    if sampling is not None:
        result["sampling"] = sampling
    return result


def candidates(raw: Record, root: Path) -> Iterator[tuple[Record, Record]]:
    for value in sequence(raw["sources"]):
        source = record(value)
        if set(source) != SOURCE_FIELDS:
            raise ValueError("unsupported retrain source fields")
        kind = string(source["kind"])
        adapter: Callable[[Path], Iterator[Record]] | None = ADAPTERS.get(kind)
        if adapter is None:
            raise ValueError("unsupported retrain source kind: " + kind)
        ref = artifact(source["artifact"], root)
        ref.verify()
        limit = integer(source["limit"])
        for count, row in enumerate(adapter(ref.path)):
            if count >= limit:
                break
            yield source, row


def first(item: tuple[str, Record]) -> str:
    return item[0]


def select_retrain(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != FIELDS or raw["schema"] != "retrain-bank-v1":
        raise ValueError("unsupported retrain bank specification")
    if output.exists():
        raise ValueError("retrain bank output must be new")
    root = specification.resolve().parent
    target, binary = artifact(raw["target"], root), artifact(raw["tokenizer"], root)
    target.verify()
    binary.verify()
    seed = integer(raw["seed"])
    dev_permille = integer(raw["dev_permille"])
    maximum = integer(raw["maximum_input_tokens"])
    if not 0 < dev_permille < 500 or not 1024 <= maximum <= 32768:
        raise ValueError("invalid DEV share or input bound")
    protected: set[str] = {"heldout24"}
    for value in sequence(raw["exclusions"]):
        ref = artifact(value, root)
        ref.verify()
        protected.update(
            string(v) for v in sequence(read(ref.path, 256 * 1024 * 1024)["identities"])
        )
    gate_paths: list[Path] = []
    for value in sequence(raw["gate"]):
        ref = artifact(value, root)
        ref.verify()
        gate_paths.append(ref.path)
    gate, gate_exact = gate_index(gate_paths)
    quotas = {
        "train": quota_cells(record(raw["quotas"])),
        "dev": quota_cells(record(raw["dev_quotas"])),
    }
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    freeze_source(code_root, store.root / "sources", source_identity(code_root), 0)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    pools: dict[tuple[str, str], list[tuple[str, Record]]] = {}
    groups: set[str] = set()
    texts: set[str] = set()
    rejected: Counter[str] = Counter()
    offered: Counter[str] = Counter()
    available: Counter[str] = Counter()
    for source, row in candidates(raw, root):
        offered[string(source["kind"])] += 1
        names = {
            string(row["id"]),
            string(row["group"]),
            string(row["source_key"]),
            *(string(v) for v in sequence(row.get("aliases", []))),
            *text_fingerprints(row),
        }
        if names & protected:
            rejected["protected"] += 1
            continue
        group = string(row["group"])
        if group in groups:
            rejected["related_group"] += 1
            continue
        text = text_of(row)
        key = normalized(text)
        if key in texts:
            rejected["duplicate_text"] += 1
            continue
        if key in gate_exact or gate.matches(text):
            rejected["gate_overlap"] += 1
            continue
        split = string(row.get("split", group))
        role = "dev" if bucket(["role", split], seed) < dev_permille else "train"
        if row.get("inherited_role") == "dev":
            role = "dev"
        cell = matching_quota(cell_of(row), quotas[role])
        if cell is None:
            rejected["no_quota_" + role] += 1
            continue
        groups.add(group)
        texts.add(key)
        row = dict(row) | {
            "role": role,
            "license": source["license"],
            "repository": source["repository"],
            "revision": source["revision"],
        }
        pool = pools.setdefault((role, cell), [])
        pool.append((identity({"id": row["id"], "seed": seed}), row))
        available[role + ":" + cell] += 1
        bound = 2 * quotas[role][cell] + 100
        if len(pool) > 2 * bound:
            pool.sort(key=first)
            del pool[bound:]
    tokenizer = Tokenizer(binary.path, target.path, store.root / "tokenizer.log")
    selected: list[Record] = []
    filled: dict[str, Record] = {}
    try:
        for role in ("train", "dev"):
            for cell, quota in sorted(quotas[role].items()):
                pool = sorted(pools.get((role, cell), []), key=first)
                taken = 0
                for _, row in pool:
                    if taken >= quota:
                        break
                    final = finalize(row, raw, seed, tokenizer, maximum)
                    if final is None:
                        rejected["template_or_length"] += 1
                        continue
                    final["order"] = taken
                    selected.append(final)
                    taken += 1
                filled[role + ":" + cell] = {
                    "quota": quota,
                    "selected": taken,
                    "available": available[role + ":" + cell],
                }
    finally:
        tokenizer.close()
    bank = store.root / "bank.jsonl"
    with bank.open("wb") as sink:
        for row in selected:
            sink.write(canonical(row) + b"\n")
    summary: Record = {
        "schema": "retrain-bank-result-v1",
        "records": len(selected),
        "train": sum(row["role"] == "train" for row in selected),
        "dev": sum(row["role"] == "dev" for row in selected),
        "offered": dict(offered),
        "rejected": dict(rejected),
        "cells": filled,
        "thinking": sum(row["thinking"] is True for row in selected),
        "sampled": sum("sampling" in row for row in selected),
        "conventions": dict(Counter(string(row["convention"]) for row in selected)),
        "bank": file_reference(bank),
        "code_sha256": source_identity(code_root),
    }
    store.atomic(store.root / "result.json", canonical(summary))
    return summary

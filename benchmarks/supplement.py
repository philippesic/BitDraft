import json
from pathlib import Path
from typing import cast

from benchmarks.coverage import apportion
from benchmarks.native import file_reference
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.index import IndexReader
from w1a1_eagle.intent import Intent
from w1a1_eagle.records import artifact, canonical, identity, read


def stratum(row: Record) -> str:
    return string(row["category"]) + ":" + str(integer(row["length_band"]))


def cap(row: Record) -> int:
    return integer(row["output_cap"])


def supplement(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw)
        != {
            "schema",
            "bank",
            "legacy_intent",
            "quotas",
            "output_caps",
            "excluded_sources",
            "storage_bytes",
            "hidden",
            "taps",
        }
        or raw["schema"] != 1
        or output.exists()
    ):
        raise ValueError("explicit supplemental selection and a new output required")
    root = specification.resolve().parent
    bank, legacy = artifact(raw["bank"], root), artifact(raw["legacy_intent"], root)
    bank.verify()
    legacy.verify()
    intent = Intent.load(legacy.path)
    if intent.index_spec is None:
        raise ValueError("supplement requires the preserved legacy index for duplicate protection")
    dataset = IndexReader(intent.corpus, intent.index_spec, intent.memory_bytes).load()
    protected: set[str] = set(dataset.protected)
    for retained in dataset.rows:
        protected.update(
            (retained.prompt.identity, retained.prompt.group, *retained.prompt.aliases)
        )
        metadata = read(retained.capture.metadata.path)
        prefix = sequence(metadata["input_tokens"])[: integer(metadata["prompt_tokens"])]
        protected.add(identity({"tokens": prefix}))
    excluded = {string(value) for value in sequence(raw["excluded_sources"])}
    candidates: list[Record] = []
    rejected = 0
    for line in bank.path.read_bytes().splitlines():
        row = record(cast(object, json.loads(line)))
        if row["role"] != "train":
            raise ValueError("supplement must select from the admitted TRAIN bank")
        names = {
            string(row[key])
            for key in ("id", "group", "source_key", "rendered_sha256", "token_sha256")
        }
        if row["source"] in excluded or names & protected:
            rejected += 1
        else:
            candidates.append(row)
    demands = {name: integer(value) for name, value in record(raw["quotas"]).items()}
    caps = {int(name): integer(value) for name, value in record(raw["output_caps"]).items()}
    candidates = [row for row in candidates if stratum(row) in demands and cap(row) in caps]
    selected = apportion(candidates, demands, caps, stratum, cap)
    storage, hidden, taps = (integer(raw[key]) for key in ("storage_bytes", "hidden", "taps"))
    if storage not in {2, 4} or hidden != 2560 or taps != 8 or not selected:
        raise ValueError("explicit supported capture geometry required")
    feature_maximum = (
        sum(integer(row["input_tokens"]) + integer(row["output_cap"]) for row in selected)
        * hidden
        * taps
        * storage
    )
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "prompts.jsonl", b"".join(canonical(row) + b"\n" for row in selected))
    store.atomic(store.root / "ids.json", canonical({"ids": [row["id"] for row in selected]}))
    result: Record = {
        "status": "selected",
        "tasks": len(selected),
        "unique_source_groups": len({string(row["group"]) for row in selected}),
        "target_sha256": dataset.target,
        "input_tokens": sum(integer(row["input_tokens"]) for row in selected),
        "maximum_feature_bytes": feature_maximum,
        "storage_bytes": storage,
        "rejected_source_or_duplicate": rejected,
        "quotas": demands,
        "output_caps": raw["output_caps"],
        "bank": file_reference(bank.path),
        "legacy": file_reference(legacy.path),
        "scope": "new-source focused subset of the admitted bank; exact old aliases and tokenized prefixes excluded; semantic review remains required",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

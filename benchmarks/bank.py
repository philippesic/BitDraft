from collections import Counter
from pathlib import Path
from typing import Iterator

from benchmarks.coverage import FULL, SHORT, TRAIN, identifier, length_band, select
from benchmarks.sources import commitpack, glaive, legacy, qasper, ultrachat
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.numerical.prompting import Tokenizer
from w1a1_eagle.records import artifact, canonical, identity, read, reference


def input_rows(raw: Record, root: Path) -> Iterator[Record]:
    for value in sequence(raw["sources"]):
        source = record(value)
        ref = artifact(source["artifact"], root)
        ref.verify()
        kind = string(source["kind"])
        adapters = {
            "ultrachat": ultrachat,
            "glaive": glaive,
            "qasper": qasper,
            "commitpack": commitpack,
        }
        if kind not in adapters:
            raise ValueError("unsupported curated source")
        yield from adapters[kind](ref.path)
    legacy_manifest = artifact(raw["legacy_manifest"], root)
    legacy_manifest.verify()
    manifest = read(legacy_manifest.path)
    train = record(record(manifest["files"])["train"])
    for value in sequence(train["shards"]):
        shard = record(value)
        for key in ("prompts", "index"):
            path = legacy_manifest.path.parent / string(shard[key])
            artifact(
                {"path": str(path), "sha256": shard[key + "_sha256"], "bytes": path.stat().st_size},
                root,
            ).verify()
    yield from legacy(legacy_manifest.path.parent)


def curate(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) != {"schema", "sources", "legacy_manifest", "target", "tokenizer"}
        or raw["schema"] != 1
    ):
        raise ValueError("unsupported prompt bank source specification")
    if output.exists():
        raise ValueError("curation output must be new")
    root = specification.resolve().parent
    target, binary = artifact(raw["target"], root), artifact(raw["tokenizer"], root)
    target.verify()
    binary.verify()
    store = Store(output)
    store.atomic(store.root / "sources.json", specification.read_bytes())
    tokenizer = Tokenizer(binary.path, target.path, store.root / "tokenizer.log")
    pools: dict[str, list[Record]] = {"train": [], "dev": []}
    groups: set[str] = set()
    fingerprints: set[str] = set()
    rejected: Counter[str] = Counter()
    try:
        for row in input_rows(raw, root):
            group = string(row["group"])
            if group in groups:
                rejected["related_group"] += 1
                continue
            rendered = tokenizer.render(
                [record(v) for v in sequence(row["messages"])],
                [record(v) for v in sequence(row["tools"])],
            )
            tokens = tokenizer.tokenize(rendered)
            if not 1 <= len(tokens) <= 16384:
                rejected["length"] += 1
                continue
            fingerprint = identity({"tokens": list(tokens)})
            if fingerprint in fingerprints:
                rejected["token_duplicate"] += 1
                continue
            groups.add(group)
            fingerprints.add(fingerprint)
            role = (
                "train"
                if row.get("inherited_role") == "train"
                or int(identity({"group": group})[:8], 16) % 5
                else "dev"
            )
            row.update(
                {
                    "role": role,
                    "input_tokens": len(tokens),
                    "length_band": length_band(len(tokens)),
                    "token_sha256": fingerprint,
                    "rendered_sha256": identity({"text": rendered}),
                    "template_sha256": tokenizer.template_sha256,
                }
            )
            pools[role].append(row)
        store.atomic(
            store.root / "pool.jsonl",
            b"".join(canonical(r) + b"\n" for role in pools.values() for r in role),
        )
        selected = {
            "train": select(pools["train"], TRAIN, (3720, 1800, 420, 60)),
            "dev": select(pools["dev"], FULL, (300, 180, 90, 30)),
        }
        short = select(selected["dev"], SHORT, (60, 36, 18, 6))
        summaries: Record = {}
        for role, rows in selected.items():
            for index, row in enumerate(sorted(rows, key=identifier)):
                row["output_cap"] = 256 if index % 10 < 3 else 1024 if index % 10 < 8 else 2048
            store.atomic(
                store.root / (role + ".jsonl"), b"".join(canonical(r) + b"\n" for r in rows)
            )
            summaries[role] = {
                "count": len(rows),
                "categories": dict(Counter(string(r["category"]) for r in rows)),
                "length_bands": dict(Counter(str(integer(r["length_band"])) for r in rows)),
                "source_counts": dict(Counter(string(r["source"]) for r in rows)),
                "input_tokens": sum(integer(r["input_tokens"]) for r in rows),
            }
        short_ids = {string(r["id"]) for r in short}
        for name, rows in (
            ("full", selected["dev"]),
            ("short", [r for r in selected["dev"] if string(r["id"]) in short_ids]),
        ):
            tasks = [
                {
                    key: row[key]
                    for key in (
                        "id",
                        "group",
                        "role",
                        "category",
                        "messages",
                        "output_cap",
                        "tools",
                    )
                }
                | {
                    "ancestry": {
                        key: row[key]
                        for key in (
                            "source",
                            "source_key",
                            "prior_use",
                            "token_sha256",
                            "input_tokens",
                        )
                    }
                }
                for row in rows
            ]
            store.atomic(
                store.root / (name + "-workload.json"), canonical({"schema": 1, "tasks": tasks})
            )
        result: Record = {
            "status": "curated",
            "schema": 1,
            "target": reference(target),
            "tokenizer": reference(binary),
            "template_sha256": tokenizer.template_sha256,
            "coverage": summaries,
            "short_ids": sorted(short_ids),
            "rejected": dict(rejected),
            "limitations": "Public source exposure unknown; heuristics are acquisition tags, not a semantic difficulty or real-world validity proof. Review examples before capture.",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    finally:
        tokenizer.close()

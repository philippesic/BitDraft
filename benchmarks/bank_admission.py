from collections import Counter
from itertools import chain
from pathlib import Path

from benchmarks.bank import publish
from benchmarks.coverage import (
    FULL,
    SHORT,
    TRAIN,
    TRAIN_INPUT_BUDGET,
    length_band,
    select,
    selection_order,
)
from benchmarks.protection import source_names
from benchmarks.sources import ultrachat_label
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.prompting import Tokenizer, json_rows
from w1a1_eagle.records import artifact, canonical, identity, read, reference


def admit(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) - {"heldout_exclusions", "development_replays", "harnesses"}
        != {"schema", "pool", "source_result", "exclusions"}
        or raw["schema"] != 1
    ):
        raise ValueError("unsupported bank admission specification")
    refs = {
        key: artifact(raw[key], specification.resolve().parent)
        for key in ("pool", "source_result", "exclusions")
    }
    for ref in refs.values():
        ref.verify()
    previous = read(refs["source_result"].path)
    target = artifact(previous["target"], refs["source_result"].path.parent)
    binary = artifact(previous["tokenizer"], refs["source_result"].path.parent)
    target.verify()
    binary.verify()
    protected = {
        "heldout24",
        *(string(v) for v in sequence(read(refs["exclusions"].path)["identities"])),
    }
    heldout: set[str] = {"heldout24"}
    if "heldout_exclusions" in raw:
        ref = artifact(raw["heldout_exclusions"], specification.resolve().parent)
        ref.verify()
        heldout.update(string(v) for v in sequence(read(ref.path)["identities"]))
    pools: dict[str, list[Record]] = {"train": [], "dev": []}
    groups: dict[str, str] = {}
    rejected: Counter[str] = Counter()
    replays: list[Record] = []
    if "development_replays" in raw:
        ref = artifact(raw["development_replays"], specification.resolve().parent)
        ref.verify()
        replay_receipt = read(ref.path)
        if (
            replay_receipt["target_sha256"] != target.sha256
            or replay_receipt["template_sha256"] != previous["template_sha256"]
        ):
            raise ValueError("application replay target/template mismatch")
        pool_ref = artifact(replay_receipt["pool"], ref.path.parent)
        pool_ref.verify()
        replays = list(json_rows(pool_ref.path))
        if any(row["role"] != "dev" or row["source"] != "project-replay" for row in replays):
            raise ValueError("application replays must remain DEV")
    for row in chain(json_rows(refs["pool"].path), replays):
        if row["source"] == "ultrachat" and row["category"] != "conversation":
            first = record(sequence(row["messages"])[0])
            if ultrachat_label(string(first["content"])) != (row["category"], row["subtype"]):
                rejected["source_annotation_refused"] += 1
                continue
        role, group = string(row["role"]), string(row["group"])
        if role not in pools:
            raise ValueError("bank role must remain train/dev")
        if group in groups:
            raise ValueError("provisional bank repeats a source group")
        groups[group] = role
        if source_names(row) & heldout:
            rejected["heldout_identity"] += 1
            continue
        if role == "train" and source_names(row) & protected:
            rejected["protected_identity"] += 1
            continue
        pools[role].append(row)
    if output.exists():
        raise ValueError("bank admission output must be new")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    tokenizer = Tokenizer(binary.path, target.path, store.root / "tokenizer.log")
    try:
        if "harnesses" in raw:
            ref = artifact(raw["harnesses"], specification.resolve().parent)
            ref.verify()
            system = string(read(ref.path)["tools"])
            for category, source in (("code", "commitpack"), ("structured", "glaive")):
                eligible = sorted(
                    (
                        row
                        for row in pools["train"]
                        if row["category"] == category and row["source"] == source
                    ),
                    key=selection_order,
                )[:100]
                if len(eligible) != 100:
                    raise ValueError(
                        "training application prefix requires 100 code and 100 tool groups"
                    )
                for row in eligible:
                    messages = [{"role": "system", "content": system}, *sequence(row["messages"])]
                    rendered = tokenizer.render(
                        [record(value) for value in messages],
                        [record(value) for value in sequence(row["tools"])],
                    )
                    tokens = tokenizer.tokenize(rendered)
                    row.update(
                        {
                            "messages": messages,
                            "harness": "project-training",
                            "input_tokens": len(tokens),
                            "length_band": length_band(len(tokens)),
                            "token_sha256": identity({"tokens": list(tokens)}),
                            "rendered_sha256": identity({"text": rendered}),
                        }
                    )
        for role, counts, bands in (
            ("train", TRAIN, (3720, 1800, 420, 60)),
            ("dev", FULL, (300, 180, 90, 30)),
        ):
            while True:
                selected = select(
                    pools[role], counts, bands, TRAIN_INPUT_BUDGET if role == "train" else None
                )
                refused: set[str] = set()
                for row in selected:
                    rendered = tokenizer.render(
                        [record(v) for v in sequence(row["messages"])],
                        [record(v) for v in sequence(row["tools"])],
                    )
                    tokens = tokenizer.tokenize(rendered)
                    if (
                        len(tokens) != integer(row["input_tokens"])
                        or identity({"tokens": list(tokens)}) != row["token_sha256"]
                    ):
                        raise ValueError(
                            "provisional tokenization differs from pinned native target"
                        )
                    if source_names(row, rendered) & heldout or (
                        role == "train" and source_names(row, rendered) & protected
                    ):
                        refused.add(string(row["id"]))
                if not refused:
                    break
                rejected["protected_content_identity"] += len(refused)
                pools[role] = [row for row in pools[role] if string(row["id"]) not in refused]
        select(select(pools["dev"], FULL, (300, 180, 90, 30)), SHORT, (60, 36, 18, 6))
        result = publish(
            pools, store, target, binary, tokenizer.template_sha256, rejected, "admitted"
        )
        receipt: Record = {
            "status": "admitted",
            "source_sha256": code,
            "input_pool": reference(refs["pool"]),
            "source_result": reference(refs["source_result"]),
            "exclusions": reference(refs["exclusions"]),
            "complete_selected_token_reverification": True,
            "roles_changed": False,
            "training_protected_aliases_removed": sum(rejected.values()),
            "limits": "Known identity/normalized-text checks do not prove absence of semantic overlap or unknown pretraining exposure.",
        }
        store.atomic(store.root / "admission.json", canonical(receipt))
        return result
    finally:
        tokenizer.close()

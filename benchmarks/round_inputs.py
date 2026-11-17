import hashlib
import json
from collections import Counter, deque
from pathlib import Path
from typing import cast

from benchmarks.coverage import length_band
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import artifact, canonical, read, reference

SCREEN_CAP = 128
CATEGORIES = ("code", "reasoning", "documents", "conversation", "structured", "writing")
BANDS = (0, 1, 2, 3)
FIELDS = {
    "schema",
    "screen",
    "dev_bank",
    "capture",
    "train_bank",
    "train_corpus",
    "exclusions",
    "establish",
    "seed",
}

Cell = tuple[str, int]


def bank_rows(source: Artifact) -> dict[str, Record]:
    source.verify()
    rows: dict[str, Record] = {}
    with source.path.open() as lines:
        for line in lines:
            row = record(cast(object, json.loads(line)))
            identity = string(row["id"])
            if identity in rows:
                raise ValueError("duplicate bank identity")
            rows[identity] = row
    return rows


def aliases(row: Record) -> set[str]:
    return {
        string(row["id"]),
        string(row["group"]),
        string(row.get("source_key", row["id"])),
        *(string(value) for value in sequence(row.get("aliases", []))),
    }


def screen_bank(tasks: list[Record], dev: dict[str, Record], protected: set[str]) -> list[Record]:
    rows: list[Record] = []
    for task in tasks:
        identity = string(task["id"])
        row = dev.get(identity)
        ancestry = record(task["ancestry"])
        if (
            row is None
            or row["role"] != "dev"
            or task["role"] != "dev"
            or row.get("token_sha256") != ancestry.get("token_sha256")
            or row.get("input_tokens") != ancestry.get("input_tokens")
            or row["messages"] != task["messages"]
            or aliases(row) & protected
        ):
            raise ValueError("screen request differs from its DEV bank row: " + identity)
        rows.append({**row, "output_cap": SCREEN_CAP})
    return rows


def augment(
    plan: dict[Cell, int],
    available: dict[Cell, int],
    category_quota: dict[str, int],
    band_quota: dict[int, int],
) -> bool:
    deficits = [c for c in CATEGORIES if sum(plan[(c, b)] for b in BANDS) < category_quota[c]]
    if not deficits:
        return False
    start = deficits[0]
    parents: dict[str | int, str | int] = {}
    queue: deque[str | int] = deque([start])
    seen: set[str | int] = {start}
    found: int | None = None
    while queue and found is None:
        node = queue.popleft()
        if isinstance(node, str):
            for band in BANDS:
                if band not in seen and plan[(node, band)] < available[(node, band)]:
                    seen.add(band)
                    parents[band] = node
                    if sum(plan[(c, band)] for c in CATEGORIES) < band_quota[band]:
                        found = band
                        break
                    queue.append(band)
        else:
            for category in CATEGORIES:
                if category not in seen and plan[(category, node)] > 0:
                    seen.add(category)
                    parents[category] = node
                    queue.append(category)
    if found is None:
        raise ValueError("TRAIN bank cannot meet the screen's category and length marginals")
    step: str | int = found
    while step != start:
        parent = parents[step]
        if isinstance(step, int):
            plan[(cast(str, parent), step)] += 1
        else:
            plan[(step, cast(int, parent))] -= 1
        step = parent
    return True


def marginal_plan(screen: Counter[Cell], available: dict[Cell, int]) -> dict[Cell, int]:
    category_quota = {c: sum(screen[(c, b)] for b in BANDS) for c in CATEGORIES}
    band_quota = {b: sum(screen[(c, b)] for c in CATEGORIES) for b in BANDS}
    plan = {(c, b): min(screen[(c, b)], available[(c, b)]) for c in CATEGORIES for b in BANDS}
    while augment(plan, available, category_quota, band_quota):
        pass
    if any(sum(plan[(c, b)] for c in CATEGORIES) != band_quota[b] for b in BANDS):
        raise ValueError("TRAIN plan misses the screen's length marginals")
    return plan


def order(seed: int, row: Record) -> str:
    text = f"seen-screen-v1:{seed}:{string(row['group'])}:{string(row['id'])}"
    return hashlib.sha256(text.encode()).hexdigest()


def seen_selection(
    screen: Counter[Cell], train: list[Record], seed: int
) -> tuple[list[Record], dict[Cell, int]]:
    cells: dict[Cell, list[Record]] = {(c, b): [] for c in CATEGORIES for b in BANDS}
    for row in train:
        key = (string(row["category"]), length_band(integer(row["input_tokens"])))
        if key in cells:
            cells[key].append(row)
    for rows in cells.values():
        rows.sort(key=lambda row: order(seed, row))
    available = {key: len({string(row["group"]) for row in rows}) for key, rows in cells.items()}
    plan = marginal_plan(screen, available)
    used: set[str] = set()
    selected: list[Record] = []
    for key in sorted(plan):
        wanted = plan[key]
        for row in cells[key]:
            if wanted == 0:
                break
            if string(row["group"]) in used:
                continue
            used.add(string(row["group"]))
            selected.append(row)
            wanted -= 1
        if wanted:
            raise ValueError("TRAIN cell has too few independent groups: " + str(key))
    return selected, plan


def workload_task(row: Record) -> Record:
    task: Record = {
        "id": row["id"],
        "group": row["group"],
        "role": "train",
        "category": row["category"],
        "messages": row["messages"],
        "tools": row.get("tools", []),
        "output_cap": SCREEN_CAP,
        "ancestry": {
            "input_tokens": row["input_tokens"],
            "prior_use": row.get("prior_use", "unknown"),
            "source": row["source"],
            "source_key": row["source_key"],
            "token_sha256": row["token_sha256"],
        },
    }
    return task


def prepare(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != FIELDS or raw["schema"] != "round-inputs-v1":
        raise ValueError("unsupported round-inputs specification")
    if output.exists():
        raise ValueError("round-inputs output must be new")
    root = specification.resolve().parent
    seed = integer(raw["seed"])
    protected = {"heldout24"}
    exclusions = [artifact(value, root) for value in sequence(raw["exclusions"])]
    for exclusion in exclusions:
        exclusion.verify()
        protected.update(string(value) for value in sequence(read(exclusion.path)["identities"]))
    screen_ref = artifact(raw["screen"], root)
    screen_ref.verify()
    screen_spec = read(screen_ref.path)
    if screen_spec["tier"] != "screen" or screen_spec["role"] != "dev":
        raise ValueError("the screen specification must be the DEV training screen")
    workload = artifact(screen_spec["workload"], screen_ref.path.parent)
    workload.verify()
    tasks = [record(value) for value in sequence(read(workload.path)["tasks"])]
    dev = bank_rows(artifact(raw["dev_bank"], root))
    capture_rows = screen_bank(tasks, dev, protected)
    template_ref = artifact(raw["capture"], root)
    template_ref.verify()
    template = read(template_ref.path)
    if template["schema"] != "native-hard-capture-v1":
        raise ValueError("capture template must be a native capture specification")
    store = Store(output)
    bank_path = store.root / "screen-capture-bank.jsonl"
    store.atomic(bank_path, "".join(json.dumps(row) + "\n" for row in capture_rows).encode())
    template_root = template_ref.path.parent
    capture_spec: Record = {
        **{
            key: reference(artifact(template[key], template_root))
            for key in ("source", "binary", "target", "tokenizer", "exclusions")
        },
        "libraries": [
            reference(artifact(value, template_root)) for value in sequence(template["libraries"])
        ],
        "heldout_exclusions": reference(
            artifact(template.get("heldout_exclusions", template["exclusions"]), template_root)
        ),
        "schema": "native-hard-capture-v1",
        "bank": reference(Artifact.capture(bank_path)),
        "ids": [row["id"] for row in capture_rows],
        "feature_dtype": "float16",
        "context": template["context"],
        "capture_limit_bytes": template["capture_limit_bytes"],
        "role": "dev",
        "window_policy": "up-to-15",
    }
    store.atomic(store.root / "screen-capture.json", canonical(capture_spec))
    screen_cells: Counter[Cell] = Counter(
        (string(task["category"]), length_band(integer(record(task["ancestry"])["input_tokens"])))
        for task in tasks
    )
    corpus_ref = artifact(raw["train_corpus"], root)
    corpus_ref.verify()
    seen_ids = {
        string(record(value)["id"])
        for value in sequence(read(corpus_ref.path, 64 * 1024 * 1024)["records"])
    }
    train_rows = bank_rows(artifact(raw["train_bank"], root))
    eligible = [
        row
        for row in train_rows.values()
        if row["role"] == "train"
        and "harness" not in row
        and string(row["id"]) in seen_ids
        and not aliases(row) & protected
    ]
    selected, plan = seen_selection(screen_cells, eligible, seed)
    seen_workload: Record = {"schema": 1, "tasks": [workload_task(row) for row in selected]}
    store.atomic(store.root / "seen-workload.json", canonical(seen_workload))
    establish_ref = artifact(raw["establish"], root)
    establish_ref.verify()
    establish = read(establish_ref.path)
    if establish["tier"] != "screen" or "references" in establish:
        raise ValueError("the establish specification must run fresh screen arms")
    establish_root = establish_ref.path.parent
    compare: Record = {
        **establish,
        **{
            key: reference(artifact(establish[key], establish_root))
            for key in ("source", "binary", "target", "exclusions")
            if key in establish
        },
        "libraries": [
            reference(artifact(value, establish_root)) for value in sequence(establish["libraries"])
        ],
        "arms": [
            {
                "name": record(arm)["name"],
                "drafter": None
                if record(arm)["drafter"] is None
                else reference(artifact(record(arm)["drafter"], establish_root)),
            }
            for arm in sequence(establish["arms"])
        ],
        "workload": reference(Artifact.capture(store.root / "seen-workload.json")),
        "role": "train",
    }
    store.atomic(store.root / "seen-compare.json", canonical(compare))
    result: Record = {
        "schema": "round-inputs-result-v1",
        "seed": seed,
        "screen": reference(screen_ref),
        "screen_capture": {
            "specification": reference(Artifact.capture(store.root / "screen-capture.json")),
            "requests": len(capture_rows),
            "output_cap": SCREEN_CAP,
        },
        "seen": {
            "workload": reference(Artifact.capture(store.root / "seen-workload.json")),
            "compare": reference(Artifact.capture(store.root / "seen-compare.json")),
            "eligible": len(eligible),
            "screen_cells": {f"{c}|{b}": screen_cells[(c, b)] for c, b in sorted(screen_cells)},
            "selected_cells": {f"{c}|{b}": n for (c, b), n in sorted(plan.items()) if n},
            "selected": [
                {
                    "id": row["id"],
                    "group": row["group"],
                    "category": row["category"],
                    "input_tokens": row["input_tokens"],
                }
                for row in selected
            ],
            "rules": "TRAIN rows of the bound bank that are records of the bound TRAIN corpus, without a harness prefix and outside the exclusions; category and input-band marginals equal the screen's, joint cells as close to the screen's as the bank allows; one row per source group; seeded SHA256 order within each cell",
        },
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

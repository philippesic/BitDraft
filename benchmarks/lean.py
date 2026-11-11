from collections import Counter
from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.coverage import apportion, length_band, selection_order
from benchmarks.native import file_reference
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.records import artifact, canonical, read


def prepare(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "requests", "harness", "live"} or raw["schema"] != 1:
        raise ValueError("lean preparation requires frozen request/harness/live specifications")
    if output.exists():
        raise ValueError("lean preparation output must be new")
    store = Store(output)
    published: Record = {}
    for panel, key, tier in (
        ("screen", "requests", "screen"),
        ("requests", "requests", "comparison"),
        ("harness", "harness", "prefix"),
        ("live", "live", "live"),
    ):
        ref = artifact(raw[key], specification.resolve().parent)
        ref.verify()
        source = Comparison.load(ref.path)
        source.verify()
        tasks = list(source.tasks)
        if panel in {"screen", "requests"}:
            counts = (15, 10, 12, 10, 8, 5) if panel == "screen" else (30, 20, 25, 20, 15, 10)
            demands = dict(
                zip(
                    ("code", "reasoning", "documents", "conversation", "structured", "writing"),
                    counts,
                    strict=True,
                )
            )
            lengths = dict(enumerate((30, 18, 9, 3) if panel == "screen" else (60, 36, 18, 6)))
            required = [task for task in tasks if "live_source" in record(task.get("ancestry", {}))]

            def band(task: Record) -> int:
                return length_band(integer(record(task["ancestry"])["input_tokens"]))

            for task in required:
                demands[string(task["category"])] -= 1
                lengths[band(task)] -= 1
            tasks = required + apportion(
                [task for task in tasks if task not in required],
                demands,
                lengths,
                lambda task: string(task["category"]),
                band,
            )
            if panel == "screen":
                tasks = [{**task, "output_cap": 128} for task in tasks]
        elif panel == "harness":
            selected: set[str] = set()
            for category in ("code", "structured"):
                groups = sorted(
                    {string(task["group"]) for task in tasks if task["category"] == category}
                )
                selected.update(groups[:3])
            tasks = [task for task in tasks if task["group"] in selected]
        tasks.sort(key=selection_order)
        workload = store.root / (panel + "-workload.json")
        store.atomic(workload, canonical({"schema": 1, "tasks": tasks}))
        template = read(ref.path)
        if source.references:
            raise ValueError(
                "prepare from established multi-arm specifications, not cached projections"
            )
        template.update(workload=file_reference(workload), tier=tier, repeats=1, warmups=1)
        template["arms"] = [
            *sequence(template["arms"])[:2],
            {
                "name": "candidate",
                "drafter": record(sequence(template["arms"])[-1])["drafter"],
            },
        ]
        path = store.root / (panel + "-establish.json")
        store.atomic(path, canonical(template))
        Comparison.load(path).verify()
        published[panel] = {
            "comparison": file_reference(path),
            "tasks": len(tasks),
            "categories": dict(Counter(string(task["category"]) for task in tasks)),
        }
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "result.json", canonical(published))
    return published

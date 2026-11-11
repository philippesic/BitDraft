from pathlib import Path

from benchmarks.native import file_reference
from benchmarks.sandbox import TOOLS
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, record, sequence, string
from w1a1_eagle.records import artifact, canonical, read

SYSTEMS = {
    "minimal": "You are a helpful coding assistant.",
    "coding": "You are working in a project. Inspect relevant context, preserve existing behavior, make the smallest complete change, and verify your answer. Explain assumptions and remaining limitations clearly.",
    "tools": "You are a coding agent with file and test tools. Read the current files before editing. Use tools when they help, follow project instructions, preserve unrelated work, run focused checks, and give a concise final answer grounded in their results.",
}


def panels(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "workload", "live_definitions"} or raw["schema"] != 1:
        raise ValueError("unsupported panel preparation specification")
    if output.exists():
        raise ValueError("panel output must be new")
    root = specification.resolve().parent
    workload = artifact(raw["workload"], root)
    definitions = artifact(raw["live_definitions"], root)
    workload.verify()
    definitions.verify()
    tasks = [record(value) for value in sequence(read(workload.path)["tasks"])]
    selected: list[Record] = []
    groups: set[str] = set()
    for category in ("code", "structured"):
        category_rows = []
        for task in tasks:
            if task["category"] == category and string(task["group"]) not in groups:
                category_rows.append(task)
                groups.add(string(task["group"]))
                if len(category_rows) == 15:
                    break
        if len(category_rows) != 15:
            raise ValueError(
                "harness requires 15 distinct code and 15 distinct structured DEV groups"
            )
        selected.extend(category_rows)
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    harness: list[Record] = []
    for task in selected:
        if task["role"] != "dev":
            raise ValueError("development harness cannot consume confirmation tasks")
        for name, system in SYSTEMS.items():
            variant = dict(task)
            ancestry = dict(record(task.get("ancestry", {})))
            if "token_sha256" in ancestry:
                ancestry["source_token_sha256"] = ancestry.pop("token_sha256")
            variant.update(
                {
                    "id": string(task["id"]) + "-" + name,
                    "harness": name,
                    "ancestry": ancestry,
                    "messages": [
                        {"role": "system", "content": system},
                        *sequence(task["messages"]),
                    ],
                }
            )
            harness.append(variant)
    live_tasks: list[Record] = []
    for value in sequence(read(definitions.path)["tasks"]):
        definition = record(value)
        name = string(definition["id"])
        if not name.replace("-", "").isalnum() or name in {task["id"] for task in live_tasks}:
            raise ValueError("unsafe or duplicate live task identity")
        fixture = store.root / (name + ".json")
        store.atomic(fixture, canonical(definition["fixture"]))
        live_task: Record = {
            "id": name,
            "group": name,
            "role": "dev",
            "category": definition["category"],
            "messages": [
                {"role": "system", "content": SYSTEMS["tools"]},
                {"role": "user", "content": definition["prompt"]},
            ],
            "output_cap": 1024,
            "tools": TOOLS,
            "fixture": file_reference(fixture),
        }
        if "followup" in definition:
            live_task["followup"] = definition["followup"]
        live_tasks.append(live_task)
    if len(live_tasks) != 12:
        raise ValueError("development live panel requires 12 tasks")
    store.atomic(store.root / "harness-workload.json", canonical({"schema": 1, "tasks": harness}))
    store.atomic(store.root / "live-workload.json", canonical({"schema": 1, "tasks": live_tasks}))
    result: Record = {
        "status": "complete",
        "harness_tasks": len(harness),
        "harness_groups": len(selected),
        "live_tasks": len(live_tasks),
        "role": "dev",
        "scope": "system-instruction sensitivity on fixed public requests and controlled tool tasks; application representativeness remains an empirical question",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

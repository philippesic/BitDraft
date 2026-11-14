from collections import Counter
from pathlib import Path

from benchmarks.native import file_reference
from benchmarks.project_recipes import check_script, replace
from benchmarks.sandbox import SYSTEMS, TOOLS, Sandbox
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, record, sequence, string
from w1a1_eagle.records import artifact, canonical, read


def changed(files: dict[str, str], edits: object) -> dict[str, str]:
    result = dict(files)
    for path, changes in record(edits).items():
        result[path] = replace(result.get(path, ""), changes)
    return result


def prepare_projects(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "role", "projects", "recipes", "provenance"} or raw["schema"] != 2:
        raise ValueError("project collection requires role, projects, recipes and provenance")
    role = string(raw["role"])
    if role not in {"dev", "confirmation"} or output.exists():
        raise ValueError("explicit evaluation role and new project output required")
    root = specification.resolve().parent
    provenance = artifact(raw["provenance"], root)
    provenance.verify()
    sources: dict[str, dict[str, str]] = {}
    store = Store(output)
    for name, value in record(raw["projects"]).items():
        files: dict[str, str] = {}
        for path, reference in record(value).items():
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("project source name escapes its snapshot")
            ref = artifact(reference, root)
            ref.verify()
            if ref.size > 128 * 1024:
                raise ValueError("project workflow source exceeds tool read bound")
            files[path] = ref.path.read_text()
        sources[name] = files
    recipes = artifact(raw["recipes"], root)
    recipes.verify()
    tasks: list[Record] = []
    checks: list[Record] = []
    for value in sequence(read(recipes.path)["tasks"]):
        recipe = record(value)
        name, project = string(recipe["id"]), string(recipe["project"])
        if not name.replace("-", "").isalnum() or name in {task["id"] for task in tasks}:
            raise ValueError("unsafe or duplicate project task name")
        if recipe["style"] not in {"minimal", "coding", "tools"}:
            raise ValueError("unknown agentic instruction style")
        original = sources[project]
        broken = changed(original, recipe["mutations"])
        oracle = changed(original, recipe.get("initial_changes", {}))
        fixture: Record = {"schema": 1, "files": broken, "checks": check_script(recipe["checks"])}
        if "followup" in recipe:
            fixture["followup_checks"] = check_script(recipe["followup_checks"])
        if role == "dev":
            sandbox = Sandbox(store.root / (name + "-qualification"), fixture)
            phases = [sandbox.test()]
            for path, text in oracle.items():
                sandbox.write(path, text)
            phases.append(sandbox.test())
            if [phase["passed"] for phase in phases] != [False, True]:
                raise ValueError("broken/repaired controls failed: " + name + ": " + str(phases))
            if "followup" in recipe:
                sandbox.advance()
                phases.append(sandbox.test())
                for path, text in changed(oracle, recipe["followup_changes"]).items():
                    sandbox.write(path, text)
                phases.append(sandbox.test())
                if [phase["passed"] for phase in phases[-2:]] != [False, True]:
                    raise ValueError("followup controls failed: " + name + ": " + str(phases[-2:]))
            checks.append({"task": name, "phases": phases})
        task: Record = {
            "id": name,
            "group": "project:" + project,
            "project": project,
            "scenario": "agentic-coding",
            "style": recipe["style"],
            "task_kind": recipe["task_kind"],
            "category": "code",
            "prompt": recipe["prompt"],
            "fixture": fixture,
        }
        if "followup" in recipe:
            task["followup"] = recipe["followup"]
        tasks.append(task)
    if len(tasks) != (60 if role == "dev" else 15):
        raise ValueError("agentic collection requires sixty DEV or fifteen confirmation tasks")
    store.atomic(
        store.root / "definitions.json", canonical({"schema": 2, "role": role, "tasks": tasks})
    )
    store.atomic(store.root / "recipes.json", recipes.path.read_bytes())
    store.atomic(store.root / "specification.json", canonical(raw))
    result: Record = {
        "status": "qualified" if role == "dev" else "frozen-unexecuted",
        "role": role,
        "checks": checks,
        "definitions": file_reference(store.root / "definitions.json"),
        "provenance": file_reference(provenance.path),
        "controls_executed": role == "dev",
        "access": [
            {"purpose": "curation", "model_execution": False, "controls_execution": role == "dev"}
        ],
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result


def prepare_panels(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "live_definitions"} or raw["schema"] != 2 or output.exists():
        raise ValueError("new agentic panel requires schema two and live definitions")
    definitions = artifact(raw["live_definitions"], specification.resolve().parent)
    definitions.verify()
    collection = read(definitions.path, 64 * 1024 * 1024)
    if collection["schema"] != 2 or collection["role"] not in {"dev", "confirmation"}:
        raise ValueError("role-bound agentic collection required")
    store = Store(output)
    tasks: list[Record] = []
    for value in sequence(collection["tasks"]):
        definition = record(value)
        name = string(definition["id"])
        if not name.replace("-", "").isalnum() or name in {task["id"] for task in tasks}:
            raise ValueError("unsafe or duplicate agentic task")
        fixture = store.root / (name + ".json")
        store.atomic(fixture, canonical(definition["fixture"]))
        task: Record = {
            key: definition[key]
            for key in ("id", "group", "project", "scenario", "style", "task_kind", "category")
        }
        task.update(
            role=collection["role"],
            messages=[
                {"role": "system", "content": SYSTEMS[string(task["style"])]},
                {"role": "user", "content": definition["prompt"]},
            ],
            output_cap=1024,
            tools=TOOLS,
            fixture=file_reference(fixture),
        )
        if "followup" in definition:
            task["followup"] = definition["followup"]
        tasks.append(task)
    store.atomic(store.root / "live-workload.json", canonical({"schema": 1, "tasks": tasks}))
    store.atomic(store.root / "specification.json", canonical(raw))
    result: Record = {
        "status": "complete",
        "role": collection["role"],
        "tasks": len(tasks),
        "projects": dict(Counter(string(task["project"]) for task in tasks)),
        "styles": dict(Counter(string(task["style"]) for task in tasks)),
        "model_execution": False,
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result


def prepare_checks(specification: Path, output: Path) -> Record:
    from benchmarks.comparison import Comparison

    raw = read(specification)
    if set(raw) != {"schema", "live", "slice_ids"} or raw["schema"] != 2 or output.exists():
        raise ValueError(
            "agentic checks require schema two, live specification and fixed slice IDs"
        )
    ref = artifact(raw["live"], specification.resolve().parent)
    ref.verify()
    source = Comparison.load(ref.path)
    source.verify()
    if source.tier != "agentic-suite" or source.references:
        raise ValueError("prepare from a complete fresh agentic suite specification")
    selected = {string(value) for value in sequence(raw["slice_ids"])}
    if len(selected) != 12 or not selected <= {string(task["id"]) for task in source.tasks}:
        raise ValueError("routine slice requires twelve distinct existing tasks")
    store = Store(output)
    published: Record = {}
    for panel, tasks in (
        ("agentic-suite", list(source.tasks)),
        ("agentic-slice", [task for task in source.tasks if task["id"] in selected]),
    ):
        workload = store.root / (panel + "-workload.json")
        store.atomic(workload, canonical({"schema": 1, "tasks": tasks}))
        template = read(ref.path)
        template.update(workload=file_reference(workload), tier=panel, repeats=1, warmups=1)
        path = store.root / (panel + "-establish.json")
        store.atomic(path, canonical(template))
        Comparison.load(path).verify()
        published[panel] = {"comparison": file_reference(path), "tasks": len(tasks)}
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "result.json", canonical(published))
    return published

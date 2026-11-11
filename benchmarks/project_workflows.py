from pathlib import Path

from benchmarks.native import file_reference
from benchmarks.sandbox import Sandbox
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, record, sequence, string
from w1a1_eagle.records import artifact, canonical, read


def replace(text: str, changes: object) -> str:
    for value in sequence(changes):
        change = record(value)
        old, new = string(change["old"]), change["new"]
        if not isinstance(new, str):
            raise ValueError("replacement must be text")
        if not old or text.count(old) != 1:
            raise ValueError("project fixture mutation must match exactly once")
        text = text.replace(old, new, 1)
    return text


def check_script(value: object) -> str:
    return (
        "import sys\nsys.path.insert(0, '/work')\n"
        + string(value)
        + "\nprint('ACCEPTANCE_CHECKS_PASSED')\n"
    )


def prepare(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "sources", "recipes"} or raw["schema"] != 1 or output.exists():
        raise ValueError("new project workflow output and authenticated sources required")
    root = specification.resolve().parent
    sources: dict[str, str] = {}
    frozen: Record = {}
    store = Store(output)
    for name, value in record(raw["sources"]).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("project source name escapes its snapshot")
        ref = artifact(value, root)
        ref.verify()
        if ref.size > 128 * 1024:
            raise ValueError("project workflow source exceeds tool read bound")
        sources[name] = ref.path.read_text()
        destination = store.root / "source" / name
        Store(destination.parent).atomic(destination, ref.path.read_bytes())
        frozen[name] = file_reference(destination)
    recipes = artifact(raw["recipes"], root)
    recipes.verify()
    store.atomic(store.root / "recipes.json", recipes.path.read_bytes())
    store.atomic(store.root / "specification.json", canonical(raw))
    tasks: list[Record] = []
    checks: list[Record] = []
    for value in sequence(read(recipes.path)["tasks"]):
        recipe = record(value)
        name, path = string(recipe["id"]), string(recipe["path"])
        if not name.replace("-", "").isalnum():
            raise ValueError("unsafe project task name")
        original = sources[path]
        oracle = replace(original, recipe.get("initial_changes", []))
        broken = dict(sources)
        broken[path] = replace(original, recipe["mutations"])
        fixture: Record = {"schema": 1, "files": broken, "checks": check_script(recipe["checks"])}
        if "followup_checks" in recipe:
            fixture["followup_checks"] = check_script(recipe["followup_checks"])
        sandbox = Sandbox(store.root / (name + "-qualification"), fixture)
        initial = sandbox.test()
        if initial["passed"] is not False:
            raise ValueError("broken fixture unexpectedly passes: " + name)
        sandbox.write(path, oracle)
        repaired = sandbox.test()
        if repaired["passed"] is not True:
            raise ValueError("project oracle fails: " + name + ": " + string(repaired["output"]))
        phases: list[Record] = [initial, repaired]
        if "followup" in recipe:
            sandbox.advance()
            before = sandbox.test()
            sandbox.write(path, replace(oracle, recipe["followup_changes"]))
            after = sandbox.test()
            if before["passed"] is not False or after["passed"] is not True:
                raise ValueError("followup fixture failed its phase controls: " + name)
            phases.extend([before, after])
        task: Record = {
            "id": name,
            "group": "project:" + path,
            "category": recipe["category"],
            "prompt": recipe["prompt"],
            "fixture": fixture,
        }
        if "followup" in recipe:
            task["followup"] = recipe["followup"]
        tasks.append(task)
        checks.append({"task": name, "phases": phases})
    if len(tasks) != 12:
        raise ValueError("project DEV panel requires twelve tasks")
    store.atomic(store.root / "definitions.json", canonical({"schema": 1, "tasks": tasks}))
    result: Record = {
        "status": "complete",
        "checks": checks,
        "definitions": file_reference(store.root / "definitions.json"),
        "frozen_sources": frozen,
        "frozen_recipes": file_reference(store.root / "recipes.json"),
        "scope": "controlled regressions in authenticated project source; one project, DEV only, no unseen deployment claim",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

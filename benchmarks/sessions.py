import ast
import hashlib
import json
import re
from pathlib import Path
from typing import cast

from benchmarks.native import file_reference
from benchmarks.project_workflows import check_script
from benchmarks.protection import source_names
from benchmarks.sandbox import Sandbox
from benchmarks.serving import native_server
from benchmarks.workflows import execute
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import artifact, canonical, identity, read

TRAILER = re.compile(r"\s*[A-Za-z-]+-by:|\s*(?:Change-Id|Cc|Fixes|Closes|Refs):", re.IGNORECASE)
PERMISSIVE = frozenset({"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc"})
PROJECT_FILES = 8
FILE_BYTES = 24 * 1024
PROJECT_BYTES = 96 * 1024


def json_lines(path: Path) -> list[Record]:
    with path.open() as source:
        return [record(cast(object, json.loads(line))) for line in source]


def text_value(row: Record, key: str) -> str:
    value = row.get(key) or ""
    if not isinstance(value, str):
        raise ValueError("source text field must be a string")
    return value


def parses(text: str) -> bool:
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


def safe_path(name: str) -> bool:
    path = Path(name)
    return (
        name.endswith(".py")
        and not path.is_absolute()
        and ".." not in path.parts
        and all(part and not part.startswith(".") for part in path.parts)
    )


def components(rows: list[Record]) -> dict[str, list[Record]]:
    parents: dict[str, str] = {}

    def find(name: str) -> str:
        parents.setdefault(name, name)
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for row in rows:
        names = sorted(set(string(row["repos"]).split(",")))
        for name in names:
            left, right = find(names[0]), find(name)
            parents[max(left, right)] = min(left, right)
    grouped: dict[str, list[Record]] = {}
    for row in rows:
        grouped.setdefault(find(sorted(string(row["repos"]).split(","))[0]), []).append(row)
    return grouped


def commit_order(row: Record) -> tuple[str, str]:
    return string(row["commit"]), text_value(row, "old_file")


def project_task(root: str, rows: list[Record], system: str, cap: int) -> Record | None:
    ordered = sorted(rows, key=commit_order)
    if any(string(row["license"]) not in PERMISSIVE for row in ordered):
        return None
    snapshot: dict[str, str] = {}
    for row in ordered:
        name, text = text_value(row, "old_file"), text_value(row, "new_contents")
        if (
            name not in snapshot
            and safe_path(name)
            and 0 < len(text.encode()) <= FILE_BYTES
            and parses(text)
        ):
            snapshot[name] = text
    for row in ordered:
        name, old, new = (
            text_value(row, key) for key in ("old_file", "old_contents", "new_contents")
        )
        message = "\n".join(
            line
            for line in text_value(row, "message").splitlines()
            if not TRAILER.match(line) and "@" not in line
        ).strip()
        if not (
            name in snapshot
            and text_value(row, "new_file") == name
            and 1024 <= len(old.encode()) <= FILE_BYTES
            and old != new
            and 16 <= len(message) <= 600
            and parses(old)
        ):
            continue
        nearby = [other for other in sorted(snapshot) if Path(other).parent == Path(name).parent]
        distant = [other for other in sorted(snapshot) if Path(other).parent != Path(name).parent]
        others = [other for other in nearby + distant if other != name][: PROJECT_FILES - 1]
        files = {other: snapshot[other] for other in others} | {name: old}
        if len(files) < 2 or sum(len(text.encode()) for text in files.values()) > PROJECT_BYTES:
            continue
        digest = identity({"text": old})
        checks = (
            "import hashlib, pathlib\n"
            "for path in sorted(pathlib.Path('/work').rglob('*.py')):\n"
            "    compile(path.read_text(), str(path.relative_to('/work')), 'exec')\n"
            "changed = pathlib.Path('/work/" + name + "').read_text()\n"
            "assert hashlib.sha256(changed.encode()).hexdigest() != '"
            + hashlib.sha256(old.encode()).hexdigest()
            + "', 'the requested change is not present in "
            + name
            + "'\n"
        )
        prompt = (
            message
            + "\n\nThe change belongs in "
            + name
            + ". Inspect the relevant files, make the edit and run the tests."
        )
        key = string(row["commit"]) + ":" + name
        return {
            "id": "session-commitpack-" + identity({"key": key})[:24],
            "source": "session-commitpack",
            "source_key": key,
            "group": "commitpack:" + root,
            "aliases": sorted({"commitpack:" + root, key, digest}),
            "role": "train",
            "category": "code",
            "subtype": "agentic_session",
            "license": string(row["license"]),
            "prior_use": "unknown",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "output_cap": cap,
            "fixture": {"schema": 1, "files": files, "checks": check_script(checks)},
        }
    return None


def build_tasks(raw: Record, root: Path) -> tuple[list[Record], Record]:
    source, prior = artifact(raw["source"], root), artifact(raw["prior_source"], root)
    harnesses = artifact(raw["harnesses"], root)
    for ref in (source, prior, harnesses):
        ref.verify()
    system = string(read(harnesses.path)["tools"])
    protected: set[str] = {"heldout24"}
    for key in ("exclusions", "heldout_exclusions"):
        ref = artifact(raw[key], root)
        ref.verify()
        protected.update(string(value) for value in sequence(read(ref.path)["identities"]))
    seen = {name for row in json_lines(prior.path) for name in string(row["repos"]).split(",")}
    rows = json_lines(source.path)
    grouped = components(rows)
    seed = integer(raw["seed"])
    ranked = sorted(
        (identity({"group": name, "seed": seed}), name)
        for name, members in grouped.items()
        if not any(repo in seen for row in members for repo in string(row["repos"]).split(","))
        and int(identity({"group": "commitpack:" + name})[:8], 16) % 5 != 0
    )
    eligible = [name for _, name in ranked]
    tasks: list[Record] = []
    refused = {"protected": 0, "no_project": 0}
    for name in eligible:
        task = project_task(name, grouped[name], system, integer(raw["output_cap"]))
        if task is None:
            refused["no_project"] += 1
            continue
        if source_names(task) & protected:
            refused["protected"] += 1
            continue
        tasks.append(task)
        if len(tasks) == integer(raw["tasks"]):
            break
    if len(tasks) != integer(raw["tasks"]):
        raise ValueError("source cannot supply the requested TRAIN session tasks")
    summary: Record = {
        "source_rows": len(rows),
        "components": len(grouped),
        "eligible_train_components": len(eligible),
        "refused_examined": refused,
    }
    return tasks, summary


def generate(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw)
        != {
            "schema",
            "role",
            "runtime",
            "target",
            "context",
            "source",
            "prior_source",
            "harnesses",
            "exclusions",
            "heldout_exclusions",
            "tasks",
            "seed",
            "output_cap",
        }
        or raw["schema"] != 1
        or raw["role"] != "train"
    ):
        raise ValueError("unsupported TRAIN session specification")
    root = specification.resolve().parent
    runtime, target = artifact(raw["runtime"], root), artifact(raw["target"], root)
    runtime.verify()
    target.verify()
    built = [
        artifact(value, runtime.path.parent) for value in sequence(read(runtime.path)["artifacts"])
    ]
    servers = [ref for ref in built if ref.path.name == "llama-server"]
    if len(servers) != 1:
        raise ValueError("runtime must provide one authenticated server")
    for ref in built:
        ref.verify()
    context = integer(raw["context"])
    if not 4096 <= context <= 32768 or not 1 <= integer(raw["output_cap"]) <= 2048:
        raise ValueError("invalid session context or output cap")
    tasks, summary = build_tasks(raw, root)
    store = Store(output)
    definitions = canonical({"schema": 1, "tasks": tasks})
    if (store.root / "specification.json").exists():
        if (
            identity(read(store.root / "specification.json")) != identity(raw)
            or (store.root / "tasks.json").read_bytes() != definitions
        ):
            raise ValueError("cannot resume sessions with a changed specification")
    else:
        store.atomic(store.root / "specification.json", canonical(raw))
        store.atomic(store.root / "tasks.json", definitions)
    progress = store.root / "progress.json"
    completed: list[Record] = (
        [record(value) for value in sequence(read(progress, 64 * 1024 * 1024)["sessions"])]
        if progress.exists()
        else []
    )
    for value in completed:
        artifact(value["transcript"], store.root).verify()
    done = {string(value["id"]) for value in completed}
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    attempt = 0
    while (store.root / ("attempt-" + str(attempt))).exists():
        attempt += 1
    cell = Store(store.root / ("attempt-" + str(attempt)))
    try:
        with device_lease("cuda"):
            cell.atomic(cell.root / "gpu.txt", cuda_identity().encode())
            with native_server(servers[0], target, None, context, 1, cell) as port:
                for task in tasks:
                    name = string(task["id"])
                    if name in done:
                        continue
                    task_store = Store(store.root / "tasks" / name)
                    sandbox = Sandbox(
                        Store(cell.root / "sandboxes" / name).root, record(task["fixture"])
                    )
                    row, messages, responses = execute(port, task, sandbox, context, False)
                    task_store.atomic(
                        task_store.root / "transcript.json",
                        canonical(
                            {
                                "id": name,
                                "messages": messages,
                                "responses": responses,
                                "row": row,
                                "code_sha256": code,
                            }
                        ),
                    )
                    completed.append(
                        {
                            "id": name,
                            "termination": row["termination"],
                            "turns": row["turns"],
                            "completed": row["completed"],
                            "transcript": file_reference(task_store.root / "transcript.json"),
                        }
                    )
                    done.add(name)
                    store.atomic(progress, canonical({"sessions": completed}))
        result: Record = {
            "status": "complete",
            "role": "train",
            "sessions": len(completed),
            "completed_tasks": sum(value["completed"] is True for value in completed),
            "turns": sum(integer(value["turns"]) for value in completed),
            "tasks": file_reference(store.root / "tasks.json"),
            "progress": file_reference(progress),
            "source_summary": summary,
            "runtime": file_reference(runtime.path),
            "target_sha256": target.sha256,
            "code_sha256": code,
            "scope": "target-only greedy TRAIN tool sessions in isolated CommitPackFT project copies; task success is not required for supervision",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        cell.atomic(
            cell.root / "failure.json",
            canonical({"error": str(error), "completed": len(completed)}),
        )
        raise

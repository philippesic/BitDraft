from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.coverage import length_band
from benchmarks.native import file_reference
from benchmarks.sources import candidate
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, record, sequence, string
from w1a1_eagle.numerical.prompting import Tokenizer
from w1a1_eagle.records import artifact, canonical, identity, read


def prepare(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "run", "tokenizer"} or raw["schema"] != 1 or output.exists():
        raise ValueError("authenticated live run/tokenizer and new replay output required")
    run = artifact(raw["run"], specification.resolve().parent)
    binary = artifact(raw["tokenizer"], specification.resolve().parent)
    run.verify()
    binary.verify()
    measured = read(run.path, 64 * 1024 * 1024)
    comparison = Comparison.load(run.path.parent / "specification.json")
    comparison.verify()
    if measured["status"] != "complete" or comparison.role != "dev" or comparison.tier != "live":
        raise ValueError("replay preparation requires a completed live DEV panel")
    tasks = {string(task["id"]): task for task in comparison.tasks}
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    tokenizer = Tokenizer(binary.path, comparison.target.path, store.root / "tokenizer.log")
    rows: list[Record] = []
    groups: set[str] = set()
    try:
        for value in sequence(measured["rows"]):
            row = record(value)
            if row["arm"] != "target" or row["rep"] != 0:
                continue
            task = tasks[string(row["task"])]
            group = string(task["group"])
            if group in groups:
                continue
            transcript = run.path.parent / "0-target" / string(task["id"]) / "transcript.json"
            messages = [record(item) for item in sequence(read(transcript)["messages"])]
            readers: set[str] = set()
            prefix: list[Record] | None = None
            for index, message in enumerate(messages):
                if message["role"] == "assistant":
                    for item in sequence(message.get("tool_calls", [])):
                        call = record(item)
                        if record(call["function"])["name"] == "read_file":
                            readers.add(string(call["id"]))
                if message["role"] == "tool" and string(message["tool_call_id"]) in readers:
                    prefix = messages[: index + 1]
                    break
            if prefix is None:
                raise ValueError("live task has no first file-read replay boundary")
            category = (
                "documents"
                if task["category"] == "documentation"
                else "structured"
                if task["category"] == "tool-workflow"
                else "code"
            )
            tools = [record(item) for item in sequence(task["tools"])]
            prepared = candidate(
                "project-replay",
                string(task["id"]),
                group,
                category,
                "editing" if category == "code" else "application_replay",
                prefix,
                tools,
            )
            rendered = tokenizer.render(prefix, tools)
            tokens = tokenizer.tokenize(rendered)
            prepared.update(
                {
                    "group": group,
                    "role": "dev",
                    "inherited_role": "dev",
                    "prior_use": "consumed_dev",
                    "input_tokens": len(tokens),
                    "length_band": length_band(len(tokens)),
                    "token_sha256": identity({"tokens": list(tokens)}),
                    "rendered_sha256": identity({"text": rendered}),
                    "template_sha256": tokenizer.template_sha256,
                    "live_source": {
                        "run": file_reference(run.path),
                        "transcript": file_reference(transcript),
                        "boundary": "first file-read tool reply; outcome-independent; one per source group",
                    },
                }
            )
            rows.append(prepared)
            groups.add(group)
        store.atomic(store.root / "pool.jsonl", b"".join(canonical(row) + b"\n" for row in rows))
        result: Record = {
            "status": "complete",
            "tasks": len(rows),
            "target_sha256": comparison.target.sha256,
            "template_sha256": tokenizer.template_sha256,
            "pool": file_reference(store.root / "pool.jsonl"),
            "scope": "measured local project application sessions, including failed tasks; single-project DEV exposure, no unseen-application claim",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    finally:
        tokenizer.close()

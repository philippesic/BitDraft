import json
from collections import Counter
from pathlib import Path
from typing import Callable, cast

from benchmarks.conventions import apply
from benchmarks.native import file_reference
from benchmarks.retrain_sources import openhands_call
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.prompting import json_rows, parquet_rows
from w1a1_eagle.records import artifact, canonical, identity, read

FIELDS = {
    "schema",
    "definition",
    "qualitative",
    "throughput",
    "agentic",
    "agentic_tools",
    "tools",
    "seed",
    "counts",
}
THINKING_CATEGORIES = {"math", "reasoning", "coding", "stem"}
AGENT_CHARACTERS = (24000, 96000)


def order(value: object, seed: int) -> str:
    return identity({"value": value, "seed": seed})


def task(
    key: str,
    group: str,
    category: str,
    scenario: str,
    messages: list[Record],
    cap: int,
    source: str,
    tools: list[Record] | None = None,
    thinking: bool = False,
) -> Record:
    result: Record = {
        "id": "gate-" + identity({"key": key, "scenario": scenario})[:24],
        "group": group,
        "role": "confirmation",
        "category": category,
        "scenario": scenario,
        "messages": messages,
        "output_cap": cap,
        "thinking": thinking,
        "ancestry": {
            "source": source,
            "source_key": key,
            "prior_use": "public-benchmark",
            "role_origin": "release-gate-v1",
        },
    }
    if tools:
        result["tools"] = tools
    return result


def speed_messages(row: Record) -> list[Record]:
    turns = [string(v) for v in sequence(row["turns"])]
    return [{"role": "user", "content": turns[0]}]


def object_types(value: object) -> object:
    if isinstance(value, dict):
        mapped = {string(k): object_types(v) for k, v in cast(dict[object, object], value).items()}
        if mapped.get("type") == "dict":
            mapped["type"] = "object"
        if mapped.get("type") == "float":
            mapped["type"] = "number"
        if mapped.get("type") == "tuple":
            mapped["type"] = "array"
        return mapped
    if isinstance(value, list):
        return [object_types(v) for v in cast(list[object], value)]
    return value


def by_order(field: str, seed: int) -> Callable[[Record], str]:
    def key(row: Record) -> str:
        return order(row[field], seed)

    return key


def by_name(seed: int) -> Callable[[str], str]:
    def key(name: str) -> str:
        return order(name, seed)

    return key


def bfcl_rows(path: Path) -> list[Record]:
    return list(json_rows(path))


def prepare_gate(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != FIELDS or raw["schema"] != "release-gate-prepare-v1":
        raise ValueError("unsupported release gate preparation specification")
    if output.exists():
        raise ValueError("release gate output must be new")
    root = specification.resolve().parent
    definition = artifact(raw["definition"], root)
    definition.verify()
    seed = integer(raw["seed"])
    counts = record(raw["counts"])
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    freeze_source(code_root, store.root / "sources", source_identity(code_root), 0)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    texts: list[str] = []
    workloads: dict[str, list[Record]] = {}

    qualitative_ref = artifact(raw["qualitative"], root)
    qualitative_ref.verify()
    qualitative = list(parquet_rows(qualitative_ref.path))
    if len(qualitative) != 880:
        raise ValueError("SPEED-Bench qualitative split must hold 880 prompts")
    masked = sum(
        "FULL BENCHMARK DATA SHOULD BE FETCHED" in json.dumps(r["turns"]) for r in qualitative
    )
    if masked:
        raise ValueError("qualitative prompts still masked; run the upstream preparation")
    per_category: dict[str, list[Record]] = {}
    for row in qualitative:
        texts.extend(string(v) for v in sequence(row["turns"]))
        per_category.setdefault(string(row["category"]), []).append(row)
    general: list[Record] = []
    thinking: list[Record] = []
    sampled: list[Record] = []
    sampled_count = integer(counts["sampled_per_category"])
    for category, rows in sorted(per_category.items()):
        for row in rows:
            key = "speed:" + string(row["question_id"])
            messages = speed_messages(row)
            general.append(
                task(key, key, category, "general", messages, 1024, "speed-bench-qualitative")
            )
            if category in THINKING_CATEGORIES:
                thinking.append(
                    task(
                        key,
                        key,
                        category,
                        "thinking-on",
                        messages,
                        2048,
                        "speed-bench-qualitative",
                        thinking=True,
                    )
                )
        chosen = sorted(rows, key=by_order("question_id", seed))[:sampled_count]
        for row in chosen:
            key = "speed:" + string(row["question_id"])
            sampled.append(
                task(
                    key,
                    key,
                    category,
                    "sampled",
                    speed_messages(row),
                    1024,
                    "speed-bench-qualitative",
                )
            )
    workloads["general"] = general
    workloads["thinking-on"] = thinking
    workloads["sampled"] = sampled

    long_rows: list[Record] = []
    per_class = integer(counts["long_per_entropy_class"])
    for value in sequence(raw["throughput"]):
        ref = artifact(value, root)
        ref.verify()
        rows = list(parquet_rows(ref.path))
        classes: dict[str, list[Record]] = {}
        for row in rows:
            classes.setdefault(string(row["category"]), []).append(row)
        for entropy, members in sorted(classes.items()):
            for row in sorted(members, key=by_order("question_id", seed))[:per_class]:
                key = "speed-throughput:" + ref.path.parent.name + ":" + string(row["question_id"])
                turns = [string(v) for v in sequence(row["turns"])]
                if any("FULL BENCHMARK DATA SHOULD BE FETCHED" in t for t in turns):
                    raise ValueError("throughput prompt still masked; run the upstream preparation")
                texts.append(turns[0])
                long_rows.append(
                    task(
                        key,
                        key,
                        entropy,
                        "long-input",
                        [{"role": "user", "content": turns[0]}],
                        512,
                        "speed-bench-throughput",
                    )
                )
    workloads["long-input"] = long_rows

    agentic_ref, tools_ref = artifact(raw["agentic"], root), artifact(raw["agentic_tools"], root)
    agentic_ref.verify()
    tools_ref.verify()
    agent_tools = [
        record(v) for v in sequence(cast(object, json.loads(tools_ref.path.read_text())))
    ]
    repositories: dict[str, list[str]] = {}
    for row in parquet_rows(agentic_ref.path, ["repo", "instance_id"]):
        repositories.setdefault(string(row["repo"]).lower(), []).append(string(row["instance_id"]))
    reserved = sorted(repositories, key=by_name(seed))[: integer(counts["agentic_repositories"])]
    chosen_instances = {min(repositories[name], key=by_name(seed)) for name in reserved}
    trajectories: dict[str, Record] = {}
    for row in parquet_rows(agentic_ref.path, ["repo", "instance_id", "trajectory"]):
        if string(row["instance_id"]) in chosen_instances:
            trajectories[string(row["repo"]).lower()] = row
    agentic: list[Record] = []
    for repository in reserved:
        trajectory = trajectories[repository]
        history: list[Record] = []
        cuts: list[int] = []
        size = 0
        for value in sequence(trajectory["trajectory"]):
            step = record(value)
            role = string(step["role"])
            content = step.get("content")
            text = content if isinstance(content, str) else ""
            if role == "assistant":
                if AGENT_CHARACTERS[0] <= size <= AGENT_CHARACTERS[1]:
                    cuts.append(len(history))
                message: Record = {"role": "assistant", "content": text}
                if step.get("tool_calls"):
                    message["tool_calls"] = [
                        openhands_call(v) for v in sequence(step["tool_calls"])
                    ]
                history.append(message)
            elif role == "tool":
                history.append(
                    {
                        "role": "tool",
                        "content": text,
                        "tool_call_id": str(step.get("tool_call_id", "")),
                    }
                )
            elif role in {"system", "user"}:
                history.append({"role": role, "content": text})
            size += len(text)
        if not cuts:
            continue
        cut = cuts[int(order(trajectory["instance_id"], seed)[:4], 16) % len(cuts)]
        key = "swe-rebench:" + string(trajectory["instance_id"])
        agentic.append(
            task(
                key,
                "repository:" + repository,
                "agentic",
                "agentic-replay",
                history[:cut],
                1024,
                "swe-rebench-openhands",
                agent_tools,
            )
        )
    workloads["agentic-replay"] = agentic

    tool_rows: list[Record] = []
    held_out: list[Record] = []
    per_file = integer(counts["tool_calls_per_file"])
    for value in sequence(raw["tools"]):
        ref = artifact(value, root)
        ref.verify()
        rows = bfcl_rows(ref.path)
        for item in rows:
            for turn in sequence(item["question"]):
                for part in sequence(turn):
                    texts.append(string(record(part)["content"]))
        for entry in sorted(rows, key=by_order("id", seed))[:per_file]:
            request: list[Record] = [record(m) for m in sequence(sequence(entry["question"])[0])]
            functions = [record(object_types(f)) for f in sequence(entry["function"])]
            offered: list[Record] = [{"type": "function", "function": f} for f in functions]
            key = "bfcl:" + string(entry["id"])
            category = ref.path.stem.removeprefix("BFCL_v3_")
            tool_rows.append(task(key, key, category, "tool-calls", request, 512, "bfcl", offered))
            converted, _ = apply("react", request, offered)
            held_out.append(task(key, key, category, "held-out-convention", converted, 512, "bfcl"))
    workloads["tool-calls"] = tool_rows
    workloads["held-out-convention"] = held_out

    files: dict[str, Record] = {}
    sizes: dict[str, int] = {}
    workload_store = Store(store.root / "workloads")
    for name, tasks in workloads.items():
        if not tasks or len(tasks) > 600:
            for part in range(0, len(tasks), 440):
                chunk = tasks[part : part + 440]
                path = workload_store.root / (name + "-" + str(part // 440) + ".json")
                workload_store.atomic(path, canonical({"schema": 1, "tasks": chunk}))
                files[name + "-" + str(part // 440)] = file_reference(path)
        else:
            path = workload_store.root / (name + ".json")
            workload_store.atomic(path, canonical({"schema": 1, "tasks": tasks}))
            files[name] = file_reference(path)
        sizes[name] = len(tasks)
    gate_texts = store.root / "gate-texts.jsonl"
    with gate_texts.open("wb") as sink:
        for text in texts:
            if text.strip():
                sink.write(canonical({"text": text}) + b"\n")
    exclusions = {
        "schema": 1,
        "identities": sorted(
            {"repository:" + name for name in reserved}
            | {
                string(record(t["ancestry"])["source_key"])
                for tasks in workloads.values()
                for t in tasks
            }
        ),
    }
    store.atomic(store.root / "gate-exclusions.json", canonical(exclusions))
    result: Record = {
        "schema": "release-gate-prepared-v1",
        "definition": file_reference(definition.path),
        "workloads": files,
        "sizes": sizes,
        "categories": {
            name: dict(Counter(string(t["category"]) for t in tasks))
            for name, tasks in workloads.items()
        },
        "reserved_repositories": reserved,
        "gate_texts": file_reference(gate_texts),
        "exclusions": file_reference(store.root / "gate-exclusions.json"),
        "limitations": [
            "multi-turn SPEED-Bench prompts use their first turn only",
            "agentic replay histories come from Qwen3-Coder-480B trajectories, cut before one assistant turn",
            "agentic histories are decontaminated by repository split only; their shared OpenHands boilerplate would match every trajectory",
        ],
        "code_sha256": source_identity(code_root),
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

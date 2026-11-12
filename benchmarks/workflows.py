import json
import re
from pathlib import Path
from typing import cast

from benchmarks.comparison import Comparison
from benchmarks.measurement import now, request
from benchmarks.metrics import totals
from benchmarks.runner import native_counts, reconcile, signature, valid_sample, verify_generation
from benchmarks.runtime_policy import LIVE_SECONDS, LIVE_TURNS, payload
from benchmarks.sandbox import TOOLS, Sandbox
from benchmarks.serving import session
from benchmarks.statistics import paired, partitions
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import artifact, canonical, identity, read


def calls(content: str) -> list[tuple[str, Record]]:
    values: list[tuple[str, Record]] = []
    for body in cast(list[str], re.findall(r"<tool_call>(.*?)</tool_call>", content, re.S)):
        if body.lstrip().startswith("{"):
            raw = record(cast(object, json.loads(body)))
            values.append((string(raw["name"]), record(raw["arguments"])))
            continue
        function = re.fullmatch(r"\s*<function=([a-z_]+)>(.*?)</function>\s*", body, re.S)
        if function is None:
            raise ValueError("malformed native tool call")
        parameters = cast(
            list[tuple[str, str]],
            re.findall(r"<parameter=([a-z_]+)>\s?([\s\S]*?)</parameter>", function[2]),
        )
        if len({name for name, _ in parameters}) != len(parameters):
            raise ValueError("duplicate tool parameters")
        values.append(
            (
                function[1],
                {
                    name: int(value.strip())
                    if name in {"start_line", "end_line"}
                    else value.strip()
                    if name == "path"
                    else value
                    for name, value in parameters
                },
            )
        )
    if len(values) > 8 or "<tool_call>" in content and not values:
        raise ValueError("invalid or excessive tool calls")
    return values


def live(specification: Path, output: Path) -> Record:
    spec = Comparison.load(specification)
    spec.verify()
    if spec.sampling.temperature:
        raise ValueError(
            "sampled live workflows require a separately qualified harness; this entry is greedy"
        )
    if spec.tier not in {"live", "pilot"} or output.exists():
        raise ValueError("live tier or explicit pilot and a new output required")
    fixtures: dict[str, Record] = {}
    for task in spec.tasks:
        ref = artifact(task["fixture"], specification.resolve().parent)
        ref.verify()
        fixture = read(ref.path)
        if (
            set(fixture) - {"followup_checks"} != {"schema", "files", "checks"}
            or fixture["schema"] != 1
        ):
            raise ValueError("unsupported live fixture")
        if ("followup_checks" in fixture) != ("followup" in task):
            raise ValueError("live followup requires separate protected phase checks")
        fixtures[string(task["id"])] = fixture
    with device_lease("cuda"):
        return live_leased(specification, output, spec, fixtures)


def live_leased(
    specification: Path, output: Path, spec: Comparison, fixtures: dict[str, Record]
) -> Record:
    began_panel = now()
    store = Store(output)
    store.atomic(store.root / "gpu.txt", cuda_identity().encode())
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    rows: list[Record] = []
    cached = spec.cached()
    if any(value["gpu"] != cuda_identity() for value in cached.values()):
        raise ValueError(
            "reference hardware/driver differs; establish new controls before comparison"
        )
    references = (
        {key: string(value) for key, value in record(cached["target"]["signatures"]).items()}
        if cached
        else {}
    )
    try:
        for rep in range(spec.repeats):
            arms = list(spec.arms)
            if rep:
                arms = arms[rep % len(arms) :] + arms[: rep % len(arms)]
            for arm in arms:
                cell = Store(store.root / (str(rep) + "-" + arm.name))
                responses: list[Record] = []
                with session(spec, arm, cell) as port:
                    for index in range(spec.warmups):
                        warm = Store(cell.root / ("warmup-" + str(index)))
                        task = spec.tasks[0]
                        sandbox = Sandbox(warm.root / "sandbox", fixtures[string(task["id"])])
                        row, transcript, generated = execute(
                            port, task, sandbox, spec.context, arm.drafter is not None
                        )
                        responses.extend(generated)
                        warm.atomic(
                            warm.root / "result.json",
                            canonical(
                                {
                                    "row": row,
                                    "messages": transcript,
                                    "responses": generated,
                                    "scope": "warmup; excluded from primary task totals",
                                }
                            ),
                        )
                    for task in spec.tasks:
                        name = string(task["id"])
                        task_store = Store(cell.root / name)
                        sandbox = Sandbox(task_store.root / "sandbox", fixtures[name])
                        row, transcript, generated = execute(
                            port, task, sandbox, spec.context, arm.drafter is not None
                        )
                        responses.extend(generated)
                        task_store.atomic(
                            task_store.root / "transcript.json",
                            canonical({"messages": transcript, "responses": generated}),
                        )
                        state = {
                            str(path.relative_to(sandbox.work.root)): path.read_text()
                            for path in sorted(sandbox.work.root.rglob("*"))
                            if path.is_file() and not path.is_symlink()
                        }
                        fingerprint = identity(
                            {
                                "responses": [signature(value) for value in generated],
                                "files": state,
                                "passed": row["completed"],
                            }
                        )
                        if arm.name == "target" and name not in references:
                            references[name] = fingerprint
                        row.update(
                            {
                                "arm": arm.name,
                                "task": name,
                                "group": task["group"],
                                "category": task["category"],
                                "rep": rep,
                                "correct": references.get(name) == fingerprint,
                                "state_sha256": fingerprint,
                            }
                        )
                        rows.append(row)
                        store.atomic(store.root / "progress.json", canonical({"rows": rows}))
                if arm.drafter:
                    reconcile((cell.root / "server.log").read_text(), responses)
                    if not any(
                        integer(record(row["native"])["proposed"])
                        for row in rows
                        if row["arm"] == arm.name
                    ):
                        raise ValueError("fallback-only live arm; activation not established")
        if source_identity(code_root) != code:
            raise ValueError("live benchmark source changed during execution")
        historical_rows = [
            record(row) for value in cached.values() for row in sequence(value["rows"])
        ]
        names = [arm.name for arm in spec.arms] + list(cached)
        result: Record = {
            "status": "complete",
            "protocol": spec.reference_protocol(),
            "role": spec.role,
            "tier": spec.tier,
            "rows": rows,
            "elapsed_seconds": now() - began_panel,
            "historical_references": {name: value["reference"] for name, value in cached.items()},
            "summary": {
                arm.name: {
                    **summarize([row for row in rows if row["arm"] == arm.name]),
                    "execution": "measured",
                }
                for arm in spec.arms
            }
            | {
                name: {
                    **summarize([record(row) for row in sequence(value["rows"])]),
                    "execution": "historical",
                }
                for name, value in cached.items()
            },
            "code_sha256": code,
            "by_category": partitions(
                rows + historical_rows,
                "category",
                lambda values: {
                    name: {
                        **summarize([row for row in values if row["arm"] == name]),
                        "execution": "historical" if name in cached else "measured",
                    }
                    for name in names
                },
            ),
            "historical_timing_scope": "separate executions; descriptive timings only"
            if cached
            else None,
            "historical_outcomes": partitions(historical_rows, "arm", summarize) if cached else {},
            "paired_task_rate_against_baseline": {}
            if cached
            else {
                arm.name: paired(
                    [{**row, "tokens": 1} for row in rows],
                    arm.name,
                    spec.arms[1].name,
                    family=len(spec.arms) - 2,
                )
                for arm in spec.arms[2:]
            },
            "scope": "controlled live filesystem/tool tasks; all tasks and failures count; deployment transfer requires application traces",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        store.atomic(
            store.root / "failure.json",
            canonical({"error": str(error), "completed_tasks": len(rows)}),
        )
        raise


def execute(
    port: int, task: Record, sandbox: Sandbox, context: int, speculative: bool
) -> tuple[Record, list[Record], list[Record]]:
    messages = [dict(record(value)) for value in sequence(task["messages"])]
    responses: list[Record] = []
    model_seconds, tool_seconds = 0.0, 0.0
    tokens, proposed, accepted = 0, 0, 0
    began = now()
    reason = "turn-limit"
    followup = task.get("followup")
    phase_checks: list[Record] = []
    for turn in range(LIVE_TURNS):
        if now() - began >= LIVE_SECONDS:
            reason = "task-deadline"
            break
        rendered = request(port, "apply-template", {"messages": messages, "tools": TOOLS})
        identifiers = sequence(
            request(
                port,
                "tokenize",
                {"content": rendered["prompt"], "add_special": True, "parse_special": True},
            )["tokens"]
        )
        if len(identifiers) + integer(task["output_cap"]) > context:
            reason = "context-limit"
            break
        start = now()
        query = payload(identifiers, integer(task["output_cap"]), 0, 1.0, 0, 0.0, 8101)
        query["cache_prompt"] = turn > 0
        response = request(port, "completion", query)
        model_seconds += now() - start
        responses.append(response)
        verify_generation(response, query, speculative)
        if not valid_sample(response, integer(task["output_cap"])):
            raise ValueError("live native response has invalid tokens or stopping")
        tokens += len(sequence(response["tokens"]))
        counters = native_counts(response, speculative)
        proposed += integer(counters["proposed"])
        accepted += integer(counters["accepted"])
        content = response["content"]
        if not isinstance(content, str):
            raise ValueError("native content is not text")
        try:
            selected = calls(content)
        except (ValueError, KeyError) as error:
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Tool format error: " + str(error)},
                ]
            )
            continue
        if not selected:
            messages.append({"role": "assistant", "content": content})
            if followup is not None:
                start = now()
                phase_checks.append(sandbox.test())
                sandbox.advance()
                tool_seconds += now() - start
                messages.append({"role": "user", "content": string(followup)})
                followup = None
                continue
            reason = "model-finished"
            break
        ids = ["call_" + str(turn) + "_" + str(index) for index in range(len(selected))]
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                    for call, (name, arguments) in zip(ids, selected, strict=True)
                ],
            }
        )
        for call, (name, arguments) in zip(ids, selected, strict=True):
            start = now()
            try:
                reply = sandbox.invoke(name, arguments)
            except (OSError, ValueError) as error:
                reply = "Tool error: " + (
                    str(error.strerror) if isinstance(error, OSError) else str(error)
                )
            tool_seconds += now() - start
            bounded = (
                reply
                if len(reply) <= 16384
                else reply[:16384] + "\n[Output truncated; request a smaller line range.]"
            )
            messages.append({"role": "tool", "tool_call_id": call, "content": bounded})
    start = now()
    checks = sandbox.test()
    tool_seconds += now() - start
    elapsed = now() - began
    return (
        {
            "completed": checks["passed"] is True
            and followup is None
            and all(check["passed"] is True for check in phase_checks),
            "phase_checks": phase_checks,
            "termination": reason,
            "checks": checks,
            "turns": len(responses),
            "tokens": tokens,
            "wall_seconds": elapsed,
            "model_seconds": model_seconds,
            "tool_seconds": tool_seconds,
            "overhead_seconds": max(0.0, elapsed - model_seconds - tool_seconds),
            "native": {"proposed": proposed, "accepted": accepted},
        },
        messages,
        responses,
    )


def summarize(rows: list[Record]) -> Record:
    counters = totals(rows)
    seconds = sum(number(row["wall_seconds"]) for row in rows)
    completed = sum(row["completed"] is True for row in rows)
    return {
        "tasks": len(rows),
        "completed": completed,
        "completion_rate": completed / len(rows),
        "elapsed_seconds": seconds,
        "seconds_per_assigned_task": seconds / len(rows),
        "model_seconds": sum(number(row["model_seconds"]) for row in rows),
        "tool_seconds": sum(number(row["tool_seconds"]) for row in rows),
        "correct": all(row["correct"] is True for row in rows),
        "tokens": counters["tokens"],
        "proposed": counters["proposed"],
        "accepted": counters["accepted"],
        "acceptance": counters["acceptance"],
    }

import json
import re
from pathlib import Path
from typing import cast

from benchmarks.comparison import Comparison
from benchmarks.measurement import now, request
from benchmarks.runner import native_counts, reconcile, signature, valid_sample, verify_generation
from benchmarks.sandbox import TOOLS, Sandbox
from benchmarks.serving import session
from benchmarks.statistics import paired, partitions
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.device import device_lease
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
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    rows: list[Record] = []
    references: dict[str, str] = {}
    try:
        for rep in range(spec.repeats):
            arms = list(spec.arms)
            if rep:
                arms = arms[rep % len(arms) :] + arms[: rep % len(arms)]
            for arm in arms:
                cell = Store(store.root / (str(rep) + "-" + arm.name))
                responses: list[Record] = []
                with session(spec, arm, cell) as port:
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
        if source_identity(code_root) != code:
            raise ValueError("live benchmark source changed during execution")
        result: Record = {
            "status": "complete",
            "role": spec.role,
            "tier": spec.tier,
            "rows": rows,
            "summary": {
                arm.name: summarize([row for row in rows if row["arm"] == arm.name])
                for arm in spec.arms
            },
            "code_sha256": code,
            "by_category": partitions(rows, "category", summarize),
            "paired_task_rate_against_baseline": {
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
    for turn in range(16):
        if now() - began >= 900:
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
        payload: Record = {
            "prompt": identifiers,
            "n_predict": task["output_cap"],
            "temperature": 0,
            "seed": 8101,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "cache_prompt": turn > 0,
            "return_tokens": True,
        }
        response = request(port, "completion", payload)
        model_seconds += now() - start
        responses.append(response)
        verify_generation(response, payload, speculative)
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
    }

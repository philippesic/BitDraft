import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import cast

from benchmarks.comparison import Comparison
from benchmarks.measurement import now, pause, port_number, ready, request
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, reference


def signature(response: Record) -> Record:
    return {
        key: response[key]
        for key in ("tokens", "content", "stop_type", "stopping_word", "truncated")
    }


def native_counts(response: Record, speculative: bool) -> Record:
    timings = record(response["timings"])
    if not speculative:
        return {"proposed": 0, "accepted": 0, "acceptance": None}
    proposed, accepted = integer(timings["draft_n"]), integer(timings["draft_n_accepted"])
    if not 0 <= accepted <= proposed:
        raise ValueError("native draft counters inconsistent")
    return {
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
    }


def totals(rows: list[Record]) -> Record:
    if not rows:
        raise ValueError("empty comparison")
    seconds = sum(number(row["wall_seconds"]) for row in rows)
    tokens = sum(integer(row["tokens"]) for row in rows)
    proposed = sum(integer(record(row["native"])["proposed"]) for row in rows)
    accepted = sum(integer(record(row["native"])["accepted"]) for row in rows)
    if seconds <= 0:
        raise ValueError("nonpositive measured request time")
    return {
        "requests": len(rows),
        "tokens": tokens,
        "wall_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
        "correct": all(row["correct"] is True for row in rows),
    }


def reconcile(log: str, responses: list[Record]) -> None:
    pattern = r"#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)"
    matches = list(re.finditer(pattern, log))
    if len(matches) != len(responses):
        raise ValueError("native final log does not reconcile with request count")
    last_proposed, last_accepted = 0, 0
    for index, (match, response) in enumerate(zip(matches, responses, strict=True), 1):
        if int(match[1]) != index:
            raise ValueError("native request boundary is ambiguous")
        proposed, accepted = int(match[6]), int(match[7])
        timings = record(response["timings"])
        if (proposed - last_proposed, accepted - last_accepted) != (
            timings["draft_n"],
            timings["draft_n_accepted"],
        ):
            raise ValueError("native response and independent final log disagree")
        last_proposed, last_accepted = proposed, accepted


def compare(specification: Path, output: Path) -> Record:
    spec = Comparison.load(specification)
    spec.verify()
    if output.exists():
        raise ValueError("comparison output must be new; preserve prior attempts")
    with device_lease("cuda"):
        return compare_leased(specification, output, spec)


def compare_leased(specification: Path, output: Path, spec: Comparison) -> Record:
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code_identity = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code_identity, 0)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    store.atomic(store.root / "workload.json", spec.workload.path.read_bytes())
    store.atomic(
        store.root / "owner.json",
        canonical(cast(Record, asdict(ProcessIdentity.read(os.getpid())))),
    )
    rows: list[Record] = []
    references: dict[str, Record] = {}
    try:
        for rep in range(spec.repeats):
            arms = list(spec.arms)
            if rep:
                shift = rep % len(arms)
                arms = arms[shift:] + arms[:shift]
            for arm in arms:
                cell = Store(store.root / (str(rep) + "-" + arm.name))
                port = port_number()
                argv = [
                    str(spec.binary.path),
                    "-m",
                    str(spec.target.path),
                    "-ngl",
                    "99",
                    "-c",
                    str(spec.context),
                    "-np",
                    "1",
                    "-b",
                    "512",
                    "-ub",
                    "512",
                    "-t",
                    "4",
                    "-fa",
                    "on",
                    "--cache-type-k",
                    "f16",
                    "--cache-type-v",
                    "f16",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--metrics",
                    "--jinja",
                    "--reasoning",
                    "off",
                    "--no-webui",
                    "--no-warmup",
                    "--spec-type",
                    "draft-dflash" if arm.drafter else "none",
                    "-lv",
                    "2",
                ]
                if arm.drafter:
                    argv += [
                        "-md",
                        str(arm.drafter.path),
                        "-ngld",
                        "99",
                        "--cache-type-k-draft",
                        "f16",
                        "--cache-type-v-draft",
                        "f16",
                        "--spec-draft-n-max",
                        str(spec.depth),
                        "--spec-draft-n-min",
                        "0",
                        "--spec-draft-p-min",
                        "0",
                        "--spec-draft-sampling",
                        "greedy",
                    ]
                cell.atomic(
                    cell.root / "command.json",
                    canonical(
                        {
                            "argv": argv,
                            "binary": reference(spec.binary),
                            "target": reference(spec.target),
                        }
                    ),
                )
                responses: list[Record] = []
                with (cell.root / "server.log").open("wb") as log:
                    process = subprocess.Popen(
                        argv,
                        env=native_environment(spec.binary.path.parent),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        deadline = now() + 180
                        while not ready(port):
                            if process.poll() is not None or now() >= deadline:
                                raise ValueError("native server startup failed; inspect server.log")
                            pause(0.25)
                        owner = ProcessIdentity.read(process.pid)
                        cell.atomic(
                            cell.root / "server-owner.json", canonical(cast(Record, asdict(owner)))
                        )
                        cell.atomic(
                            cell.root / "library-maps.txt",
                            Path(f"/proc/{process.pid}/maps").read_bytes(),
                        )
                        tasks = [
                            spec.tasks[i % len(spec.tasks)] for i in range(spec.warmups)
                        ] + list(spec.tasks)
                        for index, task in enumerate(tasks):
                            rendered = request(
                                port,
                                "apply-template",
                                {"messages": task["messages"], "tools": task.get("tools", [])},
                            )
                            prompt = string(rendered["prompt"])
                            identifiers = sequence(
                                request(
                                    port,
                                    "tokenize",
                                    {"content": prompt, "add_special": True, "parse_special": True},
                                )["tokens"]
                            )
                            if len(identifiers) + integer(task["output_cap"]) > spec.context:
                                raise ValueError(
                                    "task exceeds context; preserve workload and revise explicitly"
                                )
                            payload: Record = {
                                "prompt": identifiers,
                                "n_predict": task["output_cap"],
                                "temperature": 0,
                                "seed": 8101,
                                "cache_prompt": False,
                                "return_tokens": True,
                            }
                            cell.atomic(cell.root / f"{index}-request.json", canonical(payload))
                            began = now()
                            response = request(port, "completion", payload)
                            elapsed = now() - began
                            cell.atomic(cell.root / f"{index}-response.json", canonical(response))
                            responses.append(response)
                            current = signature(response)
                            key = string(task["id"])
                            if arm.name == "target":
                                if key in references and references[key] != current:
                                    raise ValueError("target-only generation is not repeatable")
                                references[key] = current
                            correct = key in references and references[key] == current
                            metrics = native_counts(response, arm.drafter is not None)
                            if index >= spec.warmups:
                                row: Record = {
                                    "rep": rep,
                                    "arm": arm.name,
                                    "task": key,
                                    "group": task["group"],
                                    "category": task["category"],
                                    "input_tokens": len(identifiers),
                                    "tokens": len(sequence(response["tokens"])),
                                    "wall_seconds": elapsed,
                                    "correct": correct,
                                    "native": metrics,
                                }
                                rows.append(row)
                                store.atomic(
                                    store.root / "progress.json", canonical({"rows": rows})
                                )
                            if not correct:
                                raise ValueError(
                                    "greedy token/text/stopping mismatch: " + arm.name + "/" + key
                                )
                    finally:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=15)
                        cell.atomic(
                            cell.root / "release.json",
                            canonical({"returncode": process.returncode, "reaped": True}),
                        )
                if arm.drafter:
                    reconcile((cell.root / "server.log").read_text(), responses)
                    if not any(
                        integer(record(row["native"])["proposed"])
                        for row in rows
                        if row["arm"] == arm.name
                    ):
                        raise ValueError(
                            "fallback-only speculative arm; activation not established"
                        )
        if source_identity(code_root) != code_identity:
            raise ValueError(
                "executed source changed during comparison; preserve and rerun affected evidence"
            )
        aggregate: Record = {
            arm.name: totals([row for row in rows if row["arm"] == arm.name]) for arm in spec.arms
        }
        result: Record = {
            "status": "complete",
            "role": spec.role,
            "tier": spec.tier,
            "rows": rows,
            "summary": aggregate,
            "code_sha256": code_identity,
            "scientific": "matched request panel only; live workflow and harness claims require their panels",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        store.atomic(
            store.root / "failure.json",
            canonical({"status": "failed", "error": str(error), "completed_requests": len(rows)}),
        )
        raise

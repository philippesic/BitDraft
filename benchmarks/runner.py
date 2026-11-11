import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import cast

from benchmarks.comparison import Arm, Comparison
from benchmarks.coverage import length_band, selection_order
from benchmarks.measurement import now, request
from benchmarks.serving import session
from benchmarks.statistics import paired, partitions
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, identity


def signature(response: Record) -> Record:
    return {
        key: response[key]
        for key in ("tokens", "content", "stop_type", "stopping_word", "truncated")
    }


def valid_sample(response: Record, cap: int) -> bool:
    tokens = sequence(response["tokens"])
    return (
        bool(tokens)
        and len(tokens) <= cap
        and all(type(token) is int and integer(token) >= 0 for token in tokens)
        and response.get("truncated") is False
        and response.get("stop_type") in {"eos", "limit"}
        and (response["stop_type"] != "limit" or len(tokens) == cap)
    )


def verify_generation(response: Record, payload: Record, speculative: bool) -> None:
    settings = record(response["generation_settings"])
    for key in ("seed", "top_k", "n_predict"):
        if integer(settings[key]) != integer(payload[key]):
            raise ValueError("native generation setting differs from request: " + key)
    for key in ("temperature", "top_p", "min_p"):
        if abs(number(settings[key]) - number(payload[key])) > 1e-6:
            raise ValueError("native sampling setting differs from request: " + key)
    if settings.get("ignore_eos") is not False or settings.get("grammar") != "":
        raise ValueError(
            "native stopping/grammar settings differ from the qualified request policy"
        )
    if any(
        number(settings[key]) != expected
        for key, expected in (
            ("repeat_penalty", 1.0),
            ("frequency_penalty", 0.0),
            ("presence_penalty", 0.0),
            ("dynatemp_range", 0.0),
        )
    ):
        raise ValueError("unrequested native penalty or dynamic temperature")
    modes = string(settings["speculative.types"])
    if ("draft-dflash" in modes) != speculative:
        raise ValueError("actual native speculative mode differs from the selected arm")


def native_counts(response: Record, speculative: bool) -> Record:
    timings = record(response["timings"])
    if not speculative:
        return {"proposed": 0, "accepted": 0, "acceptance": None}
    if ("draft_n" in timings) != ("draft_n_accepted" in timings):
        raise ValueError("incomplete native draft counters")
    proposed, accepted = (
        integer(timings.get("draft_n", 0)),
        integer(timings.get("draft_n_accepted", 0)),
    )
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
        timings = native_counts(response, True)
        if (proposed - last_proposed, accepted - last_accepted) != (
            timings["proposed"],
            timings["accepted"],
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


def execution_blocks(
    spec: Comparison,
) -> list[tuple[int, int, tuple[Record, ...], tuple[Arm, ...], int]]:
    grouped = spec.tier in {"full", "harness"}
    count = len(spec.arms) if grouped else 1
    ordered = tuple(sorted(spec.tasks, key=selection_order)) if grouped else spec.tasks
    blocks: list[tuple[int, int, tuple[Record, ...], tuple[Arm, ...], int]] = []
    for rep in range(spec.repeats):
        for group in range(count):
            shift = (rep + group) % len(spec.arms)
            arms = spec.arms[shift:] + spec.arms[:shift]
            blocks.extend(
                (rep, group, ordered[group::count], arms, seed) for seed in spec.sampling.seeds
            )
    return blocks


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
    pending: dict[str, list[tuple[Record, Record | None]]] = {}
    try:
        for rep, group, selected, arms, seed in execution_blocks(spec):
            for arm in arms:
                cell = Store(
                    store.root
                    / (
                        str(rep)
                        + "-"
                        + str(group)
                        + "-"
                        + (str(seed) + "-" if spec.sampling.temperature else "")
                        + arm.name
                    )
                )
                responses: list[Record] = []
                with session(spec, arm, cell) as port:
                    tasks = [selected[i % len(selected)] for i in range(spec.warmups)] + list(
                        selected
                    )
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
                        ancestry = record(task.get("ancestry", {}))
                        if (
                            "token_sha256" in ancestry
                            and identity({"tokens": identifiers}) != ancestry["token_sha256"]
                        ):
                            raise ValueError(
                                "native serving template/tokenizer differs from the frozen workload"
                            )
                        if len(identifiers) + integer(task["output_cap"]) > spec.context:
                            raise ValueError(
                                "task exceeds context; preserve workload and revise explicitly"
                            )
                        payload: Record = {
                            "prompt": identifiers,
                            "n_predict": task["output_cap"],
                            "temperature": spec.sampling.temperature,
                            "seed": seed,
                            "top_p": spec.sampling.top_p,
                            "top_k": spec.sampling.top_k,
                            "min_p": spec.sampling.min_p,
                            "cache_prompt": False,
                            "return_tokens": True,
                        }
                        cell.atomic(cell.root / f"{index}-request.json", canonical(payload))
                        began = now()
                        response = request(port, "completion", payload)
                        elapsed = now() - began
                        cell.atomic(cell.root / f"{index}-response.json", canonical(response))
                        verify_generation(response, payload, arm.drafter is not None)
                        responses.append(response)
                        current = signature(response)
                        key = string(task["id"])
                        if arm.name == "target" and not spec.sampling.temperature:
                            if key in references and references[key] != current:
                                raise ValueError("target-only generation is not repeatable")
                            references[key] = current
                            for observed, prior_row in pending.pop(key, []):
                                if observed != current:
                                    raise ValueError(
                                        "greedy output/stopping mismatch before target reference: "
                                        + key
                                    )
                                if prior_row is not None:
                                    prior_row["correct"] = True
                        correct = (
                            valid_sample(response, integer(task["output_cap"]))
                            if spec.sampling.temperature
                            else references[key] == current
                            if key in references
                            else None
                        )
                        metrics = native_counts(response, arm.drafter is not None)
                        row: Record | None = None
                        if index >= spec.warmups:
                            row = {
                                "rep": rep,
                                "seed": seed,
                                "order_group": group,
                                "arm": arm.name,
                                "task": key,
                                "group": task["group"],
                                "category": task["category"],
                                "input_tokens": len(identifiers),
                                "length_band": length_band(len(identifiers)),
                                "harness": task.get("harness", "default"),
                                "stop_type": response["stop_type"],
                                "cap_hit": response["stop_type"] == "limit",
                                "tokens": len(sequence(response["tokens"])),
                                "wall_seconds": elapsed,
                                "correct": correct,
                                "native": metrics,
                            }
                            rows.append(row)
                            store.atomic(store.root / "progress.json", canonical({"rows": rows}))
                        if correct is None:
                            pending.setdefault(key, []).append((current, row))
                        if correct is False:
                            raise ValueError(
                                "greedy token/text/stopping mismatch: " + arm.name + "/" + key
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
        if pending:
            raise ValueError("unverified requests remain without a target-only reference")
        if source_identity(code_root) != code_identity:
            raise ValueError(
                "executed source changed during comparison; preserve and rerun affected evidence"
            )
        aggregate: Record = {
            arm.name: totals([row for row in rows if row["arm"] == arm.name]) for arm in spec.arms
        }
        result: Record = {
            "status": "complete",
            "diagnostic": spec.diagnostic,
            "sampled": spec.sampling.temperature > 0,
            "distribution_qualified": spec.sampling.qualification is not None,
            "temperature": spec.sampling.temperature,
            "seeds": list(spec.sampling.seeds),
            "role": spec.role,
            "tier": spec.tier,
            "rows": rows,
            "summary": aggregate,
            "by_category": partitions(
                rows,
                "category",
                lambda values: {
                    arm.name: totals([row for row in values if row["arm"] == arm.name])
                    for arm in spec.arms
                },
            ),
            "by_length": partitions(
                rows,
                "length_band",
                lambda values: {
                    arm.name: totals([row for row in values if row["arm"] == arm.name])
                    for arm in spec.arms
                },
            ),
            "paired_against_baseline": {}
            if spec.sampling.temperature
            else {
                arm.name: paired(rows, arm.name, spec.arms[1].name, family=len(spec.arms) - 2)
                for arm in spec.arms[2:]
            },
            "code_sha256": code_identity,
            "scientific": "sampled descriptive robustness; unequal outputs cannot rank checkpoints by throughput"
            if spec.sampling.temperature
            else "matched request panel only; live workflow and harness claims require their panels",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        store.atomic(
            store.root / "failure.json",
            canonical({"status": "failed", "error": str(error), "completed_requests": len(rows)}),
        )
        raise

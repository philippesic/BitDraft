import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

from benchmarks.comparison import Arm, Comparison
from benchmarks.coverage import length_band, selection_order
from benchmarks.measurement import deadline, now, request
from benchmarks.metrics import (
    native_counts as native_counts,
)
from benchmarks.metrics import (
    reconcile as reconcile,
)
from benchmarks.metrics import (
    signature as signature,
)
from benchmarks.metrics import (
    totals as totals,
)
from benchmarks.metrics import (
    valid_sample as valid_sample,
)
from benchmarks.metrics import (
    verify_generation as verify_generation,
)
from benchmarks.runtime_policy import payload
from benchmarks.serving import session
from benchmarks.statistics import paired, partitions
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, identity


def compare(specification: Path, output: Path) -> Record:
    spec = Comparison.load(specification)
    spec.verify()
    if output.exists():
        raise ValueError("comparison output must be new; preserve prior attempts")
    with deadline(120 if spec.tier == "screen" and spec.references else None), device_lease("cuda"):
        return compare_leased(specification, output, spec)


def execution_blocks(
    spec: Comparison,
) -> list[tuple[int, int, tuple[Record, ...], tuple[Arm, ...], int]]:
    count = spec.block_count()
    grouped = count > 1
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


def compare_leased(
    specification: Path, output: Path, spec: Comparison, managed_group: bool = False
) -> Record:
    began_panel = now()
    store = Store(output)
    store.atomic(store.root / "gpu.txt", cuda_identity().encode())
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
    cached = spec.cached()
    if any(value["gpu"] != cuda_identity() for value in cached.values()):
        raise ValueError(
            "reference hardware/driver differs; establish new controls before comparison"
        )
    references = (
        {key: record(value) for key, value in record(cached["target"]["signatures"]).items()}
        if cached
        else {}
    )
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
                with session(spec, arm, cell, managed_group) as port:
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
                        query = payload(
                            identifiers,
                            integer(task["output_cap"]),
                            spec.sampling.temperature,
                            spec.sampling.top_p,
                            spec.sampling.top_k,
                            spec.sampling.min_p,
                            seed,
                        )
                        cell.atomic(cell.root / f"{index}-request.json", canonical(query))
                        began = now()
                        response = request(port, "completion", query)
                        elapsed = now() - began
                        cell.atomic(cell.root / f"{index}-response.json", canonical(response))
                        verify_generation(response, query, arm.drafter is not None)
                        if not valid_sample(response, integer(task["output_cap"])):
                            raise ValueError("invalid native tokens or stopping")
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
        historical_rows = [
            record(row) for value in cached.values() for row in sequence(value["rows"])
        ]
        all_rows = rows + historical_rows
        names = [arm.name for arm in spec.arms] + list(cached)
        aggregate: Record = {
            name: {
                **totals([row for row in all_rows if row["arm"] == name]),
                "execution": "historical" if name in cached else "measured",
            }
            for name in names
        }
        result: Record = {
            "status": "complete",
            "protocol": spec.reference_protocol(),
            "diagnostic": spec.diagnostic,
            "sampled": spec.sampling.temperature > 0,
            "distribution_qualified": spec.sampling.qualification is not None,
            "temperature": spec.sampling.temperature,
            "seeds": list(spec.sampling.seeds),
            "role": spec.role,
            "tier": spec.tier,
            "rows": rows,
            "historical_references": {name: value["reference"] for name, value in cached.items()},
            "historical_comparisons": historical_comparisons(rows, cached),
            "elapsed_seconds": now() - began_panel,
            "summary": aggregate,
            "by_category": partitions(
                all_rows,
                "category",
                lambda values: {
                    name: totals([row for row in values if row["arm"] == name]) for name in names
                },
            ),
            "by_length": partitions(
                all_rows,
                "length_band",
                lambda values: {
                    name: totals([row for row in values if row["arm"] == name]) for name in names
                },
            ),
            "paired_against_baseline": {}
            if spec.sampling.temperature or cached
            else {
                arm.name: paired(rows, arm.name, spec.arms[1].name, family=len(spec.arms) - 2)
                for arm in spec.arms[2:]
            },
            "by_harness": partitions(
                all_rows,
                "harness",
                lambda values: {
                    "summary": {
                        name: {
                            **totals([row for row in values if row["arm"] == name]),
                            "execution": "historical" if name in cached else "measured",
                        }
                        for name in names
                    },
                    "paired_against_baseline": {}
                    if spec.sampling.temperature or cached
                    else {
                        arm.name: paired(
                            values, arm.name, spec.arms[1].name, family=len(spec.arms) - 2
                        )
                        for arm in spec.arms[2:]
                    },
                    "historical_comparisons": historical_comparisons(
                        [row for row in values if row["arm"] not in cached], cached
                    )
                    if cached
                    else {},
                },
            )
            if spec.tier in {"harness", "prefix"}
            else {},
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


def historical_comparisons(rows: list[Record], cached: dict[str, Record]) -> Record:
    output: Record = {}
    for arm in sorted({string(row["arm"]) for row in rows}):
        for name, value in cached.items():
            if name == "target":
                continue
            tasks = {row["task"] for row in rows if row["arm"] == arm}
            old = [record(row) for row in sequence(value["rows"]) if record(row)["task"] in tasks]
            statistics = paired(rows + old, arm, name)
            output[arm + "/" + name] = {
                key: item for key, item in statistics.items() if "speed" not in key
            } | {
                "historical_speed_ratio": statistics["speed_ratio"],
                "timing_scope": "separate executions; historical descriptive ratio, no paired timing claim",
            }
    return output

from dataclasses import replace
from pathlib import Path

from benchmarks.comparison import Comparison, protocol_covers, routine_sampled
from benchmarks.scenarios import faster_verdict, level2_decision, scenario_report, task_scenario
from benchmarks.statistics import paired
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.identity import source_identity
from w1a1_eagle.records import artifact, canonical, read


def report(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) != {"schema", "level", "candidate", "panels", "confirmation"}
        or raw["schema"] != "scenario-report-v1"
    ):
        raise ValueError("unsupported scenario report specification")
    level = integer(raw["level"])
    if level not in {2, 3} or output.exists():
        raise ValueError("new output and level 2 or 3 required")
    root = specification.resolve().parent
    candidate = string(raw["candidate"])
    panels: Record = {}
    scenarios: Record = {}
    identities: set[tuple[str, ...]] = set()
    model_identities: set[str] = set()
    groups_seen: set[str] = set()
    main_comparisons: Record = {}
    for name, value in record(raw["panels"]).items():
        spec, result, cached = load_panel(value, root)
        identities.add(spec.runtime_identity())
        selected_arm = next((arm for arm in spec.arms if arm.name == candidate), None)
        if selected_arm is None or selected_arm.drafter is None:
            raise ValueError("candidate must be freshly measured in each report panel")
        model_identities.add(selected_arm.drafter.sha256)
        groups_seen.update(string(task["group"]) for task in spec.tasks)
        rows = [record(row) for row in sequence(result["rows"])] + [
            record(row) for reference in cached.values() for row in sequence(reference["rows"])
        ]
        groups = scenario_report(rows, spec, cached, record(result.get("resident_gpu_mib", {})))
        if name == "requests":
            if level == 2 and spec.tier not in {"comparison", "full"}:
                raise ValueError("Level 2 requests must be the broader request panel")
            for control in ("baseline", "prior-leader"):
                if any(row["arm"] == control for row in rows):
                    main_comparisons[candidate + "/" + control] = paired(rows, candidate, control)
        panels[name] = {
            "evidence": value,
            "scenarios": groups,
            "coverage_qualified": {
                scenario: qualified_scenario(spec, level, scenario) for scenario in groups
            },
            "elapsed_seconds": result.get("elapsed_seconds"),
            "arm_elapsed_seconds": result.get("arm_elapsed_seconds"),
            "resident_gpu_mib": result.get("resident_gpu_mib"),
            "memory_scope": result.get("memory_scope", "not recorded"),
            "projection": result.get("projection", False),
            "arm_depths": spec.arm_depths(),
            "selected_tasks": len(spec.tasks),
            "selected_seeds": list(spec.sampling.seeds),
            "source_elapsed_seconds": result.get("source_elapsed_seconds"),
            "source_arm_elapsed_seconds": result.get("source_arm_elapsed_seconds"),
        }
        if name not in {"screen", "prefix", "live"}:
            if set(scenarios) & set(groups):
                raise ValueError(
                    "primary scenario appears in multiple panels; do not pool overlapping requests"
                )
            scenarios.update(
                {
                    scenario: value
                    for scenario, value in groups.items()
                    if qualified_scenario(spec, level, scenario)
                }
            )
    if len(identities) != 1 or len(model_identities) != 1:
        raise ValueError("report panels must share target, runtime, context, role and candidate")
    verdict = faster_verdict(scenarios, candidate)
    if raw["confirmation"] is not None:
        confirmation = artifact(raw["confirmation"], root)
        confirmation.verify()
        fresh = read(confirmation.path)
        fresh_panels = record(fresh.get("panels", {}))
        fresh_groups: set[str] = set()
        confirmed_scenarios: Record = {}
        if (
            fresh.get("schema") != "scenario-report-result-v1"
            or fresh.get("level") != 3
            or fresh.get("candidate_sha256") != next(iter(model_identities))
        ):
            raise ValueError("confirmation must be a Level 3 report for the same candidate content")
        for name, panel in fresh_panels.items():
            confirmed_spec, measured, confirmed_cached = load_panel(
                record(panel)["evidence"], confirmation.path.parent
            )
            if (
                confirmed_spec.role != "confirmation"
                or confirmed_cached
                or confirmed_spec.runtime_identity()[:-1] != next(iter(identities))[:-1]
            ):
                raise ValueError("confirmation requires fresh arms and matching serving conditions")
            model = next(
                (arm.drafter for arm in confirmed_spec.arms if arm.name == candidate), None
            )
            if model is None or model.sha256 != next(iter(model_identities)):
                raise ValueError("confirmation candidate content differs")
            if name not in {"screen", "prefix", "live"}:
                grouped = scenario_report(
                    [record(row) for row in sequence(measured["rows"])], confirmed_spec, {}
                )
                if set(confirmed_scenarios) & set(grouped):
                    raise ValueError("duplicated primary confirmation scenario")
                confirmed_scenarios.update(
                    {
                        scenario: value
                        for scenario, value in grouped.items()
                        if qualified_scenario(confirmed_spec, 3, scenario)
                    }
                )
            fresh_groups.update(string(task["group"]) for task in confirmed_spec.tasks)
        if (
            not fresh_groups
            or fresh_groups & groups_seen
            or faster_verdict(confirmed_scenarios, candidate)["verdict"] != "confirmation-required"
        ):
            raise ValueError(
                "confirmation must pass every scenario on disjoint confirmation groups"
            )
        verdict = faster_verdict(scenarios, candidate, confirmed=True)
    final: Record = {
        "schema": "scenario-report-result-v1",
        "code_sha256": source_identity(Path(__file__).resolve().parents[1]),
        "level": level,
        "candidate": candidate,
        "candidate_sha256": next(iter(model_identities)),
        "panels": panels,
        "primary_scenarios": scenarios,
        "verdict": verdict if level == 3 else None,
        "level2": level2_decision(scenarios, main_comparisons, candidate) if level == 2 else None,
        "scope": "source-group intervals cover request/task variation, not session drift; confirmation is separate",
    }
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "result.json", canonical(final))
    return final


def load_panel(value: object, root: Path) -> tuple[Comparison, Record, dict[str, Record]]:
    descriptor = record(value)
    projected = "result" in descriptor
    if projected and set(descriptor) - {"references"} != {"result", "comparison"}:
        raise ValueError("panel projection requires result and comparison references")
    source_ref = artifact(descriptor["result"] if projected else value, root)
    source_ref.verify()
    result = read(source_ref.path, 64 * 1024 * 1024)
    source = Comparison.load(source_ref.path.parent / "specification.json")
    source.verify()
    validate_panel(source, result)
    if not projected:
        return source, result, source.cached()
    comparison_ref = artifact(descriptor["comparison"], root)
    comparison_ref.verify()
    selected = Comparison.load(comparison_ref.path)
    selected.verify()
    original_tasks = {string(task["id"]): task for task in source.tasks}
    if (
        not protocol_covers(source.reference_protocol(), selected.reference_protocol())
        or selected.arms != source.arms
        or selected.references != source.references
        or selected.repeats != source.repeats
        or selected.warmups != source.warmups
        or any(original_tasks.get(string(task["id"])) != task for task in selected.tasks)
    ):
        raise ValueError("panel projection changes source protocol, arms or task definitions")
    wanted = {string(task["id"]) for task in selected.tasks}
    rows = [
        record(row)
        for row in sequence(result["rows"])
        if record(row)["task"] in wanted
        and record(row).get("seed", 8101) in selected.sampling.seeds
    ]
    subset = selected.tasks != source.tasks or selected.sampling.seeds != source.sampling.seeds
    measured = {
        **result,
        "rows": rows,
        "protocol": selected.reference_protocol(),
        "projection": subset,
        "source_elapsed_seconds": result.get("elapsed_seconds"),
        "source_arm_elapsed_seconds": result.get("arm_elapsed_seconds"),
        "elapsed_seconds": None if subset else result.get("elapsed_seconds"),
        "arm_elapsed_seconds": None if subset else result.get("arm_elapsed_seconds"),
    }
    validate_panel(selected, measured)
    if "references" in descriptor:
        refs = tuple(
            (name, artifact(reference, root))
            for name, reference in record(descriptor["references"]).items()
        )
        if {name for name, _ in refs} != {
            arm.name for arm in selected.arms
        } or selected.repeats != 1:
            raise ValueError("registered panel evidence must cover every measured arm once")
        registered = replace(selected, references=refs).cached()
        for arm in selected.arms:
            saved = registered[arm.name]
            saved_model = record(saved["drafter"])["sha256"] if saved["drafter"] else None
            if (
                artifact(saved["result"], root) != source_ref
                or saved["source_arm"] != arm.name
                or saved_model != (arm.drafter.sha256 if arm.drafter else None)
                or saved["rows"] != [row for row in rows if row["arm"] == arm.name]
            ):
                raise ValueError("registered panel evidence differs from this measured source")
    return selected, measured, selected.cached()


def validate_panel(spec: Comparison, result: Record) -> None:
    if result.get("status") != "complete" or result.get("diagnostic", False):
        raise ValueError("scenario reports require complete clean results")
    if result.get("protocol", spec.reference_protocol()) != spec.reference_protocol():
        raise ValueError("result protocol differs from its specification")
    expected = {
        (arm.name, string(task["id"]), rep, seed)
        for arm in spec.arms
        for task in spec.tasks
        for rep in range(spec.repeats)
        for seed in spec.sampling.seeds
    }
    observed = [
        (
            string(record(row)["arm"]),
            string(record(row)["task"]),
            integer(record(row)["rep"]),
            integer(record(row).get("seed", 8101)),
        )
        for row in sequence(result["rows"])
    ]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("scenario report refuses incomplete or duplicated panels")


def qualified_scenario(spec: Comparison, level: int, scenario: str) -> bool:
    if spec.role == "confirmation":
        selected = [
            task
            for task in spec.tasks
            if task_scenario(task, spec.sampling.temperature > 0) == scenario
        ]
        if len(selected) < (15 if scenario == "agentic-coding" else 20):
            return False
        if scenario == "agentic-coding":
            return spec.tier == "agentic-confirmation"
        if scenario == "thinking-on":
            return spec.sampling.temperature == 0 and all(
                task.get("thinking") is True and task["output_cap"] == 2048 for task in selected
            )
        if scenario == "sampled":
            return (
                routine_sampled(spec.sampling)
                and spec.sampling.qualification is not None
                and len(spec.sampling.seeds) == 3
                and all(
                    task.get("thinking", False) is False and task["output_cap"] == 2048
                    for task in selected
                )
            )
        return True
    if scenario == "thinking-on":
        return spec.tier in (
            {"thinking-suite"} if level == 3 else {"thinking-suite", "thinking-slice"}
        )
    if scenario == "sampled":
        return spec.tier in (
            {"sampled-suite"} if level == 3 else {"sampled-suite", "sampled-slice"}
        )
    if scenario == "agentic-coding":
        return spec.tier in (
            {"agentic-suite"} if level == 3 else {"agentic-suite", "agentic-slice"}
        )
    return spec.tier in {"comparison", "full"}

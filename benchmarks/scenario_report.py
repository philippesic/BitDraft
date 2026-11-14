from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.scenarios import faster_verdict, level2_decision, scenario_report
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
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
    for name, value in record(raw["panels"]).items():
        ref = artifact(value, root)
        ref.verify()
        result = read(ref.path, 64 * 1024 * 1024)
        if result.get("status") != "complete" or result.get("diagnostic", False):
            raise ValueError("scenario reports require complete clean results")
        spec = Comparison.load(ref.path.parent / "specification.json")
        spec.verify()
        identities.add(spec.serving_identity())
        selected_arm = next((arm for arm in spec.arms if arm.name == candidate), None)
        if selected_arm is None or selected_arm.drafter is None:
            raise ValueError("candidate must be freshly measured in each report panel")
        model_identities.add(selected_arm.drafter.sha256)
        groups_seen.update(string(task["group"]) for task in spec.tasks)
        cached = spec.cached()
        rows = [record(row) for row in sequence(result["rows"])] + [
            record(row) for reference in cached.values() for row in sequence(reference["rows"])
        ]
        groups = scenario_report(rows, spec, cached)
        panels[name] = {
            "evidence": value,
            "scenarios": groups,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "resident_gpu_mib": result.get("resident_gpu_mib"),
            "memory_scope": result.get("memory_scope", "not recorded"),
        }
        if name not in {"screen", "prefix", "live"}:
            if set(scenarios) & set(groups):
                raise ValueError(
                    "primary scenario appears in multiple panels; do not pool overlapping requests"
                )
            scenarios.update(groups)
    if len(identities) != 1 or len(model_identities) != 1:
        raise ValueError("report panels must share target, runtime, context, depth and role")
    verdict = faster_verdict(scenarios, candidate)
    if raw["confirmation"] is not None:
        confirmation = artifact(raw["confirmation"], root)
        confirmation.verify()
        fresh = read(confirmation.path)
        fresh_panels = record(fresh.get("panels", {}))
        fresh_groups: set[str] = set()
        if (
            fresh.get("schema") != "scenario-report-result-v1"
            or fresh.get("level") != 3
            or fresh.get("candidate_sha256") != next(iter(model_identities))
        ):
            raise ValueError("confirmation must be a Level 3 report for the same candidate content")
        for panel in fresh_panels.values():
            evidence = artifact(record(panel)["evidence"], confirmation.path.parent)
            evidence.verify()
            confirmed_spec = Comparison.load(evidence.path.parent / "specification.json")
            confirmed_spec.verify()
            if (
                confirmed_spec.role != "confirmation"
                or confirmed_spec.references
                or confirmed_spec.serving_identity()[:-1] != next(iter(identities))[:-1]
            ):
                raise ValueError("confirmation requires fresh arms and matching serving conditions")
            fresh_groups.update(string(task["group"]) for task in confirmed_spec.tasks)
        if (
            not fresh_groups
            or fresh_groups & groups_seen
            or record(fresh.get("verdict", {})).get("verdict") != "confirmation-required"
        ):
            raise ValueError(
                "confirmation must pass every scenario on disjoint confirmation groups"
            )
        verdict = faster_verdict(scenarios, candidate, confirmed=True)
    final: Record = {
        "schema": "scenario-report-result-v1",
        "level": level,
        "candidate": candidate,
        "candidate_sha256": next(iter(model_identities)),
        "panels": panels,
        "primary_scenarios": scenarios,
        "verdict": verdict if level == 3 else None,
        "level2": level2_decision(
            scenarios,
            record(record(scenarios.get("coding-requests", {"comparisons": {}}))["comparisons"]),
            candidate,
        )
        if level == 2
        else None,
        "scope": "source-group intervals cover request/task variation, not session drift; confirmation is separate",
    }
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "result.json", canonical(final))
    return final

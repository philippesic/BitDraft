from pathlib import Path

import pytest

from benchmarks.comparison import Comparison
from benchmarks.runner import prompt_tokens
from benchmarks.scenarios import PRIMARY, faster_verdict, level2_decision, task_scenario
from benchmarks.statistics import paired
from tests.test_references import evidence
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.records import canonical, read


def scenario(lower: float, upper: float = 1.2) -> Record:
    return {
        "summary": {
            "candidate": {"correct": True, "completed_tasks": 12},
            "baseline": {"completed_tasks": 12},
        },
        "comparisons": {
            "candidate/" + arm: {
                "speed_ratio": (lower + upper) / 2,
                "speed_ratio_95_interval": {"lower": lower, "upper": upper},
            }
            for arm in ("target", "baseline", "prior-leader")
        },
    }


def test_verdict_requires_every_scenario_and_fresh_confirmation() -> None:
    values: Record = {name: scenario(1.01) for name in PRIMARY}
    assert faster_verdict(values, "candidate")["verdict"] == "confirmation-required"
    assert faster_verdict(values, "candidate", confirmed=True)["verdict"] == "faster"
    values["agentic-coding"] = scenario(0.99)
    assert faster_verdict(values, "candidate", confirmed=True)["verdict"] == "not-established"
    values.pop("agentic-coding")
    assert record(faster_verdict(values, "candidate")["scenarios"])["agentic-coding"] == "missing"
    values["agentic-coding"] = scenario(1.0)
    assert faster_verdict(values, "candidate")["verdict"] == "not-established"
    values["agentic-coding"] = scenario(1.01)
    record(record(record(values["agentic-coding"])["comparisons"])["candidate/baseline"])[
        "speed_ratio_95_interval"
    ] = {"lower": 1.0, "upper": 1.1}
    assert faster_verdict(values, "candidate")["verdict"] == "confirmation-required"


def test_level2_regression_or_direction_disagreement_requires_level3() -> None:
    main = record(scenario(1.01)["comparisons"])
    slices: Record = {name: scenario(1.01) for name in PRIMARY}
    slices["sampled"] = scenario(0.8, 0.9)
    assert level2_decision(slices, main, "candidate")["level3_required"]
    slices["sampled"] = scenario(0.8, 1.1)
    assert level2_decision(slices, main, "candidate")["level3_required"]
    slices["sampled"] = {"comparisons": {"candidate/baseline": {"historical_speed_ratio": 0.95}}}
    historical = level2_decision(slices, main, "candidate")
    assert historical["level3_required"]
    assert historical["timing_unresolved"]
    assert "sampled: direction disagrees with main panel against baseline" in sequence(
        historical["reasons"]
    )
    slices["sampled"] = scenario(1.01)
    assert not level2_decision(slices, main, "candidate")["level3_required"]
    assert level2_decision(slices, {}, "candidate")["level3_required"]
    candidate = record(record(record(slices["agentic-coding"])["summary"])["candidate"])
    candidate["completed_tasks"] = 11
    assert level2_decision(slices, main, "candidate")["level3_required"]
    candidate["completed_tasks"] = 12
    candidate["correct"] = False
    assert level2_decision(slices, main, "candidate")["level3_required"]


def test_sampled_pairing_keeps_seeds_and_rejects_unmatched_seed() -> None:
    rows: list[Record] = [
        {
            "arm": arm,
            "group": str(group),
            "task": str(group),
            "rep": 0,
            "seed": seed,
            "tokens": 4 if arm == "candidate" else 5,
            "wall_seconds": 1.0,
            "category": "code",
            "native": {"proposed": 2, "accepted": 1},
        }
        for arm in ("candidate", "baseline")
        for group in range(4)
        for seed in (1, 2)
    ]
    result = paired(rows, "candidate", "baseline", draws=100)
    assert result["groups"] == 4 and result["speed_ratio"] == 0.8
    rows[0]["seed"] = 3
    with pytest.raises(ValueError, match="unmatched"):
        paired(rows, "candidate", "baseline", draws=100)


def test_thinking_template_is_explicit_and_legacy_request_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Record] = []

    def request(port: int, route: str, payload: Record) -> Record:
        calls.append(payload)
        return {"prompt": "thinking"} if route == "apply-template" else {"tokens": [1]}

    monkeypatch.setattr("benchmarks.runner.request", request)
    task: Record = {"messages": [{"role": "user", "content": "problem"}], "category": "reasoning"}
    prompt_tokens(0, task, 2, 16)
    assert "chat_template_kwargs" not in calls[0]
    calls.clear()
    prompt_tokens(0, task | {"thinking": True}, 2, 16)
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert task_scenario(task | {"thinking": True}) == "thinking-on"
    assert task_scenario(task, sampled=True) == "sampled"
    assert task_scenario(task | {"scenario": "agentic-coding"}) == "agentic-coding"


def test_thinking_policy_changes_reference_identity_without_changing_legacy(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    original = Comparison.load(path).reference_protocol()
    workload = read(tmp_path / "workload.json")
    from benchmarks.native import file_reference
    from w1a1_eagle.contracts import sequence

    record(sequence(workload["tasks"])[0])["thinking"] = True
    (tmp_path / "workload.json").write_bytes(canonical(workload))
    spec = read(path)
    spec["workload"] = file_reference(tmp_path / "workload.json")
    path.write_bytes(canonical(spec))
    updated = Comparison.load(path).reference_protocol()
    assert original != updated and original["target_arguments"] == updated["target_arguments"]


def test_scenario_report_refuses_partial_evidence_and_missing_scenarios(tmp_path: Path) -> None:
    from benchmarks.native import file_reference
    from benchmarks.scenario_report import report, validate_panel
    from w1a1_eagle.contracts import sequence

    path = evidence(tmp_path)
    comparison = Comparison.load(path)
    result = read(tmp_path / "result.json")
    validate_panel(comparison, result)
    with pytest.raises(ValueError, match="incomplete"):
        validate_panel(comparison, result | {"rows": sequence(result["rows"])[:-1]})
    with pytest.raises(ValueError, match="duplicated"):
        validate_panel(
            comparison, result | {"rows": [*sequence(result["rows"]), sequence(result["rows"])[0]]}
        )
    report_spec = tmp_path / "report.json"
    report_spec.write_bytes(
        canonical(
            {
                "schema": "scenario-report-v1",
                "level": 3,
                "candidate": "baseline",
                "panels": {"requests": file_reference(tmp_path / "result.json")},
                "confirmation": None,
            }
        )
    )
    output = report(report_spec, tmp_path / "report-output")
    assert record(output["verdict"])["verdict"] == "not-established"
    assert record(record(output["verdict"])["scenarios"])["agentic-coding"] == "missing"


def test_small_development_slice_cannot_establish_level3_coverage(tmp_path: Path) -> None:
    from dataclasses import replace

    from benchmarks.scenario_report import qualified_scenario

    spec = Comparison.load(evidence(tmp_path))
    assert not qualified_scenario(spec, 3, "coding-requests")
    assert not qualified_scenario(replace(spec, tier="thinking-slice"), 3, "thinking-on")
    assert qualified_scenario(replace(spec, tier="thinking-slice"), 2, "thinking-on")
    assert qualified_scenario(replace(spec, tier="thinking-suite"), 3, "thinking-on")
    assert not qualified_scenario(replace(spec, role="confirmation"), 3, "coding-requests")


def test_workflow_totals_do_not_infer_model_calls_from_task_count() -> None:
    from benchmarks.scenarios import unit_totals

    rows: list[Record] = [
        {
            "tokens": 12,
            "wall_seconds": 2.0,
            "native": {"accepted": 5, "proposed": 10},
            "completed": completed,
            "correct": True,
        }
        for completed in (True, False)
    ]
    summary = unit_totals(rows)
    assert summary["tasks"] == 2
    assert summary["tokens_per_second"] == 6.0
    assert summary["acceptance"] == 0.5
    assert "derived_rounds" not in summary and "cap_hits" not in summary


def test_fixed_variant_coverage_and_confirmation_settings(tmp_path: Path) -> None:
    from dataclasses import replace

    from benchmarks.comparison import Sampling, require_cohort, routine_sampled
    from benchmarks.scenario_report import qualified_scenario

    tasks: tuple[Record, ...] = tuple(
        {
            "id": str(index),
            "group": str(index),
            "category": "code" if index < 6 else "reasoning",
            "thinking": False,
            "output_cap": 2048,
        }
        for index in range(12)
    )
    require_cohort("sampled-slice", tasks)
    require_cohort(
        "sampled-slice",
        tuple({key: value for key, value in task.items() if key != "thinking"} for task in tasks),
    )
    for change in (
        {"category": "writing"},
        {"group": "1"},
        {"thinking": True},
        {"output_cap": 128},
    ):
        with pytest.raises(ValueError, match="scenario cohort"):
            require_cohort("sampled-slice", (tasks[0] | change, *tasks[1:]))
    require_cohort("thinking-slice", tuple(task | {"thinking": True} for task in tasks))
    spec = Comparison.load(evidence(tmp_path))
    selected = tuple(
        spec.tasks[0]
        | {
            "id": str(index),
            "group": str(index),
            "role": "confirmation",
            "scenario": "thinking-on",
            "thinking": False,
            "output_cap": 2048,
        }
        for index in range(20)
    )
    confirmation = replace(spec, role="confirmation", tasks=selected)
    assert not qualified_scenario(confirmation, 3, "thinking-on")
    thinking = replace(confirmation, tasks=tuple(task | {"thinking": True} for task in selected))
    assert qualified_scenario(thinking, 3, "thinking-on")
    sampled = replace(
        confirmation,
        tasks=tuple(task | {"scenario": "sampled"} for task in selected),
        sampling=Sampling(0.7, 0.9, 0, 0.0, (9101, 9102, 9103), spec.source),
    )
    assert routine_sampled(sampled.sampling)
    assert qualified_scenario(sampled, 3, "sampled")
    assert qualified_scenario(
        replace(
            sampled,
            tasks=tuple(
                {key: value for key, value in task.items() if key != "thinking"}
                for task in sampled.tasks
            ),
        ),
        3,
        "sampled",
    )
    assert not qualified_scenario(
        replace(sampled, sampling=replace(sampled.sampling, temperature=0.1)), 3, "sampled"
    )
    assert not qualified_scenario(
        replace(sampled, sampling=replace(sampled.sampling, seeds=(9101, 9102))), 3, "sampled"
    )


def test_projected_panel_authenticates_full_source_and_registered_arms(tmp_path: Path) -> None:
    from benchmarks.native import file_reference
    from benchmarks.references import register
    from benchmarks.scenario_report import load_panel

    original = evidence(tmp_path)
    source_result = read(tmp_path / "result.json")
    workload = read(tmp_path / "workload.json")
    selected_task = record(sequence(workload["tasks"])[0])
    selected_workload = tmp_path / "selected-workload.json"
    selected_workload.write_bytes(canonical({"schema": 1, "tasks": [selected_task]}))
    selected_spec = read(original) | {"workload": file_reference(selected_workload)}
    selected_path = tmp_path / "selected.json"
    selected_path.write_bytes(canonical(selected_spec))
    references: Record = {}
    for arm in ("target", "baseline"):
        registration = tmp_path / (arm + "-registration.json")
        registration.write_bytes(
            canonical(
                {
                    "schema": 1,
                    "comparison": file_reference(selected_path),
                    "result": file_reference(tmp_path / "result.json"),
                    "arm": arm,
                    "rep": 0,
                }
            )
        )
        references[arm] = register(registration, tmp_path / "registry")["reference"]
    descriptor: Record = {
        "comparison": file_reference(selected_path),
        "result": file_reference(tmp_path / "result.json"),
        "references": references,
    }
    spec, projected, cached = load_panel(descriptor, tmp_path)
    assert len(spec.tasks) == 1 and len(sequence(projected["rows"])) == 2
    assert projected["projection"] is True and projected["arm_elapsed_seconds"] is None
    assert not cached
    with pytest.raises(ValueError, match="every measured arm"):
        load_panel(descriptor | {"references": {"target": references["target"]}}, tmp_path)
    separate = tmp_path / "separate-execution"
    separate.mkdir()
    evidence(separate)
    (separate / "specification.json").write_bytes(
        canonical(read(original) | {"workload": file_reference(separate / "workload.json")})
    )
    for command in separate.glob("*/command.json"):
        captured = read(command)
        command.write_bytes(
            canonical(
                captured
                | {
                    "argv": [
                        str(value).replace(str(separate), str(tmp_path))
                        for value in sequence(captured["argv"])
                    ]
                }
            )
        )
    historical: Record = {}
    for arm in ("target", "baseline"):
        registration = separate / (arm + "-registration.json")
        registration.write_bytes(
            canonical(
                {
                    "schema": 1,
                    "comparison": file_reference(selected_path),
                    "result": file_reference(separate / "result.json"),
                    "arm": arm,
                    "rep": 0,
                }
            )
        )
        historical[arm] = register(registration, separate / "registry")["reference"]
    with pytest.raises(ValueError, match="differs from this measured source"):
        load_panel(descriptor | {"references": historical}, tmp_path)
    (tmp_path / "result.json").write_bytes(
        canonical(source_result | {"rows": sequence(source_result["rows"])[:-1]})
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_panel(descriptor | {"result": file_reference(tmp_path / "result.json")}, tmp_path)
    (tmp_path / "result.json").write_bytes(canonical(source_result))
    selected_workload.write_bytes(
        canonical({"schema": 1, "tasks": [selected_task | {"output_cap": 3}]})
    )
    selected_path.write_bytes(
        canonical(selected_spec | {"workload": file_reference(selected_workload)})
    )
    with pytest.raises(ValueError, match="changes source"):
        load_panel(descriptor | {"comparison": file_reference(selected_path)}, tmp_path)


def test_agentic_scenario_uses_assigned_task_rate_and_keeps_failures(tmp_path: Path) -> None:
    from dataclasses import replace

    from benchmarks.scenarios import scenario_report

    spec = Comparison.load(evidence(tmp_path))
    spec = replace(spec, tasks=tuple(task | {"scenario": "agentic-coding"} for task in spec.tasks))
    measured = read(tmp_path / "result.json")
    rows = [
        record(row)
        | {
            "tokens": 100 if record(row)["arm"] == "baseline" else 10,
            "wall_seconds": 1.0 if record(row)["arm"] == "baseline" else 2.0,
            "raw_wall_seconds": 3.0,
            "completed": record(row)["task"] == "0",
        }
        for row in sequence(measured["rows"])
    ]
    result = record(scenario_report(rows, spec, {})["agentic-coding"])
    comparison = record(record(result["comparisons"])["baseline/target"])
    summary = record(record(result["summary"])["baseline"])
    assert comparison["speed_ratio"] == 2.0
    assert summary["assigned_tasks_per_second"] == 1.0
    assert summary["speed_vs_target"] == 2.0
    assert summary["completed_tasks"] == 1 and summary["tasks"] == 2
    assert summary["raw_wall_seconds"] == 6.0
    assert set(record(result["by_outcome"])) == {"completed", "failed"}


def test_merged_admission_keeps_agentic_calibration_and_request_variants(tmp_path: Path) -> None:
    from benchmarks.native import file_reference

    path = evidence(tmp_path)
    original = read(path)
    template = Comparison.load(path).tasks[0]
    workload = tmp_path / "merged-workload.json"

    def load(tasks: list[Record], changes: Record) -> Comparison:
        workload.write_bytes(canonical({"schema": 1, "tasks": tasks}))
        path.write_bytes(canonical(original | changes | {"workload": file_reference(workload)}))
        return Comparison.load(path)

    agentic = [
        template
        | {
            "id": str(index),
            "project": "owner/project" + str(index // 2),
            "group": "project:owner/project" + str(index // 2),
            "scenario": "agentic-coding",
            "style": ("minimal", "coding", "tools")[index % 3],
            "task_kind": "bug-fix",
            "fixture": {},
        }
        for index in range(12)
    ]
    admitted = load(agentic, {"tier": "agentic-slice"})
    assert admitted.tasks[0]["task_kind"] == "bug-fix"
    assert admitted.reference_protocol()["policy"] == "native-live-v3"
    assert record(admitted.reference_protocol()["live"])["loop_accounting"] == (
        "exclude-detection-call-from-primary; retain-raw-wall"
    )
    calibration: Record = {
        "tier": "pilot",
        "calibration": "target-only",
        "arms": [{"name": "target", "drafter": None}],
    }
    assert len(load(agentic[:1], calibration).arms) == 1
    with pytest.raises(ValueError, match="calibration"):
        load(agentic[:1], calibration | {"arms": original["arms"]})
    requests = [
        template
        | {
            "id": str(index),
            "group": str(index),
            "category": "code" if index < 6 else "reasoning",
            "output_cap": 2048,
            "thinking": True,
        }
        for index in range(12)
    ]
    thinking = load(requests, {"tier": "thinking-slice"})
    assert thinking.reference_protocol()["policy"] == "native-greedy-request-v3"
    assert thinking.reference_protocol()["live"] is None
    sampled: Record = {
        "tier": "sampled-slice",
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "seeds": [8101, 8102],
            "qualification": original["source"],
        },
    }
    with pytest.raises(ValueError, match="scenario cohort"):
        load(requests, sampled)
    sampling = load([task | {"thinking": False} for task in requests], sampled)
    assert sampling.reference_protocol()["policy"] == "native-sampled-request-v3"
    assert sampling.sampling.seeds == (8101, 8102)

from pathlib import Path

import pytest

from benchmarks.comparison import Comparison
from benchmarks.runner import prompt_tokens
from benchmarks.scenarios import PRIMARY, faster_verdict, level2_decision, task_scenario
from benchmarks.statistics import paired
from tests.test_references import evidence
from w1a1_eagle.contracts import Record, record
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
    assert level2_decision({"sampled": scenario(0.8, 0.9)}, main, "candidate")["level3_required"]
    assert level2_decision({"sampled": scenario(0.8, 1.1)}, main, "candidate")["level3_required"]
    assert not level2_decision({name: scenario(1.01) for name in PRIMARY}, main, "candidate")[
        "level3_required"
    ]


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

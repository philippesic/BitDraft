import json
import random
from pathlib import Path

from benchmarks.gate_verdict import RULES, gate_verdict, judge
from benchmarks.native import file_reference
from w1a1_eagle.contracts import Record, record, sequence, string

GUARDRAILS = ["thinking-on", "sampled", "long-input", "agentic-replay", "tool-calls"]
REPORTED = ["held-out-convention"]


def rows(
    scenario: str,
    speeds: dict[str, float],
    tasks: int = 40,
    seeds: tuple[int, ...] = (8101,),
    prefix: str = "",
) -> list[Record]:
    generator = random.Random(scenario + prefix)
    result: list[Record] = []
    for index in range(tasks):
        base = generator.uniform(0.5, 2.0)
        for seed in seeds:
            for arm, speed in speeds.items():
                noise = generator.uniform(0.995, 1.005)
                result.append(
                    {
                        "arm": arm,
                        "scenario": scenario,
                        "task": f"{prefix}{scenario}-{index}",
                        "group": f"{prefix}{scenario}-{index}",
                        "category": "c" + str(index % 2),
                        "rep": 0,
                        "seed": seed,
                        "tokens": 200,
                        "wall_seconds": base / (speed * noise),
                        "correct": True,
                        "native": {"accepted": 60, "proposed": 100},
                    }
                )
    return result


def failures(verdict: Record) -> list[str]:
    return [string(value) for value in sequence(verdict["failures"])]


def gate(speeds: dict[str, dict[str, float]]) -> list[Record]:
    result = rows("general", speeds["general"], prefix="a-") + rows(
        "general", speeds["general"], prefix="b-"
    )
    for name in [*GUARDRAILS, *REPORTED]:
        seeds = (8101, 8102, 8103) if name == "sampled" else (8101,)
        result += rows(name, speeds.get(name, speeds["general"]), seeds=seeds)
    return result


def test_clear_win_passes_against_stock_and_control() -> None:
    data = gate({"general": {"candidate": 1.05, "baseline": 1.0, "control": 1.0}})
    stock = judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)
    control = judge(data, "candidate", "control", RULES["control"], GUARDRAILS, REPORTED)
    assert stock["passed"] is True and control["passed"] is True
    assert stock["reported_only"] == REPORTED
    general = record(record(stock["scenarios"])["general"])
    assert general["groups"] == 80


def test_reported_convention_cannot_fail_the_gate() -> None:
    data = gate(
        {
            "general": {"candidate": 1.05, "baseline": 1.0},
            "held-out-convention": {"candidate": 0.7, "baseline": 1.0},
        }
    )
    assert judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)["passed"]


def test_tie_fails_the_strict_primary_but_passes_the_control_margin() -> None:
    data = gate({"general": {"candidate": 0.99, "baseline": 1.0}})
    stock = judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)
    control = judge(data, "candidate", "baseline", RULES["control"], GUARDRAILS, REPORTED)
    assert stock["passed"] is False
    assert any("primary" in failure for failure in failures(stock))
    assert control["passed"] is True


def test_guardrail_floor_and_parity_count() -> None:
    low = {"candidate": 0.9, "baseline": 1.0}
    data = gate({"general": {"candidate": 1.05, "baseline": 1.0}, "long-input": low})
    verdict = judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)
    assert any("long-input" in failure for failure in failures(verdict))
    slight = {"candidate": 0.985, "baseline": 1.0}
    data = gate(
        {
            "general": {"candidate": 1.05, "baseline": 1.0},
            "thinking-on": slight,
            "sampled": slight,
            "long-input": slight,
        }
    )
    verdict = judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)
    assert any("below parity" in failure for failure in failures(verdict))


def test_missing_scenario_and_invalid_output_fail() -> None:
    data = [
        row
        for row in gate({"general": {"candidate": 1.05, "baseline": 1.0}})
        if row["scenario"] != "agentic-replay"
    ]
    verdict = judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)
    assert "missing scenario agentic-replay" in failures(verdict)
    data = gate({"general": {"candidate": 1.05, "baseline": 1.0}})
    data[0]["correct"] = False
    verdict = judge(data, "candidate", "baseline", RULES["stock"], GUARDRAILS, REPORTED)
    assert verdict["passed"] is False


DEFINITION = Path(__file__).resolve().parents[1] / "benchmarks/definitions/release-gate-v1.json"


def full_gate(speeds: dict[str, dict[str, float]], arms: list[str]) -> list[Record]:
    result = rows("general", speeds["general"], prefix="a-") + rows(
        "general", speeds["general"], prefix="b-"
    )
    for name in [*GUARDRAILS, *REPORTED]:
        seeds = (9201, 9202, 9203) if name == "sampled" else (8101,)
        result += rows(name, speeds.get(name, speeds["general"]), seeds=seeds)
    return [row for row in result if row["arm"] in arms]


def write(path: Path, value: object) -> dict[str, object]:
    path.write_text(json.dumps(value))
    return dict(file_reference(path))


def verdict(
    tmp: Path,
    data: list[Record],
    comparators: dict[str, str],
    drift: list[Record] | None,
    roster: list[Record] | None = None,
) -> Record:
    tasks: dict[str, list[dict[str, str]]] = {}
    for row in roster if roster is not None else data:
        entry = {"id": string(row["task"]), "scenario": string(row["scenario"])}
        if entry not in tasks.setdefault(string(row["scenario"]), []):
            tasks[string(row["scenario"])].append(entry)
    workloads = {
        name: write(tmp / f"workload-{name}.json", {"schema": "gate", "tasks": items})
        for name, items in tasks.items()
    }
    manifest = write(
        tmp / "manifest.json",
        {"definition": file_reference(DEFINITION), "workloads": workloads},
    )
    spec = {
        "schema": "gate-verdict-v2",
        "definition": file_reference(DEFINITION),
        "manifest": manifest,
        "results": [write(tmp / "results.json", {"rows": data})],
        "candidate": "candidate",
        "comparators": comparators,
        "drift": None
        if drift is None
        else {
            "result": write(tmp / "drift.json", {"rows": drift}),
            "first": "baseline",
            "second": "baseline-again",
        },
    }
    (tmp / "spec.json").write_text(json.dumps(spec))
    return gate_verdict(tmp / "spec.json", tmp / "out")


def drift_rows(ratio: float) -> list[Record]:
    return rows("general", {"baseline": ratio, "baseline-again": 1.0}, prefix="a-") + rows(
        "general", {"baseline": ratio, "baseline-again": 1.0}, prefix="b-"
    )


ARMS = ["candidate", "baseline", "control"]
BOTH = {"stock": "baseline", "control": "control"}


def test_complete_evidence_with_both_comparators_and_drift_passes(tmp_path: Path) -> None:
    data = full_gate({"general": {"candidate": 1.05, "baseline": 1.0, "control": 1.0}}, ARMS)
    result = verdict(tmp_path, data, BOTH, drift_rows(1.0))
    assert result["passed"] is True, result["missing_for_release"]
    assert record(result["transfer_evidence"])["complete"] is True


def test_a_missing_comparator_or_drift_run_blocks_the_release(tmp_path: Path) -> None:
    data = full_gate({"general": {"candidate": 1.05, "baseline": 1.0, "control": 1.0}}, ARMS)
    stock_only = verdict(tmp_path, data, {"stock": "baseline"}, drift_rows(1.0))
    assert stock_only["passed"] is False
    assert "control comparator" in sequence(stock_only["missing_for_release"])
    no_drift = verdict(tmp_path, data, BOTH, None)
    assert no_drift["passed"] is False
    assert "reversed-order drift run" in sequence(no_drift["missing_for_release"])


def test_an_omitted_general_file_or_task_fails_the_verdict(tmp_path: Path) -> None:
    data = full_gate({"general": {"candidate": 1.05, "baseline": 1.0, "control": 1.0}}, ARMS)
    partial = [row for row in data if not string(row["task"]).startswith("b-general")]
    result = verdict(tmp_path, partial, BOTH, drift_rows(1.0), roster=data)
    stock = record(record(result["verdicts"])["stock"])
    assert result["passed"] is False
    assert any("incomplete general" in failure for failure in failures(stock))
    one = [row for row in data if not (row["arm"] == "control" and row["task"] == "tool-calls-3")]
    result = verdict(tmp_path, one, BOTH, drift_rows(1.0), roster=data)
    control = record(record(result["verdicts"])["control"])
    assert control["passed"] is False
    assert any("incomplete tool-calls" in failure for failure in failures(control))


def test_sampled_rows_need_all_three_frozen_seeds(tmp_path: Path) -> None:
    data = full_gate({"general": {"candidate": 1.05, "baseline": 1.0, "control": 1.0}}, ARMS)
    two = [row for row in data if row["seed"] != 9203]
    result = verdict(tmp_path, two, BOTH, drift_rows(1.0), roster=data)
    assert result["passed"] is False
    stock = record(record(result["verdicts"])["stock"])
    assert any("wrong seeds" in failure for failure in failures(stock))


def test_drift_is_two_sided(tmp_path: Path) -> None:
    data = full_gate({"general": {"candidate": 1.012, "baseline": 1.0, "control": 1.0}}, ARMS)
    result = verdict(tmp_path, data, BOTH, drift_rows(1.03))
    stock = record(record(result["verdicts"])["stock"])
    assert stock["claimable_lead"] is False
    assert result["passed"] is False
    assert "a general lead above the measured session drift" in sequence(
        result["missing_for_release"]
    )


def test_incomplete_transfer_evidence_is_reported_not_failed(tmp_path: Path) -> None:
    data = full_gate({"general": {"candidate": 1.05, "baseline": 1.0, "control": 1.0}}, ARMS)
    held = [row for row in data if row["task"] != "held-out-convention-0"]
    result = verdict(tmp_path, held, BOTH, drift_rows(1.0), roster=data)
    assert result["passed"] is True
    assert record(result["transfer_evidence"])["complete"] is False

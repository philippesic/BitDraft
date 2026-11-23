import random

from benchmarks.gate_verdict import RULES, judge
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

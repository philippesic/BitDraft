from benchmarks.comparison import Comparison
from benchmarks.metrics import relative_speed, totals
from benchmarks.statistics import paired
from w1a1_eagle.contracts import Record, number, record, string

PRIMARY = (
    "agentic-coding",
    "coding-requests",
    "chat-writing",
    "reasoning",
    "long-documents",
    "thinking-on",
    "sampled",
)


def task_scenario(task: Record, sampled: bool = False) -> str:
    if sampled:
        return "sampled"
    if task.get("thinking") is True:
        return "thinking-on"
    if "scenario" in task:
        return string(task["scenario"])
    return {
        "code": "coding-requests",
        "conversation": "chat-writing",
        "writing": "chat-writing",
        "reasoning": "reasoning",
        "documents": "long-documents",
        "structured": "structured-tools",
    }.get(string(task["category"]), "unclassified")


def scenario_report(rows: list[Record], spec: Comparison, cached: dict[str, Record]) -> Record:
    tasks = {string(task["id"]): task for task in spec.tasks}
    names = sorted({string(row["arm"]) for row in rows})
    normalized = [
        {
            **row,
            "scenario": task_scenario(tasks[string(row["task"])], spec.sampling.temperature > 0),
        }
        for row in rows
    ]
    sizes = {arm.name: arm.drafter.size if arm.drafter else 0 for arm in spec.arms}
    for name, value in cached.items():
        sizes[name] = int(number(record(value["drafter"])["bytes"])) if value["drafter"] else 0
    output: Record = {}
    for scenario in sorted({string(row["scenario"]) for row in normalized}):
        selected = [row for row in normalized if row["scenario"] == scenario]
        comparisons: Record = {}
        for name in names:
            for control in ("target", "baseline", "prior-leader"):
                if name == control or control not in names:
                    continue
                comparison = paired(selected, name, control)
                if name in cached or control in cached:
                    comparison = {
                        key: value for key, value in comparison.items() if "speed" not in key
                    } | {
                        "historical_speed_ratio": comparison["speed_ratio"],
                        "timing_scope": "separate executions; no paired timing interval",
                    }
                comparisons[name + "/" + control] = comparison
        output[scenario] = {
            "summary": relative_speed(
                {
                    name: {
                        **totals([row for row in selected if row["arm"] == name]),
                        "drafter_bytes": sizes[name],
                        **(
                            {
                                "completed_tasks": sum(
                                    row["completed"] is True
                                    for row in selected
                                    if row["arm"] == name
                                )
                            }
                            if all("completed" in row for row in selected)
                            else {}
                        ),
                        "execution": "historical" if name in cached else "measured",
                    }
                    for name in names
                }
            ),
            "comparisons": comparisons,
            "by_outcome": {
                outcome: {
                    name: totals(members)
                    for name in names
                    if (
                        members := [
                            row
                            for row in selected
                            if row["arm"] == name and row.get("completed") is completed
                        ]
                    )
                }
                for outcome, completed in (("completed", True), ("failed", False))
            }
            if any("completed" in row for row in selected)
            else {},
            "timing_scope": "sampled throughput: same requests and seeds, unequal token paths; not a quality or latency equivalence claim"
            if spec.sampling.temperature
            else "matched greedy output paths",
        }
    return output


def faster_verdict(scenarios: Record, candidate: str, confirmed: bool = False) -> Record:
    decisions: Record = {}
    for scenario in PRIMARY:
        if scenario not in scenarios:
            decisions[scenario] = "missing"
            continue
        values = record(scenarios[scenario])
        comparisons = record(values["comparisons"])
        passed = True
        available = True
        for control in ("target", "baseline"):
            comparison = record(comparisons.get(candidate + "/" + control, {}))
            bounds = record(comparison.get("speed_ratio_95_interval", {}))
            if "lower" not in bounds:
                available = False
                continue
            lower = number(bounds["lower"])
            passed &= lower > 1 if control == "target" else lower >= 1
        summary = record(record(values["summary"]).get(candidate, {}))
        passed &= summary.get("correct") is True
        if scenario == "agentic-coding":
            baseline_summary = record(record(values["summary"]).get("baseline", {}))
            if "completed_tasks" not in summary or "completed_tasks" not in baseline_summary:
                available = False
            else:
                passed &= number(summary["completed_tasks"]) >= number(
                    baseline_summary["completed_tasks"]
                )
        decisions[scenario] = "insufficient" if not available else "pass" if passed else "fail"
    eligible = all(value == "pass" for value in decisions.values())
    return {
        "verdict": "faster"
        if eligible and confirmed
        else "confirmation-required"
        if eligible
        else "not-established",
        "scenarios": decisions,
        "deciding_scenario": "agentic-coding",
        "confirmation": "passed" if confirmed else "fresh protected data not yet evaluated",
    }


def level2_decision(slices: Record, main_panel: Record, candidate: str) -> Record:
    reasons: list[str] = [
        name + ": missing Level 2 scenario" for name in PRIMARY if name not in slices
    ]
    for scenario, value in slices.items():
        comparisons = record(record(value)["comparisons"])
        for control in ("baseline", "prior-leader"):
            key = candidate + "/" + control
            comparison = record(comparisons.get(key, {}))
            bounds = record(comparison.get("speed_ratio_95_interval", {}))
            main = record(main_panel.get(key, {}))
            acceptance = record(comparison.get("acceptance_difference_95_interval", {}))
            if "upper" not in bounds and "upper" in acceptance and number(acceptance["upper"]) < 0:
                reasons.append(
                    scenario + ": acceptance below saved " + control + " beyond interval"
                )
            if "upper" in bounds and number(bounds["upper"]) < 1:
                reasons.append(scenario + ": slower than " + control + " beyond interval")
            if (
                "speed_ratio" in comparison
                and "speed_ratio" in main
                and ((number(comparison["speed_ratio"]) < 1) != (number(main["speed_ratio"]) < 1))
            ):
                reasons.append(
                    scenario + ": direction disagrees with main panel against " + control
                )
    return {"level3_required": bool(reasons), "reasons": reasons}

from dataclasses import dataclass
from pathlib import Path

from benchmarks.native import file_reference
from benchmarks.statistics import paired
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, number, record, sequence, string
from w1a1_eagle.records import artifact, canonical, read

PRIMARY = "general"
FIELDS = {"schema", "definition", "results", "candidate", "comparators", "drift"}


@dataclass(frozen=True)
class Rule:
    primary_floor: float
    primary_strict: bool
    guardrail_floor: float
    below_parity_limit: int | None


RULES = {
    "stock": Rule(1.00, True, 0.97, 2),
    "control": Rule(0.97, False, 0.95, None),
}


def scenarios(definition: Record) -> tuple[list[str], list[str]]:
    guardrails: list[str] = []
    reported: list[str] = []
    for value in sequence(definition["scenarios"]):
        scenario = record(value)
        name, role = string(scenario["name"]), string(scenario["role"])
        if name == PRIMARY:
            if role != "primary":
                raise ValueError("the general scenario must be the primary outcome")
        elif role == "guardrail":
            guardrails.append(name)
        else:
            reported.append(name)
    return guardrails, reported


def lower_bound(summary: Record) -> float | None:
    if summary.get("interval_available") is not True:
        return None
    return number(record(summary["speed_ratio_95_interval"])["lower"])


def judge(
    rows: list[Record],
    candidate: str,
    comparator: str,
    rule: Rule,
    guardrails: list[str],
    reported: list[str],
) -> Record:
    by_scenario: dict[str, list[Record]] = {}
    for row in rows:
        if string(row["arm"]) in {candidate, comparator}:
            by_scenario.setdefault(string(row["scenario"]), []).append(row)
    failures: list[str] = []
    invalid = sum(
        1 for values in by_scenario.values() for row in values if row["correct"] is not True
    )
    if invalid:
        failures.append(f"{invalid} outputs are not exact or valid")
    outcomes: dict[str, Record] = {}
    for name in [PRIMARY, *guardrails, *reported]:
        if name not in by_scenario:
            if name in reported:
                continue
            failures.append("missing scenario " + name)
            continue
        summary = paired(by_scenario[name], candidate, comparator)
        outcomes[name] = summary
        if name in reported:
            continue
        bound = lower_bound(summary)
        if bound is None:
            failures.append("no interval for " + name)
            continue
        if name == PRIMARY:
            passed = (
                bound > rule.primary_floor if rule.primary_strict else bound >= rule.primary_floor
            )
            if not passed:
                failures.append(f"primary lower bound {bound:.4f} misses {rule.primary_floor}")
        elif bound < rule.guardrail_floor:
            failures.append(
                f"guardrail {name} lower bound {bound:.4f} below {rule.guardrail_floor}"
            )
    if rule.below_parity_limit is not None:
        below = [
            name
            for name in guardrails
            if name in outcomes and (lower_bound(outcomes[name]) or 0.0) < 1.0
        ]
        if len(below) > rule.below_parity_limit:
            failures.append(f"{len(below)} guardrails below parity: " + ", ".join(below))
    return {
        "comparator": comparator,
        "passed": not failures,
        "failures": failures,
        "scenarios": outcomes,
        "reported_only": [name for name in reported if name in outcomes],
    }


def gate_verdict(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != FIELDS or raw["schema"] != "gate-verdict-v1":
        raise ValueError("unsupported gate verdict specification")
    root = specification.resolve().parent
    definition_ref = artifact(raw["definition"], root)
    definition_ref.verify()
    guardrails, reported = scenarios(read(definition_ref.path))
    rows: list[Record] = []
    sources: list[Record] = []
    for value in sequence(raw["results"]):
        ref = artifact(value, root)
        ref.verify()
        rows += [record(row) for row in sequence(read(ref.path, 1024 * 1024 * 1024)["rows"])]
        sources.append(file_reference(ref.path))
    candidate = string(raw["candidate"])
    comparators = record(raw["comparators"])
    if not set(comparators) <= set(RULES) or not comparators:
        raise ValueError("comparators are stock and/or control")
    verdicts = {
        kind: judge(rows, candidate, string(name), RULES[kind], guardrails, reported)
        for kind, name in comparators.items()
    }
    drift: Record | None = None
    if raw["drift"] is not None:
        drift_raw = record(raw["drift"])
        ref = artifact(drift_raw["result"], root)
        ref.verify()
        drift_rows = [record(row) for row in sequence(read(ref.path, 1024 * 1024 * 1024)["rows"])]
        drift = paired(drift_rows, string(drift_raw["first"]), string(drift_raw["second"]))
        stock = verdicts.get("stock")
        if stock is not None and PRIMARY in record(stock["scenarios"]):
            lead = number(record(record(stock["scenarios"])[PRIMARY])["speed_ratio"]) - 1
            spread = lower_bound(drift)
            stock["claimable_lead"] = spread is not None and lead > 1 - spread
    result: Record = {
        "schema": "gate-verdict-result-v1",
        "definition": file_reference(definition_ref.path),
        "results": sources,
        "candidate": candidate,
        "verdicts": verdicts,
        "drift": drift,
        "passed": all(verdict["passed"] is True for verdict in verdicts.values()),
    }
    Store(output).atomic(output / "verdict.json", canonical(result))
    return result

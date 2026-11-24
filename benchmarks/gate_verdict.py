from dataclasses import dataclass
from pathlib import Path

from benchmarks.native import file_reference
from benchmarks.statistics import paired
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, number, record, sequence, string
from w1a1_eagle.records import artifact, canonical, read

PRIMARY = "general"
FIELDS = {"schema", "definition", "manifest", "results", "candidate", "comparators", "drift"}
SEEDS = {"sampled": frozenset({9201, 9202, 9203})}
GREEDY = frozenset({8101})


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


def roster(manifest: Record, definition: Record, root: Path) -> dict[str, frozenset[str]]:
    if string(record(manifest["definition"])["sha256"]) != string(definition["sha256"]):
        raise ValueError("the workload manifest belongs to a different gate definition")
    expected: dict[str, set[str]] = {}
    for value in record(manifest["workloads"]).values():
        ref = artifact(value, root)
        ref.verify()
        for task in sequence(read(ref.path, 1024 * 1024 * 1024)["tasks"]):
            item = record(task)
            expected.setdefault(string(item["scenario"]), set()).add(string(item["id"]))
    return {name: frozenset(ids) for name, ids in expected.items()}


def coverage(
    rows: list[Record], arms: list[str], expected: dict[str, frozenset[str]]
) -> dict[str, list[str]]:
    seen: dict[tuple[str, str], dict[str, set[int]]] = {}
    for row in rows:
        key = (string(row["scenario"]), string(row["arm"]))
        seen.setdefault(key, {}).setdefault(string(row["task"]), set()).add(
            int(number(row.get("seed", 8101)))
        )
    problems: dict[str, list[str]] = {}
    for scenario, ids in expected.items():
        seeds = SEEDS.get(scenario, GREEDY)
        for arm in arms:
            found = seen.get((scenario, arm), {})
            missing = len(ids - set(found))
            extra = len(set(found) - ids)
            wrong = sum(1 for task in ids & set(found) if frozenset(found[task]) != seeds)
            if missing or extra or wrong:
                problems.setdefault(scenario, []).append(
                    f"{arm}: {missing} missing, {extra} unexpected, {wrong} with wrong seeds"
                )
    return problems


def drift_margin(summary: Record) -> float | None:
    if summary.get("interval_available") is not True:
        return None
    bounds = record(summary["speed_ratio_95_interval"])
    return max(abs(1 - number(bounds["lower"])), abs(number(bounds["upper"]) - 1))


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
        try:
            summary = paired(by_scenario[name], candidate, comparator)
        except ValueError as error:
            if name not in reported:
                failures.append(f"{name}: {error}")
            continue
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
    if set(raw) != FIELDS or raw["schema"] != "gate-verdict-v2":
        raise ValueError("unsupported gate verdict specification")
    root = specification.resolve().parent
    definition_ref = artifact(raw["definition"], root)
    definition_ref.verify()
    definition = file_reference(definition_ref.path)
    guardrails, reported = scenarios(read(definition_ref.path))
    manifest_ref = artifact(raw["manifest"], root)
    manifest_ref.verify()
    expected = roster(read(manifest_ref.path), definition, root)
    if set(expected) != {PRIMARY, *guardrails, *reported}:
        raise ValueError("the workload manifest's scenarios differ from the gate definition")
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
    arms = [candidate, *(string(name) for name in comparators.values())]
    gaps = coverage(rows, arms, expected)
    verdicts = {
        kind: judge(rows, candidate, string(name), RULES[kind], guardrails, reported)
        for kind, name in comparators.items()
    }
    for verdict in verdicts.values():
        for scenario in [PRIMARY, *guardrails]:
            for problem in gaps.get(scenario, []):
                verdict["failures"] = [
                    *sequence(verdict["failures"]),
                    f"incomplete {scenario}: {problem}",
                ]
                verdict["passed"] = False
    transfer = {name: gaps.get(name, []) for name in reported}
    drift: Record | None = None
    if raw["drift"] is not None:
        drift_raw = record(raw["drift"])
        ref = artifact(drift_raw["result"], root)
        ref.verify()
        drift_rows = [
            record(row)
            for row in sequence(read(ref.path, 1024 * 1024 * 1024)["rows"])
            if string(record(row)["scenario"]) == PRIMARY
        ]
        pair = [string(drift_raw["first"]), string(drift_raw["second"])]
        drift_gaps = coverage(drift_rows, pair, {PRIMARY: expected[PRIMARY]})
        drift = paired(drift_rows, pair[0], pair[1]) | {
            "complete": not drift_gaps,
            "gaps": drift_gaps.get(PRIMARY, []),
            "margin": None,
        }
        margin = drift_margin(drift) if not drift_gaps else None
        drift["margin"] = margin
        stock = verdicts.get("stock")
        if stock is not None and PRIMARY in record(stock["scenarios"]):
            lead = number(record(record(stock["scenarios"])[PRIMARY])["speed_ratio"]) - 1
            stock["claimable_lead"] = margin is not None and lead > margin
    missing = [f"{kind} comparator" for kind in RULES if kind not in verdicts]
    if drift is None:
        missing.append("reversed-order drift run")
    elif record(verdicts.get("stock", {})).get("claimable_lead") is not True:
        missing.append("a general lead above the measured session drift")
    result: Record = {
        "schema": "gate-verdict-result-v2",
        "definition": definition,
        "manifest": file_reference(manifest_ref.path),
        "results": sources,
        "candidate": candidate,
        "verdicts": verdicts,
        "drift": drift,
        "transfer_evidence": {
            "complete": not any(transfer.values()),
            "gaps": transfer,
            "role": "reported only; not a pass condition",
        },
        "missing_for_release": missing,
        "passed": not missing and all(verdict["passed"] is True for verdict in verdicts.values()),
    }
    Store(output).atomic(output / "verdict.json", canonical(result))
    return result

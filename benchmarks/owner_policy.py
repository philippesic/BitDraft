from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.coverage import length_band
from benchmarks.metrics import relative_speed
from benchmarks.scenario_report import load_panel
from benchmarks.scenarios import unit_totals
from benchmarks.statistics import paired
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.identity import source_identity
from w1a1_eagle.records import canonical, read

POLICY = ("greedy-short", "greedy-long", "thinking-on", "sampled", "agentic-coding")
SOURCES = {
    "greedy": frozenset({"comparison"}),
    "thinking-on": frozenset({"thinking-slice", "thinking-suite"}),
    "sampled": frozenset({"sampled-slice", "sampled-suite"}),
    "agentic-coding": frozenset({"agentic-slice", "agentic-suite"}),
    "screen": frozenset({"screen", "screen-v2"}),
}
FULL_TIERS = frozenset({"comparison", "thinking-suite", "sampled-suite", "agentic-suite"})
LONG_INPUT = 2048
BANDS = ("<512", "512-2047", "2048-8191", "8192-16384")
EXCEPTIONS = 2
TOLERANCE = 0.97
BASELINE = "baseline"
MERGED_LABEL = (
    "supplementary: same paired source-group bootstrap with every category stratum of fewer "
    "than two source groups merged into one pooled stratum; not the maintained rule and never "
    "used by the maintained verdict"
)


def report(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) - {"count_merged_strata"} != {"schema", "candidate", "panels"}
        or raw["schema"] != "owner-policy-report-v1"
        or output.exists()
    ):
        raise ValueError("owner policy report requires its schema and a new output")
    if type(raw.get("count_merged_strata", False)) is not bool:
        raise ValueError("count_merged_strata must be an explicit boolean")
    count_merged = raw.get("count_merged_strata", False) is True
    candidate = string(raw["candidate"])
    if candidate == BASELINE:
        raise ValueError("the candidate must differ from Baseline")
    root = specification.resolve().parent
    declared = record(raw["panels"])
    if set(declared) - set(SOURCES) or not declared:
        raise ValueError("panels are greedy, thinking-on, sampled, agentic-coding and screen")
    sources: dict[str, Record] = {}
    identities: set[tuple[str, ...]] = set()
    models: set[tuple[str, str]] = set()
    for name, evidence in declared.items():
        pooled = pool(name, [value for value in sequence(evidence)], root, candidate)
        identities.add(tuple(string(value) for value in sequence(pooled["runtime"])))
        models.add((string(pooled["candidate_sha256"]), string(pooled["baseline_sha256"])))
        sources[name] = pooled
    if len(identities) != 1 or len(models) != 1:
        raise ValueError(
            "policy panels must share target, runtime, context, role, candidate and Baseline"
        )
    scenarios: Record = {}
    if "greedy" in sources:
        greedy = sources["greedy"]
        split = greedy_split(tuple(record(task) for task in sequence(greedy["tasks"])))
        for scenario in ("greedy-short", "greedy-long"):
            scenarios[scenario] = evaluate(
                [row for row in rows_of(greedy) if split[string(row["task"])] == scenario],
                greedy,
                candidate,
            )
    for scenario in ("thinking-on", "sampled", "agentic-coding"):
        if scenario in sources:
            scenarios[scenario] = evaluate(rows_of(sources[scenario]), sources[scenario], candidate)
    bands: Record = {}
    if "greedy" in sources:
        greedy = sources["greedy"]
        tokens = {
            string(task["id"]): input_tokens(task)
            for task in (record(value) for value in sequence(greedy["tasks"]))
        }
        for index, label in enumerate(BANDS):
            members = [
                row for row in rows_of(greedy) if length_band(tokens[string(row["task"])]) == index
            ]
            if members:
                bands[label] = evaluate(members, greedy, candidate)
    supplementary: Record = (
        {"screen": evaluate(rows_of(sources["screen"]), sources["screen"], candidate)}
        if "screen" in sources
        else {}
    )
    model, baseline_model = next(iter(models))
    final: Record = {
        "schema": "owner-policy-report-result-v1",
        "code_sha256": source_identity(Path(__file__).resolve().parents[1]),
        "policy": "owner policy, working rule 2026-11-16",
        "rule": "passed: at most two scenarios with a 95% speed-ratio lower bound below 1.00, "
        "each at 0.97 or better, every other lower bound at 1.00 or better, exact greedy "
        "outputs, no invalid outputs and agentic completion not below Baseline",
        "candidate": candidate,
        "candidate_sha256": model,
        "baseline_sha256": baseline_model,
        "panels": {
            name: {key: value for key, value in pooled.items() if key not in {"rows", "tasks"}}
            for name, pooled in sources.items()
        },
        "scenarios": scenarios,
        "level3": policy_verdict(level_entries(scenarios, 3)),
        "development": {
            **policy_verdict(level_entries(scenarios, 2)),
            "label": "development-only Level 2 slice verdict for short-run finalists; never a release decision",
        },
        "count_merged_strata": count_merged,
        "merged_strata_verdicts": {
            "label": MERGED_LABEL + "; recomputed only because count_merged_strata was set",
            "level3": policy_verdict(level_entries(scenarios, 3, merged=True)),
            "development": policy_verdict(level_entries(scenarios, 2, merged=True)),
        }
        if count_merged
        else None,
        "greedy_length_bands": bands,
        "supplementary": supplementary,
        "supplementary_scope": "the 60-request screen and the length bands are reported only; they never decide the policy",
        "existing_verdict": "the stricter seven-scenario scenario-report verdict is unchanged and reported separately",
        "scope": "source-group intervals cover request/task variation; pooled matched repeats stay within their source groups; confirmation is separate",
    }
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "result.json", canonical(final))
    return final


def pool(name: str, evidence: list[object], root: Path, candidate: str) -> Record:
    if not evidence:
        raise ValueError("each declared policy panel needs at least one run")
    loaded = [load_panel(value, root) for value in evidence]
    first = loaded[0][0]
    rows: list[Record] = []
    runs: list[Record] = []
    historical: set[str] = set()
    for index, (spec, result, cached) in enumerate(loaded):
        if spec.tier not in SOURCES[name]:
            raise ValueError(name + " panel has tier " + spec.tier)
        if (
            spec.reference_protocol() != first.reference_protocol()
            or arm_models(spec, cached) != arm_models(first, loaded[0][2])
            or spec.repeats != first.repeats
        ):
            raise ValueError("pooled runs must repeat the same panel, workload, arms and depths")
        measured = {arm.name: arm for arm in spec.arms}
        if candidate not in measured or measured[candidate].drafter is None:
            raise ValueError("candidate must be freshly measured in each policy panel")
        if BASELINE not in measured and BASELINE not in cached:
            raise ValueError("policy panels require a Baseline arm")
        historical.update(cached)
        panel_rows = [record(row) for row in sequence(result["rows"])] + [
            record(row) for value in cached.values() for row in sequence(value["rows"])
        ]
        rows.extend(
            {**row, "rep": index * spec.repeats + integer(row["rep"])} for row in panel_rows
        )
        runs.append(
            {
                "evidence": evidence[index],
                "elapsed_seconds": result.get("elapsed_seconds"),
                "arm_elapsed_seconds": result.get("arm_elapsed_seconds"),
                "projection": result.get("projection", False),
            }
        )
    models = arm_models(first, loaded[0][2])
    return {
        "tier": first.tier,
        "coverage": "full" if first.tier in FULL_TIERS else "slice",
        "runs": runs,
        "pooled_runs": len(loaded),
        "arm_depths": first.arm_depths(),
        "historical_arms": sorted(historical),
        "seeds": list(first.sampling.seeds),
        "runtime": list(first.runtime_identity()),
        "candidate_sha256": models[candidate],
        "baseline_sha256": models[BASELINE],
        "tasks": list(first.tasks),
        "rows": rows,
    }


def arm_models(spec: Comparison, cached: dict[str, Record]) -> dict[str, str]:
    return {
        **{arm.name: arm.drafter.sha256 if arm.drafter else "target" for arm in spec.arms},
        **{
            name: string(record(value["drafter"])["sha256"]) if value["drafter"] else "target"
            for name, value in cached.items()
        },
    }


def rows_of(pooled: Record) -> list[Record]:
    return [record(row) for row in sequence(pooled["rows"])]


def input_tokens(task: Record) -> int:
    return integer(record(task["ancestry"])["input_tokens"])


def greedy_split(tasks: tuple[Record, ...]) -> dict[str, str]:
    return {
        string(task["id"]): "greedy-short" if input_tokens(task) < LONG_INPUT else "greedy-long"
        for task in tasks
    }


def evaluate(rows: list[Record], pooled: Record, candidate: str) -> Record:
    agentic = string(pooled["tier"]).startswith("agentic-")
    depths = record(pooled["arm_depths"])
    historical = set(sequence(pooled["historical_arms"]))
    names = sorted({string(row["arm"]) for row in rows})
    summary = relative_speed(
        {
            name: {
                **unit_totals([row for row in rows if row["arm"] == name]),
                "depth": depths.get(name),
                "execution": "historical" if name in historical else "measured",
                **(
                    {
                        "completed_tasks": sum(
                            row["completed"] is True for row in rows if row["arm"] == name
                        )
                    }
                    if agentic and all("completed" in row for row in rows)
                    else {}
                ),
            }
            for name in names
        },
        "assigned_tasks_per_second" if agentic else "tokens_per_second",
    )
    units = [{**row, "tokens": 1} for row in rows] if agentic else rows
    paired_historical = bool(historical & {candidate, BASELINE})
    comparison, lower, reason = speed(paired(units, candidate, BASELINE), paired_historical)
    merged, merged_lower, merged_reason = speed(
        paired(units, candidate, BASELINE, merge_sparse=True), paired_historical
    )
    candidate_summary = record(summary[candidate])
    baseline_summary = record(summary[BASELINE])
    completion = (
        None
        if not agentic
        else "unavailable"
        if "completed_tasks" not in candidate_summary
        else "regression"
        if number(candidate_summary["completed_tasks"])
        < number(baseline_summary["completed_tasks"])
        else "held"
    )
    return {
        "tier": pooled["tier"],
        "coverage": pooled["coverage"],
        "pooled_runs": pooled["pooled_runs"],
        "units": len({(row["task"], row["rep"], row.get("seed", 8101)) for row in rows}),
        "speed_measure": "assigned tasks per second, including failed tasks and tools"
        if agentic
        else "emitted tokens per summed request second",
        "summary": summary,
        "comparison": comparison,
        "speed_ratio_95_lower": lower,
        "interval_reason": reason,
        "supplementary_merged_strata": {
            "label": MERGED_LABEL,
            "comparison": merged,
            "merged_strata": merged.get("merged_strata", []),
            "speed_ratio_95_lower": merged_lower,
            "interval_reason": merged_reason,
        },
        "correct": candidate_summary.get("correct") is True,
        "completion": completion,
    }


def speed(comparison: Record, historical: bool) -> tuple[Record, float | None, str | None]:
    if historical:
        stripped = {key: value for key, value in comparison.items() if "speed" not in key}
        return (
            stripped | {"historical_speed_ratio": comparison["speed_ratio"]},
            None,
            "Baseline is a saved reference; separate executions have no paired interval",
        )
    bounds = record(comparison.get("speed_ratio_95_interval", {}))
    if "lower" not in bounds:
        return comparison, None, string(comparison.get("reason", "paired interval unavailable"))
    return comparison, number(bounds["lower"]), None


def level_entries(scenarios: Record, level: int, merged: bool = False) -> Record:
    entries: Record = {}
    for name, value in scenarios.items():
        entry = record(value)
        if merged:
            entry = entry | record(entry["supplementary_merged_strata"])
        insufficient = level == 3 and entry["coverage"] != "full"
        entries[name] = {
            "speed_ratio_95_lower": None if insufficient else entry["speed_ratio_95_lower"],
            "interval_reason": "slice-level panel cannot decide Level 3"
            if insufficient
            else entry["interval_reason"],
            "correct": entry["correct"],
            "completion": entry["completion"],
        }
    return entries


def policy_verdict(entries: Record) -> Record:
    decisions: Record = {}
    for name in POLICY:
        if name not in entries:
            decisions[name] = "missing"
            continue
        entry = record(entries[name])
        lower = entry["speed_ratio_95_lower"]
        if entry["correct"] is not True or entry["completion"] == "regression":
            decisions[name] = "correctness-or-completion-loss"
        elif lower is None or entry["completion"] == "unavailable":
            decisions[name] = "insufficient"
        elif number(lower) >= 1:
            decisions[name] = "at-least-baseline"
        elif number(lower) >= TOLERANCE:
            decisions[name] = "tolerated-shortfall"
        else:
            decisions[name] = "below-tolerance"
    values = list(decisions.values())
    shortfalls = [name for name, value in decisions.items() if value == "tolerated-shortfall"]
    verdict = (
        "failed"
        if "correctness-or-completion-loss" in values
        else "not-demonstrated"
        if "missing" in values or "insufficient" in values
        else "passed"
        if "below-tolerance" not in values and len(shortfalls) <= EXCEPTIONS
        else "failed"
    )
    return {
        "verdict": verdict,
        "scenarios": decisions,
        "shortfalls": shortfalls,
        "allowed_shortfalls": EXCEPTIONS,
        "tolerance_lower_bound": TOLERANCE,
    }

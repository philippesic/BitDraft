from pathlib import Path

import pytest

from benchmarks.native import file_reference
from benchmarks.owner_policy import POLICY, greedy_split, policy_verdict, report
from w1a1_eagle.contracts import Record, number, record, sequence
from w1a1_eagle.records import canonical, read

ARMS = ("target", "baseline", "candidate")
REVERSED = ("target", "candidate", "baseline")
STYLES = ("minimal", "coding", "tools")


def entry(lower: float | None, correct: bool = True, completion: str | None = None) -> Record:
    return {
        "speed_ratio_95_lower": lower,
        "interval_reason": None if lower is not None else "sparse stratum",
        "correct": correct,
        "completion": completion,
    }


def entries(**lowers: float | None) -> Record:
    values: Record = {}
    for name in POLICY:
        lower = lowers.get(name.replace("-", "_"), 1.02)
        values[name] = entry(lower, completion="held" if name == "agentic-coding" else None)
    return values


def test_policy_allows_two_tolerated_shortfalls_only() -> None:
    passed = policy_verdict(entries(greedy_long=0.98, thinking_on=0.98))
    assert passed["verdict"] == "passed" and passed["shortfalls"] == ["greedy-long", "thinking-on"]
    assert policy_verdict(entries())["verdict"] == "passed"
    assert policy_verdict(entries(sampled=0.97))["verdict"] == "passed"
    below = policy_verdict(entries(sampled=0.96))
    assert below["verdict"] == "failed"
    assert record(below["scenarios"])["sampled"] == "below-tolerance"
    three = policy_verdict(entries(greedy_short=0.99, greedy_long=0.98, sampled=0.98))
    assert three["verdict"] == "failed"


def test_policy_not_demonstrated_without_every_interval() -> None:
    missing = entries()
    missing.pop("thinking-on")
    result = policy_verdict(missing)
    assert result["verdict"] == "not-demonstrated"
    assert record(result["scenarios"])["thinking-on"] == "missing"
    unavailable = policy_verdict(entries(greedy_long=None))
    assert unavailable["verdict"] == "not-demonstrated"
    assert record(unavailable["scenarios"])["greedy-long"] == "insufficient"
    assert policy_verdict(entries(greedy_long=None, sampled=0.5))["verdict"] == "not-demonstrated"


def test_policy_fails_on_completion_or_correctness_loss() -> None:
    values = entries()
    values["agentic-coding"] = entry(1.2, completion="regression")
    assert policy_verdict(values)["verdict"] == "failed"
    values.pop("thinking-on")
    assert policy_verdict(values)["verdict"] == "failed"
    values = entries()
    values["greedy-short"] = entry(1.2, correct=False)
    assert policy_verdict(values)["verdict"] == "failed"
    values = entries()
    values["agentic-coding"] = entry(1.2, completion="unavailable")
    assert policy_verdict(values)["verdict"] == "not-demonstrated"


def test_greedy_split_counts_2048_input_tokens_as_long() -> None:
    tasks: tuple[Record, ...] = tuple(
        {"id": str(tokens), "ancestry": {"input_tokens": tokens}} for tokens in (1, 2047, 2048)
    )
    assert greedy_split(tasks) == {
        "1": "greedy-short",
        "2047": "greedy-short",
        "2048": "greedy-long",
    }


def runtime(root: Path) -> Record:
    for name in ("llama-server", "libnative.so", "target", "baseline.gguf", "candidate.bin"):
        (root / name).write_bytes(name.encode())
    (root / "build.json").write_bytes(
        canonical(
            {"artifacts": [file_reference(root / n) for n in ("llama-server", "libnative.so")]}
        )
    )
    qualification = root / "sampler.json"
    qualification.write_bytes(
        canonical(
            {
                "status": "passed",
                "eos_stopped": True,
                "target_sha256": record(file_reference(root / "target"))["sha256"],
                "runtime_source_sha256": record(file_reference(root / "build.json"))["sha256"],
                "functions_tested": ["sample-and-match", "rejection"],
            }
        )
    )
    return {
        "source": file_reference(root / "build.json"),
        "binary": file_reference(root / "llama-server"),
        "libraries": [file_reference(root / "libnative.so")],
        "target": file_reference(root / "target"),
        "baseline": file_reference(root / "baseline.gguf"),
        "candidate": file_reference(root / "candidate.bin"),
        "qualification": file_reference(qualification),
    }


def task(identity: str, category: str, **fields: object) -> Record:
    return {
        "id": identity,
        "group": identity,
        "category": category,
        "role": "dev",
        "messages": [{"role": "user", "content": "task " + identity}],
        "output_cap": 2048,
        **fields,
    }


BALANCED_LONG = {"code": 4, "reasoning": 4, "documents": 8, "conversation": 4, "structured": 2}
BALANCED_LONG["writing"] = 2
BROADER120_LONG = {"code": 3, "documents": 19, "conversation": 1, "structured": 1}


def greedy_tasks(long: dict[str, int] = BALANCED_LONG) -> list[Record]:
    totals = {"code": 30, "reasoning": 20, "documents": 25, "conversation": 20, "structured": 15}
    totals["writing"] = 10
    short = {category: count - long.get(category, 0) for category, count in totals.items()}
    tasks: list[Record] = []
    for category, count in short.items():
        for index in range(count):
            tokens = 100 if len(tasks) < 60 else 2047 if len(tasks) == 95 else 1000
            tasks.append(task("g" + str(len(tasks)), category, ancestry={"input_tokens": tokens}))
    huge = {"documents": 4, "code": 2}
    for category, count in long.items():
        for index in range(count):
            tokens = 9000 if index < huge.get(category, 0) else 2048
            tasks.append(task("g" + str(len(tasks)), category, ancestry={"input_tokens": tokens}))
    for item in tasks:
        item["output_cap"] = 256
    return tasks


def requests(prefix: str, thinking: bool | None) -> list[Record]:
    return [
        task(
            prefix + str(index),
            "code" if index < 6 else "reasoning",
            **({} if thinking is None else {"thinking": thinking}),
        )
        for index in range(12)
    ]


def agentic_tasks() -> list[Record]:
    return [
        task(
            "a" + str(index),
            "code",
            group="project:owner/p" + str(index // 2),
            project="owner/p" + str(index // 2),
            scenario="agentic-coding",
            style=STYLES[(index // 2 + index % 2) % 3],
            task_kind="bug-fix",
            fixture={},
            output_cap=1024,
        )
        for index in range(12)
    ]


def panel(
    root: Path,
    shared: Record,
    tier: str,
    tasks: list[Record],
    order: tuple[str, ...] = ARMS,
    seeds: tuple[int, ...] = (8101,),
    candidate_seconds: float = 0.9,
    completed: int = 12,
) -> Record:
    root.mkdir()
    (root / "workload.json").write_bytes(canonical({"schema": 1, "tasks": tasks}))
    spec: Record = {
        "schema": "native-comparison-v1",
        "source": shared["source"],
        "binary": shared["binary"],
        "libraries": shared["libraries"],
        "target": shared["target"],
        "arms": [
            {"name": name, "drafter": None if name == "target" else shared[name]} for name in order
        ],
        "workload": file_reference(root / "workload.json"),
        "repeats": 1,
        "warmups": 0,
        "context": 32768,
        "depth": 3,
        "depths": {"baseline": 4, "candidate": 3},
        "tier": tier,
        "role": "dev",
    }
    if len(seeds) > 1:
        spec["sampling"] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0.0,
            "seeds": list(seeds),
            "qualification": shared["qualification"],
        }
    (root / "specification.json").write_bytes(canonical(spec))
    rows: list[Record] = []
    for arm in order:
        for index, item in enumerate(tasks):
            for seed in seeds:
                seconds = {"target": 1.4, "baseline": 1.0}.get(arm, candidate_seconds)
                row: Record = {
                    "arm": arm,
                    "task": item["id"],
                    "group": item["group"],
                    "category": item["category"],
                    "rep": 0,
                    "seed": seed,
                    "tokens": 100,
                    "wall_seconds": seconds * (1 + 0.02 * (index % 5)),
                    "correct": True,
                    "native": {
                        "proposed": 0 if arm == "target" else 60,
                        "accepted": 0 if arm == "target" else 30,
                    },
                }
                if tier.startswith("agentic-"):
                    row["completed"] = index < (completed if arm == "candidate" else 12)
                rows.append(row)
    (root / "result.json").write_bytes(canonical({"status": "complete", "rows": rows}))
    return file_reference(root / "result.json")


def policy_spec(root: Path, panels: Record) -> Path:
    path = root / "policy.json"
    path.write_bytes(
        canonical({"schema": "owner-policy-report-v1", "candidate": "candidate", "panels": panels})
    )
    return path


def test_report_pools_repeats_and_decides_all_five_scenarios(tmp_path: Path) -> None:
    shared = runtime(tmp_path)
    thinking = requests("t", True)
    panels: Record = {
        "greedy": [panel(tmp_path / "greedy", shared, "comparison", greedy_tasks())],
        "thinking-on": [
            panel(tmp_path / "thinking-1", shared, "thinking-slice", thinking),
            panel(tmp_path / "thinking-2", shared, "thinking-slice", thinking, REVERSED),
        ],
        "sampled": [
            panel(
                tmp_path / "sampled",
                shared,
                "sampled-slice",
                requests("s", False),
                seeds=(8101, 8102),
            )
        ],
        "agentic-coding": [panel(tmp_path / "agentic", shared, "agentic-slice", agentic_tasks())],
    }
    result = report(policy_spec(tmp_path, panels), tmp_path / "output")
    scenarios = record(result["scenarios"])
    short, long = record(scenarios["greedy-short"]), record(scenarios["greedy-long"])
    assert short["units"] == 96 and long["units"] == 24
    pooled = record(scenarios["thinking-on"])
    assert pooled["pooled_runs"] == 2 and pooled["units"] == 24
    assert record(record(pooled["summary"])["baseline"])["depth"] == 4
    assert record(record(pooled["summary"])["candidate"])["depth"] == 3
    assert record(scenarios["sampled"])["units"] == 24
    agentic = record(scenarios["agentic-coding"])
    assert agentic["completion"] == "held"
    assert number(record(agentic["comparison"])["speed_ratio"]) > 1
    development = record(result["development"])
    assert development["verdict"] == "passed"
    assert "development-only" in str(development["label"])
    level3 = record(result["level3"])
    assert level3["verdict"] == "not-demonstrated"
    assert record(level3["scenarios"])["greedy-short"] == "at-least-baseline"
    assert record(level3["scenarios"])["thinking-on"] == "insufficient"
    assert set(record(result["greedy_length_bands"])) == {
        "<512",
        "512-2047",
        "2048-8191",
        "8192-16384",
    }
    assert record(record(result["panels"])["greedy"])["arm_depths"] == {
        "target": None,
        "baseline": 4,
        "candidate": 3,
    }
    stored = read(tmp_path / "output" / "result.json")
    assert stored["schema"] == "owner-policy-report-result-v1"


def test_report_fails_completion_regression_and_refuses_mismatched_repeats(
    tmp_path: Path,
) -> None:
    shared = runtime(tmp_path)
    panels: Record = {
        "agentic-coding": [
            panel(tmp_path / "agentic", shared, "agentic-slice", agentic_tasks(), completed=11)
        ]
    }
    result = report(policy_spec(tmp_path, panels), tmp_path / "output")
    assert record(result["development"])["verdict"] == "failed"
    assert record(record(result["scenarios"])["agentic-coding"])["completion"] == "regression"
    other = [
        item | {"messages": [{"role": "user", "content": "x"}]} for item in requests("t", True)
    ]
    panels = {
        "thinking-on": [
            panel(tmp_path / "first", shared, "thinking-slice", requests("t", True)),
            panel(tmp_path / "second", shared, "thinking-slice", other),
        ]
    }
    with pytest.raises(ValueError, match="same panel"):
        report(policy_spec(tmp_path, panels), tmp_path / "refused")
    panels = {"sampled": [panel(tmp_path / "wrong", shared, "thinking-slice", requests("w", True))]}
    with pytest.raises(ValueError, match="tier"):
        report(policy_spec(tmp_path, panels), tmp_path / "wrong-tier")
    assert not (tmp_path / "refused").exists()


def test_merged_strata_figure_is_supplementary_and_opt_in(tmp_path: Path) -> None:
    from benchmarks.statistics import MERGED, paired

    rows: list[Record] = [
        {
            "arm": arm,
            "group": str(group),
            "task": str(group),
            "rep": 0,
            "tokens": 100,
            "wall_seconds": (0.9 if arm == "candidate" else 1.0) * (1 + 0.02 * group),
            "category": "documents" if group < 4 else ("conversation", "structured")[group - 4],
            "native": {"proposed": 2, "accepted": 1},
        }
        for arm in ("candidate", "baseline")
        for group in range(6)
    ]
    maintained = paired(rows, "candidate", "baseline", draws=100)
    assert maintained["interval_available"] is False and "merged_strata" not in maintained
    merged = paired(rows, "candidate", "baseline", draws=100, merge_sparse=True)
    assert merged["interval_available"] is True
    assert merged["merged_strata"] == ["[('conversation', 1)]", "[('structured', 1)]"]
    assert record(merged["strata"])[MERGED] == 2
    shared = runtime(tmp_path)
    evidence = panel(tmp_path / "greedy", shared, "comparison", greedy_tasks(BROADER120_LONG))
    path = policy_spec(tmp_path, {"greedy": [evidence]})
    result = report(path, tmp_path / "default")
    long = record(record(result["scenarios"])["greedy-long"])
    assert long["speed_ratio_95_lower"] is None and "insufficient" in str(long["interval_reason"])
    figure = record(long["supplementary_merged_strata"])
    assert (
        figure["speed_ratio_95_lower"] is not None and len(sequence(figure["merged_strata"])) == 2
    )
    assert "never used by the maintained verdict" in str(figure["label"])
    assert record(record(result["level3"])["scenarios"])["greedy-long"] == "insufficient"
    assert result["count_merged_strata"] is False and result["merged_strata_verdicts"] is None
    raw = read(path) | {"count_merged_strata": True}
    path.write_bytes(canonical(raw))
    opted = report(path, tmp_path / "opted")
    assert opted["level3"] == result["level3"]
    variant = record(opted["merged_strata_verdicts"])
    assert record(record(variant["level3"])["scenarios"])["greedy-long"] == "at-least-baseline"
    assert record(variant["level3"])["verdict"] == "not-demonstrated"
    path.write_bytes(canonical(read(path) | {"count_merged_strata": "yes"}))
    with pytest.raises(ValueError, match="explicit boolean"):
        report(path, tmp_path / "refused")

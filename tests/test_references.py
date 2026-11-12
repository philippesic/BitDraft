from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.comparison import Comparison
from benchmarks.metrics import native_counts, signature, totals
from benchmarks.native import file_reference
from benchmarks.references import register
from benchmarks.runner import historical_comparisons
from benchmarks.runtime_policy import arguments, payload
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.records import artifact, canonical, read


def evidence(root: Path) -> Path:
    for name in ("llama-server", "libnative.so", "target", "drafter"):
        (root / name).write_bytes(name.encode())
    (root / "build.json").write_bytes(
        canonical(
            {
                "artifacts": [
                    file_reference(root / name) for name in ("llama-server", "libnative.so")
                ]
            }
        )
    )
    tasks: list[Record] = [
        {
            "id": str(i),
            "group": str(i),
            "category": "code",
            "role": "dev",
            "messages": [{"role": "user", "content": "task"}],
            "output_cap": 2,
        }
        for i in range(2)
    ]
    (root / "workload.json").write_bytes(canonical({"schema": 1, "tasks": tasks}))
    spec: Record = {
        "schema": "native-comparison-v1",
        "source": file_reference(root / "build.json"),
        "binary": file_reference(root / "llama-server"),
        "libraries": [file_reference(root / "libnative.so")],
        "target": file_reference(root / "target"),
        "arms": [
            {"name": "target", "drafter": None},
            {"name": "baseline", "drafter": file_reference(root / "drafter")},
        ],
        "workload": file_reference(root / "workload.json"),
        "repeats": 1,
        "warmups": 0,
        "context": 16,
        "depth": 2,
        "tier": "pilot",
        "role": "dev",
    }
    (root / "specification.json").write_bytes(canonical(spec))
    rows: list[Record] = []
    settings: Record = {
        "seed": 8101,
        "top_k": 0,
        "n_predict": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "ignore_eos": False,
        "grammar": "",
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "dynatemp_range": 0.0,
    }
    for arm in ("target", "baseline"):
        cell = root / ("0-0-" + arm)
        cell.mkdir()
        (cell / "command.json").write_bytes(
            canonical(
                {
                    "argv": arguments(
                        root / "llama-server",
                        root / "target",
                        root / "drafter" if arm != "target" else None,
                        16,
                        2,
                        12345,
                    )
                }
            )
        )
        for i in range(2):
            response: Record = {
                "tokens": [7],
                "content": "answer",
                "stop_type": "eos",
                "stopping_word": "",
                "truncated": False,
                "timings": {"draft_n": 2, "draft_n_accepted": 1},
                "generation_settings": settings
                | {"speculative.types": "none" if arm == "target" else "draft-dflash"},
            }
            (cell / f"{i}-response.json").write_bytes(canonical(response))
            (cell / f"{i}-request.json").write_bytes(
                canonical(payload([9], 2, 0.0, 1.0, 0, 0.0, 8101))
            )
            rows.append(
                {
                    "task": str(i),
                    "group": str(i),
                    "category": "code",
                    "arm": arm,
                    "rep": 0,
                    "correct": True,
                    "wall_seconds": 1.0,
                    "tokens": 1,
                    "native": native_counts(response, arm != "target"),
                }
            )
        (cell / "server.log").write_text(
            "\n".join(
                f"#calls(b,g,a) = {i} 0 0, #gen drafts = 0, #acc drafts = 0, #gen tokens = {i * 2}, #acc tokens = {i}"
                for i in (1, 2)
            )
        )
    (root / "gpu.txt").write_text("test GPU")
    (root / "result.json").write_bytes(
        canonical(
            {
                "status": "complete",
                "rows": rows,
                "summary": {
                    arm: totals([row for row in rows if row["arm"] == arm])
                    for arm in ("target", "baseline")
                },
            }
        )
    )
    return root / "specification.json"


def test_references_deduplicate_validate_raw_and_preserve_historical_scope(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    refs: list[tuple[str, object]] = []
    for name in ("target", "baseline"):
        registration = tmp_path / "registration.json"
        registration.write_bytes(
            canonical(
                {
                    "schema": 1,
                    "comparison": file_reference(path),
                    "result": file_reference(tmp_path / "result.json"),
                    "arm": name,
                    "rep": 0,
                }
            )
        )
        first = register(registration, tmp_path / "registry")
        second = register(registration, tmp_path / "registry")
        assert (
            first["reused"] is False
            and second["reused"] is True
            and first["reference"] == second["reference"]
        )
        refs.append((name, first["reference"]))
    spec = replace(
        Comparison.load(path),
        references=tuple((name, artifact(ref, tmp_path)) for name, ref in refs),
    )
    cached = spec.cached()
    from benchmarks.comparison import Arm
    from benchmarks.suite import evaluated_drafter

    model = artifact(file_reference(tmp_path / "drafter"), tmp_path)
    with_previous = replace(
        spec,
        arms=(Arm("candidate", model),),
        references=(*spec.references, ("prior-leader", spec.references[1][1])),
    )
    assert evaluated_drafter(with_previous, "prior-leader") == model
    assert evaluated_drafter(with_previous, "candidate") == model
    for forbidden in ("target", "baseline", "missing"):
        with pytest.raises(ValueError, match="experimental"):
            evaluated_drafter(with_previous, forbidden)
    assert record(cached["target"]["signatures"])["0"] == signature(
        read(tmp_path / "0-0-target/0-response.json")
    )
    rows = [{**record(row), "arm": "candidate"} for row in sequence(cached["baseline"]["rows"])]
    comparison = record(historical_comparisons(rows, cached)["candidate/baseline"])
    assert comparison["acceptance_difference"] == 0
    assert "speed_ratio_95_interval" not in comparison and "historical_speed_ratio" in comparison
    subset = record(historical_comparisons(rows[:1], cached)["candidate/baseline"])
    assert subset["groups"] == 1 and subset["acceptance_difference"] == 0
    for broken in (
        replace(spec, depth=3),
        replace(spec, tasks=({**spec.tasks[0], "output_cap": 3}, *spec.tasks[1:])),
        replace(spec, references=spec.references[:1]),
    ):
        with pytest.raises(ValueError):
            broken.cached()
    (tmp_path / "0-0-baseline/0-response.json").unlink()
    with pytest.raises((ValueError, OSError)):
        spec.cached()


def test_partial_duplicate_or_counter_mismatch_cannot_be_registered(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    result_path = tmp_path / "result.json"
    original = read(result_path)
    registration = tmp_path / "registration.json"
    for rows in (
        sequence(original["rows"])[:-1],
        sequence(original["rows"]) + sequence(original["rows"])[:1],
        [
            {**record(row), "native": {"accepted": 0, "proposed": 2, "acceptance": 0.0}}
            if record(row)["arm"] == "baseline"
            else row
            for row in sequence(original["rows"])
        ],
    ):
        result_path.write_bytes(canonical({**original, "rows": rows}))
        registration.write_bytes(
            canonical(
                {
                    "schema": 1,
                    "comparison": file_reference(path),
                    "result": file_reference(result_path),
                    "arm": "baseline",
                    "rep": 0,
                }
            )
        )
        with pytest.raises(ValueError):
            register(registration, tmp_path / "registry")


def test_screen_enforces_categories_lengths_caps_and_independent_groups() -> None:
    from benchmarks.comparison import require_cohort

    categories = [
        name
        for name, count in (
            ("code", 15),
            ("reasoning", 10),
            ("documents", 12),
            ("conversation", 10),
            ("structured", 8),
            ("writing", 5),
        )
        for _ in range(count)
    ]
    lengths = [100] * 30 + [1000] * 18 + [4000] * 9 + [10000] * 3
    tasks: tuple[Record, ...] = tuple(
        {
            "id": str(i),
            "group": str(i),
            "category": category,
            "output_cap": 128,
            "ancestry": {"input_tokens": length},
        }
        for i, (category, length) in enumerate(zip(categories, lengths, strict=True))
    )
    require_cohort("screen", tasks)
    for first in (
        {**tasks[0], "category": "documents"},
        {**tasks[0], "output_cap": 1024},
        {**tasks[0], "group": "1"},
        {**tasks[0], "ancestry": {"input_tokens": 10000}},
    ):
        with pytest.raises(ValueError):
            require_cohort("screen", (first, *tasks[1:]))

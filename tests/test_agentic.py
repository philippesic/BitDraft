from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.comparison import Arm, Comparison, require_agentic
from benchmarks.native import file_reference
from benchmarks.panels import panels
from benchmarks.project_workflows import prepare
from benchmarks.sandbox import SYSTEMS
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import canonical, read


def test_agentic_groups_preserve_projects_and_style_coverage() -> None:
    tasks: tuple[Record, ...] = tuple(
        {
            "project": "owner/project" + str(project),
            "group": "project:owner/project" + str(project),
            "scenario": "agentic-coding",
            "style": ("minimal", "coding", "tools")[(project + task) % 3],
            "fixture": {},
        }
        for project in range(6)
        for task in range(2)
    )
    require_agentic("agentic-slice", tasks)
    for broken in (
        tasks[1:],
        ({**tasks[0], "group": "one-independent-task"}, *tasks[1:]),
        ({**tasks[0], "scenario": "other"}, *tasks[1:]),
        tuple({**task, "style": "tools"} for task in tasks),
    ):
        with pytest.raises(ValueError, match="agentic"):
            require_agentic("agentic-slice", broken)


def test_confirmation_preparation_never_executes_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.agentic as agentic

    def forbidden(*args: object) -> None:
        raise AssertionError("confirmation controls must not execute")

    monkeypatch.setattr(agentic, "Sandbox", forbidden)
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    recipes = tmp_path / "recipes.json"
    recipes.write_bytes(
        canonical(
            {
                "tasks": [
                    {
                        "id": "reserve-" + str(index),
                        "project": "owner/reserved",
                        "style": ("minimal", "coding", "tools")[index % 3],
                        "task_kind": "bug-fix",
                        "mutations": {"source.py": [{"old": "value = 1", "new": "value = 0"}]},
                        "checks": "raise AssertionError('never executed')",
                        "prompt": "Restore the value.",
                    }
                    for index in range(15)
                ]
            }
        )
    )
    provenance = tmp_path / "audit.json"
    provenance.write_bytes(canonical({"scope": "test source"}))
    specification = tmp_path / "spec.json"
    specification.write_bytes(
        canonical(
            {
                "schema": 2,
                "role": "confirmation",
                "projects": {"owner/reserved": {"source.py": file_reference(source)}},
                "recipes": file_reference(recipes),
                "provenance": file_reference(provenance),
            }
        )
    )
    result = prepare(specification, tmp_path / "prepared")
    assert result["status"] == "frozen-unexecuted"
    assert result["controls_executed"] is False and result["checks"] == []
    panel_spec = tmp_path / "panels.json"
    panel_spec.write_bytes(canonical({"schema": 2, "live_definitions": result["definitions"]}))
    panels(panel_spec, tmp_path / "panels")
    tasks = sequence(read(tmp_path / "panels/live-workload.json")["tasks"])
    assert len(tasks) == 15
    for value in tasks:
        task = record(value)
        assert task["role"] == "confirmation" and task["scenario"] == "agentic-coding"
        style = str(task["style"])
        assert record(sequence(task["messages"])[0])["content"] == SYSTEMS[style]


def test_agentic_references_use_live_protocol_and_keep_legacy_protocol() -> None:
    ref = Artifact(Path("/example"), "1" * 64, 1)
    spec = Comparison(
        ref,
        ref,
        (ref,),
        ref,
        (Arm("target", None), Arm("baseline", ref)),
        ref,
        ({"id": "task"},),
        1,
        1,
        32768,
        3,
        "dev",
        "live",
        frozenset(),
    )
    original = spec.reference_protocol()
    assert original["policy"] == "native-live-v3"
    for tier in ("agentic-slice", "agentic-suite", "agentic-confirmation"):
        assert replace(spec, tier=tier).reference_protocol() == original
    assert replace(spec, tier="screen").reference_protocol()["policy"] == "native-greedy-request-v3"

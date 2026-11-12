from pathlib import Path

import pytest

from benchmarks.sandbox import Sandbox
from benchmarks.workflows import calls
from w1a1_eagle.contracts import sequence


def test_live_summary_keeps_model_failures_and_aggregate_acceptance() -> None:
    from benchmarks.workflows import summarize
    from w1a1_eagle.contracts import Record

    rows: list[Record] = [
        {
            "wall_seconds": 2.0,
            "model_seconds": 1.0,
            "tool_seconds": 0.5,
            "tokens": 10,
            "native": {"accepted": 8, "proposed": 10},
            "correct": True,
            "completed": True,
        },
        {
            "wall_seconds": 8.0,
            "model_seconds": 7.0,
            "tool_seconds": 0.5,
            "tokens": 20,
            "native": {"accepted": 1, "proposed": 10},
            "correct": True,
            "completed": False,
        },
    ]
    summary = summarize(rows)
    assert summary["tasks"] == 2 and summary["completed"] == 1
    assert summary["completion_rate"] == 0.5 and summary["seconds_per_assigned_task"] == 5
    assert summary["acceptance"] == 0.45 and summary["tokens"] == 30
    assert summary["correct"] is True


def test_live_fixtures_reject_original_accept_repair_and_reset(tmp_path: Path) -> None:
    from benchmarks.native import file_reference
    from benchmarks.project_workflows import prepare
    from w1a1_eagle.records import canonical

    paths = [
        "w1a1_eagle/__init__.py",
        "w1a1_eagle/contracts.py",
        "w1a1_eagle/data.py",
        "w1a1_eagle/records.py",
        "w1a1_eagle/json_stream.py",
        "w1a1_eagle/capacity.py",
        "w1a1_eagle/checkpoints/__init__.py",
        "w1a1_eagle/checkpoints/store.py",
        "benchmarks/__init__.py",
        "benchmarks/coverage.py",
        "benchmarks/statistics.py",
        "benchmarks/protection.py",
    ]
    sources = {name: file_reference(Path(name).resolve()) for name in paths}
    notes = tmp_path / "WORKFLOW.md"
    notes.write_text("# Project workflow\n\nCurrent implementation notes.\n")
    sources["WORKFLOW.md"] = file_reference(notes)
    specification = tmp_path / "spec.json"
    specification.write_bytes(
        canonical(
            {
                "schema": 1,
                "sources": sources,
                "recipes": file_reference(Path("benchmarks/fixtures/project-dev.json").resolve()),
            }
        )
    )
    result = prepare(specification, tmp_path / "prepared")
    assert result["status"] == "complete"
    assert len(sequence(result["checks"])) == 12


def test_native_tool_parser_preserves_code_and_rejects_duplicate_parameters() -> None:
    assert calls(
        "<tool_call>\n<function=write_file>\n<parameter=path>service.py</parameter>\n<parameter=content>def f(x):\n    return x < 3\n</parameter>\n</function>\n</tool_call>"
    ) == [("write_file", {"path": "service.py", "content": "def f(x):\n    return x < 3\n"})]
    assert calls('<tool_call>{"name":"run_tests","arguments":{}}</tool_call>') == [
        ("run_tests", {})
    ]
    with pytest.raises(ValueError, match="duplicate"):
        calls(
            "<tool_call><function=read_file><parameter=path>a</parameter><parameter=path>b</parameter></function></tool_call>"
        )


def test_read_ranges_unique_patch_and_protected_followup_phase(tmp_path: Path) -> None:
    sandbox = Sandbox(
        tmp_path,
        {
            "files": {"value.txt": "old\nsecond\n", "empty.txt": ""},
            "checks": "from pathlib import Path\nassert Path('/work/value.txt').read_text()=='old\\nsecond\\n'\nprint('ACCEPTANCE_CHECKS_PASSED')\n",
            "followup_checks": "from pathlib import Path\nassert Path('/work/value.txt').read_text()=='new\\nsecond\\n'\nprint('ACCEPTANCE_CHECKS_PASSED')\n",
        },
    )
    assert sandbox.invoke("read_file", {"path": "empty.txt"}) == "[Empty file.]"
    assert (
        sandbox.invoke("read_file", {"path": "value.txt", "start_line": 2, "end_line": 2})
        == "second\n"
    )
    assert sandbox.test()["passed"] is True
    sandbox.advance()
    assert sandbox.test()["passed"] is False
    sandbox.invoke("replace_text", {"path": "value.txt", "old": "old", "new": "new"})
    assert sandbox.test()["passed"] is True
    with pytest.raises(ValueError, match="exactly once"):
        sandbox.invoke("replace_text", {"path": "value.txt", "old": "missing", "new": ""})
    with pytest.raises(ValueError, match="no followup"):
        sandbox.advance()

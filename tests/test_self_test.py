from pathlib import Path

import pytest

from tools import comments
from tools.config import REQUIRED_RULES
from tools.contracts import Diagnostic, Record
from tools.pytest_gate import validate_run
from tools.repo_files import read_json
from tools.self_test import assess_outcome, run_case, run_command, run_corpus, validate_roster


def test_rule_and_case_roster() -> None:
    assert len(REQUIRED_RULES) == 125
    cases = {p.name for p in Path("tests/cases").glob("*.json")}
    assert validate_roster(list(REQUIRED_RULES), cases) == []
    assert validate_roster(list(REQUIRED_RULES) + [REQUIRED_RULES[0]], cases)


def test_disabled_checker_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = read_json(Path("tests/cases/comment_python.bad.json"))

    def disabled(path: str, text: str, exceptions: list[Record]) -> list[Diagnostic]:
        return []

    monkeypatch.setattr(comments, "scan_file", disabled)
    exit_code, diagnostics = run_case(case, tmp_path)
    assert not assess_outcome(1, [Diagnostic("COMMENT_PYTHON", "case.py")], exit_code, diagnostics)


def test_missing_tool_is_rejected(tmp_path: Path) -> None:
    result = run_command([str(tmp_path / "missing-tool")], tmp_path)
    assert result.exit == 127
    assert not assess_outcome(
        1,
        [Diagnostic("COMMENT_PYTHON", "case.py")],
        result.exit,
        [Diagnostic("SETUP_ERROR", "tool")],
    )


def test_missing_specimen_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "tests/cases").mkdir(parents=True)
    for source in Path("tests/cases").glob("*.json"):
        if source.name != "comment_python.bad.json":
            (tmp_path / "tests/cases" / source.name).write_bytes(source.read_bytes())
    assert len(list((tmp_path / "tests/cases").glob("*.json"))) == 249
    diagnostics, bad, control = run_corpus(tmp_path, tmp_path, {})
    assert diagnostics == [Diagnostic("RULE_PAIR", "policy/rules.toml")]
    assert bad == control == 0


def test_wrong_and_additional_diagnostics_are_rejected() -> None:
    expected = [Diagnostic("COMMENT_PYTHON", "case.py")]
    assert not assess_outcome(1, expected, 1, [Diagnostic("F821", "case.py")])
    assert not assess_outcome(1, expected, 1, expected + [Diagnostic("F821", "case.py")])
    assert not assess_outcome(1, expected, 0, [])
    assert assess_outcome(1, expected, 1, expected)


def test_required_test_cannot_disappear_or_skip() -> None:
    required = {"a::test_a"}
    assert validate_run(required, ["a::test_a"], {"a::test_a": "passed"}) == []
    for outcomes in [{}, {"a::test_a": "skipped"}, {"a::test_a": "xfail"}, {"a::test_a": "failed"}]:
        assert validate_run(required, ["a::test_a"], outcomes)
    assert validate_run(required, [], {})

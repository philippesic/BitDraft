import hashlib
from pathlib import Path

from tools.comments import scan_file, scan_patch, validate_exceptions
from tools.config import REQUIRED_RULES
from tools.contracts import Diagnostic, Record
from tools.diagnostics import format_diagnostic
from tools.policy_review import validate_report, verification_report
from tools.repo_policy import validate_files


def test_instruction_budget_and_pointer() -> None:
    assert validate_files({"AGENTS.md": "line\n" * 81}) == [
        Diagnostic("INSTRUCTIONS_BUDGET", "AGENTS.md")
    ]
    assert validate_files({"CLAUDE.md": "@AGENTS.md\n"}) == []
    assert validate_files({"nested/AGENTS.md": "extra"})
    assert validate_files({"w1a1_eagle/data/reader.py": "value = 1\n"}) == []
    assert validate_files({"docs/new-design.md": "# Design\n"}) == []
    assert validate_files({"native/owned.cu": "int value;\n"}) == []
    assert validate_files({"runs/reader.py": "value = 1\n"})
    assert validate_files({"results/example/model.pt": "weights"})
    assert validate_files({"results/example/source.py": "value = 1\n"})
    assert validate_files({"results/example/result.md": "# Conclusion\n"}) == []


def test_python_tokens_and_strings() -> None:
    assert scan_file("a.py", 'x = "# actual data"\n', []) == []
    assert scan_file("a.py", "x = 1 # explanation\n", []) == [Diagnostic("COMMENT_PYTHON", "a.py")]
    assert scan_file("a.py", 'class A:\n    "explanation"\n', []) == [
        Diagnostic("COMMENT_DOCSTRING", "a.py")
    ]


def test_native_strings_and_patch_context() -> None:
    assert scan_file("a.cpp", 'const char *s = R"tag(/* data */)tag";\n', []) == []
    assert scan_file("a.cpp", "int x; // explanation\n", [])
    assert scan_file("a.patch", "--- a/a.py\n+++ b/a.py\n@@ -7 +7 @@\n+x = 1\n", []) == [
        Diagnostic("COMMENT_PATCH", "a.patch")
    ]


def test_exact_machine_line() -> None:
    line = "#!/bin/sh"
    entry: Record = {
        "rule_id": "COMMENT_SHELL",
        "path": "a.sh",
        "sha256": hashlib.sha256(line.encode()).hexdigest(),
        "reason": "Executable interpreter",
        "decision_id": "006-comment-ban-and-exceptions",
        "owner": "Philip",
        "review_condition": "Interpreter changes",
    }
    assert scan_file("a.sh", line + "\n", []) == [Diagnostic("EXCEPTION_UNLISTED", "a.sh")]
    assert scan_file("a.sh", line + "\n", [entry]) == []
    assert (
        validate_exceptions({"exception": [entry], "review_date": "2026-11-09"}, {"a.sh": line})
        == []
    )


def test_receipt_cannot_approve_policy() -> None:
    files = {"policy/a.toml": "x = 1\n"}
    stages = [
        "bootstrap",
        "repository",
        "comments",
        "ruff",
        "mypy",
        "architecture",
        "dependencies",
        "documentation",
        "cpu",
        "proofs",
    ]
    receipt = verification_report(files, stages)
    assert validate_report(files, receipt) == []
    assert validate_report({"policy/a.toml": "x = 2\n"}, receipt)
    receipt["approval"] = "approved"
    assert validate_report(files, receipt)


def test_pinned_noninitial_multifile_patch() -> None:
    base = "// unchanged upstream\nint x;\nint y;\n"
    patch = "--- a/a.cpp\n+++ b/a.cpp\n@@ -2,2 +2,2 @@\n int x;\n-int y;\n+int z;\n--- /dev/null\n+++ b/b.cpp\n@@ -0,0 +1 @@\n+int b;\n"
    assert scan_patch("native.patch", patch, [], {"a.cpp": base}) == []
    assert scan_patch(
        "native.patch", patch.replace("+int z;", "+int z; // added"), [], {"a.cpp": base}
    )
    assert scan_patch("native.patch", patch, [], {"a.cpp": base.replace("int y", "int q")})
    assert (
        scan_file("new.patch", "--- /dev/null\n+++ b/new.cpp\n@@ -0,0 +1 @@\n+int n;\n", []) == []
    )


def test_diagnostic_repairs_follow_documented_patterns() -> None:
    files = {
        "docs/checks.md": "| `COMMENT_*` | Remove the comment |\n"
        "| `CHECK_CONFIG`, `PIN_*` | Restore the authorized configuration |\n"
    }
    message = format_diagnostic(Diagnostic("COMMENT_PYTHON", "source.py"), files)
    assert "COMMENT_PYTHON: source.py" in message
    assert "Fix: Remove the comment" in message
    assert "Restore the authorized configuration" in format_diagnostic(
        Diagnostic("PIN_TOOL", "environment"), files
    )
    assert "undocumented diagnostic" in format_diagnostic(Diagnostic("NEW_RULE", "a"), files)


def test_registered_diagnostics_have_repairs_and_comment_location() -> None:
    files = {
        "docs/checks.md": Path("docs/checks.md").read_text(),
        "source.py": "#!/usr/bin/env python3\nvalue = 1\n# reason\n",
    }
    for code in REQUIRED_RULES:
        assert "undocumented diagnostic" not in format_diagnostic(
            Diagnostic(code, "source.py"), files
        ), code
    message = format_diagnostic(Diagnostic("COMMENT_PYTHON", "source.py"), files)
    assert "source.py:3:1" in message
    assert "rationale" in message


def test_policy_checks_properties_without_freezing_unrelated_settings() -> None:
    import copy

    from tools.config import PROJECT, RULES, policy_data
    from tools.contracts import record
    from tools.repo_policy import validate_policy

    project = copy.deepcopy(PROJECT)
    stages = policy_data("check-stages.toml")
    rules: Record = {"rule": list(RULES)}
    assert validate_policy(project, stages, rules) == []
    tool = record(project["tool"])
    record(tool["ruff"])["line-length"] = 99
    assert validate_policy(project, stages, rules) == []
    record(tool["mypy"])["disallow_untyped_calls"] = False
    assert Diagnostic("TYPE_CONFIG", "pyproject.toml") in validate_policy(project, stages, rules)
    record(tool["mypy"])["disallow_untyped_calls"] = True
    record(record(tool["ruff"])["lint"])["per-file-ignores"] = {"*.py": ["F401"]}
    assert Diagnostic("CHECK_CONFIG", "pyproject.toml") in validate_policy(project, stages, rules)


def test_check_dependencies_are_allowed_only_in_checker_and_tests(tmp_path: Path) -> None:
    import json

    from tools.config import Result
    from tools.dependencies import dependency_checks

    for path in ("tools/checker.py", "tests/test_example.py", "w1a1_eagle/model.py"):

        def run(argv: list[str], root: Path) -> Result:
            if argv[0] == "deptry":
                payload: list[Record] = [
                    {"error": {"code": "DEP004"}, "location": {"file": path}, "module": "pytest"}
                ]
                Path(argv[-1]).write_text(json.dumps(payload))
            return Result(0, "")

        diagnostics = dependency_checks(tmp_path, [], tmp_path, run, [])
        assert diagnostics == (
            [Diagnostic("DEP004", path)] if path.startswith("w1a1_eagle/") else []
        )


def test_historical_links_accept_registered_commits_and_refuse_mutable_or_unknown_sources() -> None:
    from tools.docs import validate_docs

    prefix = "https://github.com/philippesic/binary-eagle-decoding/blob/"
    first = prefix + "1" * 40 + "/report.md"
    second = prefix + "2" * 40 + "/report.md"
    evidence: Record = {"object": [{"url": first}, {"url": second}]}
    files = {"README.md": f"[first]({first}) [second]({second})"}
    assert not any(d.code == "EVIDENCE_PIN" for d in validate_docs(files, evidence))
    for target in (prefix + "main/report.md", prefix + "3" * 40 + "/report.md"):
        files["README.md"] = f"[source]({target})"
        assert any(d.code == "EVIDENCE_PIN" for d in validate_docs(files, evidence))


def test_rule_rationale_references_must_resolve() -> None:
    from tools.docs import validate_docs

    files = {"docs/topic.md": "# Topic\n"}
    references: list[Record] = [{"evidence": "docs/topic.md#topic"}]
    rules: Record = {"rule": references}
    assert Diagnostic("DOC_LINK", "policy/rules.toml") not in validate_docs(files, {}, rules)
    for target in ("docs/missing.md", "docs/topic.md#missing"):
        references[0]["evidence"] = target
        assert Diagnostic("DOC_LINK", "policy/rules.toml") in validate_docs(files, {}, rules)

import hashlib

from tools.comments import scan_file, scan_patch, validate_exceptions
from tools.contracts import Diagnostic, Record
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

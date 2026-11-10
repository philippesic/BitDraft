import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

from dulwich.porcelain import init

from tools import boundaries, comments, dependencies, docs, policy_review, pytest_gate, repo_policy
from tools.config import (
    MYPY,
    OTHER_TOOL_CONFIG,
    PINS,
    REQUIRED_RULES,
    RUFF,
    STAGES,
    Result,
    settings,
)
from tools.contracts import Diagnostic, Record, integer, record, sequence, string, validate_contract
from tools.repo_files import read_json, temporary_root


def run_command(argv: list[str], root: Path) -> Result:
    environment = settings(Path(__file__).resolve().parent.parent).environment
    environment["PATH"] = str(Path(sys.executable).parent) + ":" + environment.get("PATH", "")
    try:
        result = subprocess.run(
            argv,
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
        return Result(result.returncode, result.stdout)
    except (OSError, subprocess.TimeoutExpired):
        return Result(127, "SETUP_ERROR")


def validate_pins(data: Record) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if data.get("python") != "3.11.15":
        diagnostics.append(Diagnostic("PIN_PYTHON", "environment"))
    if data.get("tools") != {**PINS, "uv": "0.11.32"}:
        diagnostics.append(Diagnostic("PIN_TOOL", "environment"))
    if data.get("lock_valid") is not True:
        diagnostics.append(Diagnostic("PIN_LOCK", "uv.lock"))
    return diagnostics


def installed_pins(root: Path) -> Record:
    pins: dict[str, str] = {}
    for name in PINS:
        try:
            pins[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pins[name] = "missing"
    uv = run_command(["uv", "--version"], root)
    pins["uv"] = uv.output.split()[1] if uv.exit == 0 and len(uv.output.split()) > 1 else "missing"
    lock = run_command(["uv", "lock", "--check", "--offline"], root)
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "tools": pins,
        "lock_valid": lock.exit == 0,
    }


def assess_outcome(
    expected_exit: int, expected: list[Diagnostic], actual_exit: int, actual: list[Diagnostic]
) -> bool:
    return (
        actual_exit in (0, 1)
        and actual_exit == expected_exit
        and sorted(actual) == sorted(expected)
    )


def validate_roster(ids: list[str], cases: set[str]) -> list[Diagnostic]:
    if (
        tuple(ids) != REQUIRED_RULES
        or len(cases) != 2 * len(REQUIRED_RULES)
        or cases
        != {
            stem.lower() + "." + kind + ".json"
            for stem in REQUIRED_RULES
            for kind in ("bad", "control")
        }
    ):
        return [Diagnostic("RULE_PAIR", "policy/rules.toml")]
    return []


def component(rule: str, data: Record, files: dict[str, str]) -> list[Diagnostic]:
    if rule in {"pin_python", "pin_tool", "pin_lock"}:
        return validate_pins(data)
    if rule in {"check_config", "type_config", "stage_gate"}:
        project: Record = {
            "project": {"dependencies": []},
            "tool": {
                **OTHER_TOOL_CONFIG,
                "ruff": data.get("ruff", RUFF),
                "mypy": data.get("mypy", MYPY),
            },
        }
        stages: Record = {
            "active": data.get("active", list(STAGES)),
            "status": data.get(
                "status",
                [
                    "native unavailable",
                    "runtime verify unavailable",
                    "source parity unavailable",
                    "benchmark unavailable",
                ],
            ),
        }
        rules: Record = {
            "rule": [
                {
                    "id": name,
                    "validator": name.lower(),
                    "evidence": "audit",
                    "bad": "tests/cases/" + name.lower() + ".bad.json",
                    "control": "tests/cases/" + name.lower() + ".control.json",
                    "exit": 1,
                    "diagnostic": name,
                }
                for name in REQUIRED_RULES
            ]
        }
        return repo_policy.validate_policy(project, stages, rules)
    if rule.startswith(("arch_", "owner_call", "owner_path")) or rule in {
        "discovery_branch",
        "env_owner",
        "timing_owner",
        "seeded_random",
        "public_entry",
    }:
        return boundaries.inspect_tree(files)
    if rule.startswith("comment_") or rule == "exception_unlisted":
        exceptions = [record(v) for v in sequence(data.get("exception", []))]
        return sorted(
            {d for path, text in files.items() for d in comments.scan_file(path, text, exceptions)}
        )
    if rule in {"exception_broad", "exception_stale"}:
        return comments.validate_exceptions(data, files)
    if rule.startswith(("instructions_", "claude_", "repo_")):
        return repo_policy.validate_files(
            files, [string(v) for v in sequence(data.get("escapes", []))]
        )
    if rule in {"map_entry", "doc_link", "evidence_pin"}:
        return docs.validate_docs(files, data)
    if rule == "unused_tool":
        return dependencies.reconcile_tools(record(data["uses"]), files)
    if rule == "policy_review":
        return policy_review.validate_report(files, record(data["report"]))
    if rule == "rule_pair":
        return validate_roster(
            [string(v) for v in sequence(data["ids"])], {string(v) for v in sequence(data["cases"])}
        )
    if rule in {"proof_tool", "proof_diagnostic", "proof_disabled"}:
        actual = [
            Diagnostic(string(record(v)["code"]), string(record(v)["path"]))
            for v in sequence(data["actual"])
        ]
        expected = [Diagnostic("COMMENT_PYTHON", "case.py")]
        valid = assess_outcome(1, expected, integer(data["exit"]), actual)
        return [] if valid else [Diagnostic(rule.upper(), "proof")]
    if rule == "required_tests":
        return pytest_gate.validate_run(
            {string(v) for v in sequence(data["required"])},
            [string(v) for v in sequence(data["collected"])],
            {k: string(v) for k, v in record(data["outcomes"]).items()},
        )
    return validate_contract(rule, data)


def tool_component(case: Record, root: Path) -> tuple[int, list[Diagnostic]]:
    command, rule = string(case["command"]), string(case["rule"])
    files = {k: string(v) for k, v in record(case["files"]).items()}
    for path, text in files.items():
        destination = root / path
        if not destination.resolve().is_relative_to(root) or len(text.encode()) > 1048576:
            return 127, [Diagnostic("SETUP_ERROR", "fixture")]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    expected_native = [string(v) for v in sequence(case.get("native", []))]
    native_paths: list[str] = []
    if command == "ruff":
        argv = (
            ["ruff", "format", "--check", "--no-cache", "case.py"]
            if rule == "format"
            else ["ruff", "check", "--no-cache", "--output-format", "json", "case.py"]
        )
        result = run_command(argv, root)
        if result.exit not in (0, 1):
            return 127, [Diagnostic("SETUP_ERROR", "ruff")]
        if rule == "format":
            native = (
                ["format"]
                if result.exit == 1
                and "unformatted:" in result.output
                and "case.py" in result.output
                else []
            )
        else:
            try:
                raw: object = cast(object, json.loads(result.output))
                native = [string(record(v)["code"]) for v in sequence(raw)]
                native_paths = [
                    Path(string(record(v)["filename"])).relative_to(root).as_posix()
                    for v in sequence(raw)
                ]
            except (ValueError, KeyError):
                return 127, [Diagnostic("SETUP_ERROR", "ruff-output")]
    elif command == "mypy":
        focused = record(case.get("focused", {}))
        if set(focused) - {"disallow_any_expr", "disallow_any_explicit"} or any(
            v is not False for v in focused.values()
        ):
            return 127, [Diagnostic("SETUP_ERROR", "focused-config")]
        config_path = root / "pyproject.toml"
        for key in focused:
            config_path.write_text(
                config_path.read_text().replace(key + " = true", key + " = false")
            )
        flags = [string(v) for v in sequence(case.get("flags", []))]
        allowed = {
            "--allow-untyped-defs",
            "--allow-incomplete-defs",
            "--allow-any-explicit",
            "--allow-any-expr",
            "--allow-untyped-calls",
            "--follow-imports=skip",
            "--no-incremental",
            "--warn-unused-configs",
        }
        if any(flag not in allowed for flag in flags):
            return 127, [Diagnostic("SETUP_ERROR", "mypy-flags")]
        result = run_command(
            [
                "mypy",
                "--strict",
                "--no-incremental",
                "--show-error-codes",
                *flags,
                *(["case.py", "consumer.py"] if rule == "type_reexport" else ["case.py"]),
            ],
            root,
        )
        if result.exit not in (0, 1):
            return 127, [Diagnostic("SETUP_ERROR", "mypy")]
        native = [
            m.group(1) for m in re.finditer(r"(?m)^.*error: .* \[([a-z-]+)\]$", result.output)
        ]
        native_paths = [
            m.group(1)
            for m in re.finditer(r"(?m)^(.+?):\d+: error: .* \[([a-z-]+)\]$", result.output)
        ]
        if "Warning: unused section(s)" in result.output:
            native.append("warn-unused-configs")
            native_paths.append("pyproject.toml")
        if result.exit and not native:
            return 127, [Diagnostic("SETUP_ERROR", "mypy-output")]
    elif command == "vulture":
        result = run_command(["vulture", "case.py", "--min-confidence", "60"], root)
        if result.exit not in (0, 3):
            return 127, [Diagnostic("SETUP_ERROR", "vulture")]
        native = (
            ["dead_code"]
            if result.exit == 3 and "unused function 'helper'" in result.output
            else []
        )
    elif command == "deptry":
        fixture_repo = init(str(root))
        fixture_repo.close()
        result = run_command(
            [
                "deptry",
                ".",
                "--config",
                "pyproject.toml",
                "--no-ansi",
                "--json-output",
                str(root / "issues.json"),
            ],
            root,
        )
        if result.exit not in (0, 1) or not (root / "issues.json").exists():
            return 127, [Diagnostic("SETUP_ERROR", "deptry")]
        raw = cast(object, json.loads((root / "issues.json").read_text()))
        native = [string(record(record(v)["error"])["code"]) for v in sequence(raw)]
        native_paths = [string(record(record(v)["location"])["file"]) for v in sequence(raw)]
    else:
        return 127, [Diagnostic("SETUP_ERROR", "command")]
    if not native_paths and native:
        native_paths = ["case.py"] * len(native)
    expected_paths = [
        string(v) for v in sequence(case.get("native_paths", ["case.py"] * len(expected_native)))
    ]
    if sorted(zip(native, native_paths, strict=True)) != sorted(
        zip(expected_native, expected_paths, strict=True)
    ):
        return 127, [Diagnostic("PROOF_DIAGNOSTIC", ",".join(sorted(native)))]
    return (
        (1, sorted({Diagnostic(rule.upper(), path) for path in native_paths}))
        if native
        else (0, [])
    )


def run_case(case: Record, temporary: Path) -> tuple[int, list[Diagnostic]]:
    rule = string(case["rule"])
    if rule.upper() not in REQUIRED_RULES:
        return 127, [Diagnostic("SETUP_ERROR", "rule")]
    if case.get("command") != "component":
        return tool_component(case, temporary)
    files = {k: cast(str, v) for k, v in record(case.get("files", {})).items()}
    for value in sequence(case.get("repeat", [])):
        item = record(value)
        count = integer(item["count"])
        if not 0 <= count <= 1048576:
            return 127, [Diagnostic("SETUP_ERROR", "fixture-size")]
        files[string(item["path"])] = string(item["text"]) * count
    if any(not isinstance(v, str) for v in files.values()):
        return 127, [Diagnostic("SETUP_ERROR", "files")]
    for path, parts in record(case.get("fragments", {})).items():
        files[path] = "".join(string(v) for v in sequence(parts))
    diagnostics = component(rule, record(case["input"]), files)
    return int(bool(diagnostics)), diagnostics


def run_corpus(root: Path, base: Path, digests: Record) -> tuple[list[Diagnostic], int, int]:
    diagnostics = validate_roster(
        list(REQUIRED_RULES), {p.name for p in (root / "tests/cases").glob("*.json")}
    )
    if diagnostics:
        return diagnostics, 0, 0
    bad = control = 0
    for rule in REQUIRED_RULES:
        for kind in ("bad", "control"):
            path = f"tests/cases/{rule.lower()}.{kind}.json"
            source = root / path
            if hashlib.sha256(source.read_bytes()).hexdigest() != digests.get(path):
                diagnostics.append(Diagnostic("PROOF_DISABLED", path))
                continue
            if source.stat().st_size > 65536:
                diagnostics.append(Diagnostic("PROOF_DISABLED", path))
                continue
            case = read_json(source)
            if case.get("rule") != rule.lower() or case.get("exit") != (1 if kind == "bad" else 0):
                diagnostics.append(Diagnostic("PROOF_DISABLED", path))
                continue
            expected = [Diagnostic(rule, string(case["path"]))] if kind == "bad" else []
            if kind == "bad" and "diagnostics" in case:
                expected = [
                    Diagnostic(string(record(v)["code"]), string(record(v)["path"]))
                    for v in sequence(case["diagnostics"])
                ]
                if any(d.code != rule for d in expected):
                    diagnostics.append(Diagnostic("PROOF_DISABLED", path))
                    continue
            with temporary_root(base) as temporary:
                actual_exit, actual = run_case(case, temporary)
            if not assess_outcome(1 if kind == "bad" else 0, expected, actual_exit, actual):
                diagnostics.append(Diagnostic("PROOF_DIAGNOSTIC", path + ":" + str(actual)))
            elif kind == "bad":
                bad += 1
            else:
                control += 1
    return diagnostics, bad, control

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import cast

from packaging.requirements import Requirement

from tools import boundaries, comments, dependencies, docs, policy_review, pytest_gate, repo_policy
from tools.config import (
    MYPY,
    PINS,
    PYTHON,
    REQUIRED_RULES,
    RUFF,
    STAGES,
    TOOL_CONFIG,
    UV,
    Result,
    settings,
)
from tools.contracts import Diagnostic, Record, integer, record, sequence, string, validate_contract
from tools.repo_files import read_json


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
    if data.get("python") != PYTHON:
        diagnostics.append(Diagnostic("PIN_PYTHON", "environment"))
    if data.get("tools") != {**PINS, "uv": UV}:
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


def locked_check_packages(root: Path) -> dict[str, str]:
    result = run_command(
        [
            "uv",
            "--quiet",
            "export",
            "--locked",
            "--offline",
            "--only-group",
            "check",
            "--no-emit-project",
            "--no-header",
            "--no-annotate",
            "--no-hashes",
        ],
        root,
    )
    if result.exit:
        raise ValueError("cannot export locked check environment: " + result.output)
    expected: dict[str, str] = {}
    for line in result.output.splitlines():
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError("check environment requires exact locked versions")
        expected[requirement.name.lower().replace("_", "-")] = specifiers[0].version
    if not expected:
        raise ValueError("locked check environment is empty")
    return expected


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
        observed: Record = {
            "python": data.get("python", PYTHON),
            "tools": {**PINS, "uv": UV, **record(data.get("tools", {}))},
            "lock_valid": data.get("lock_valid", True),
        }
        return validate_pins(observed)
    if rule in {"check_config", "type_config", "stage_gate"}:
        project: Record = {
            "project": {"dependencies": []},
            "tool": {
                **TOOL_CONFIG,
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


def run_case(case: Record) -> tuple[int, list[Diagnostic]]:
    rule = string(case["rule"])
    if rule.upper() not in REQUIRED_RULES:
        return 127, [Diagnostic("SETUP_ERROR", "rule")]
    if case.get("command") != "component":
        return 127, [
            Diagnostic("SETUP_ERROR", "only project-authored component proofs are supported")
        ]
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


def run_corpus(root: Path) -> tuple[list[Diagnostic], int, int]:
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
            actual_exit, actual = run_case(case)
            if not assess_outcome(1 if kind == "bad" else 0, expected, actual_exit, actual):
                diagnostics.append(Diagnostic("PROOF_DIAGNOSTIC", path + ":" + str(actual)))
            elif kind == "bad":
                bad += 1
            else:
                control += 1
    return diagnostics, bad, control

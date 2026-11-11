import importlib.metadata
import os
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    os.execv(sys.executable, [sys.executable, "-m", "tools.check", *sys.argv[1:]])

from tools import boundaries, comments, dependencies, docs, policy_review, repo_policy, self_test
from tools.config import (
    GROUP_COUNTS,
    MYPY,
    REQUIRED_RULES,
    STAGES,
    needs_reexec,
    settings,
)
from tools.contracts import Diagnostic, Record, record, sequence
from tools.diagnostics import format_diagnostic
from tools.repo_files import (
    content_digest,
    index_matches,
    inventory,
    read_toml,
    temporary_root,
)
from w1a1_eagle.clock import Monotonic


def main() -> int:
    root = Path.cwd().resolve()
    config = settings(root)
    config.temporary.mkdir(parents=True, exist_ok=True)
    if needs_reexec(config):
        os.execve(
            sys.executable, [sys.executable, "-m", "tools.check", *sys.argv[1:]], config.environment
        )
    if sys.argv[1:] not in ([], ["--full"], ["--self-test"]):
        print(
            "SETUP_ERROR: modes are tools/check.py (quick, partial), --full (commit gate)"
            " and --self-test (rule proofs only)"
        )
        return 2
    fast = sys.argv[1:] == []

    def mypy_cache(profile: str) -> list[str]:
        if fast:
            return [f"--cache-dir={config.temporary}-mypy-{profile}"]
        return ["--no-incremental", "--cache-dir=/dev/null"]

    try:
        files, escapes = inventory(root)
        project = read_toml(root / "pyproject.toml")
        stages = read_toml(root / "policy/check-stages.toml")
        rules = read_toml(root / "policy/rules.toml")
        evidence = read_toml(root / "policy/evidence.toml")
        exceptions_data = read_toml(root / "policy/exceptions.toml")
        exceptions = [record(v) for v in sequence(exceptions_data["exception"])]
        toolchain = read_toml(root / "policy/toolchain.toml")
        diagnostics = self_test.validate_pins(self_test.installed_pins(root))
        diagnostics.extend(repo_policy.validate_policy(project, stages, rules))
        installed = {
            d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in importlib.metadata.distributions()
        }
        if installed != self_test.locked_check_packages(root):
            diagnostics.append(
                Diagnostic("PIN_TOOL", "installed packages differ from the locked check group")
            )
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            diagnostics.append(Diagnostic("PIN_TOOL", "supported platform"))
        hook = ".githooks/pre-commit"
        if files.get(hook) != (
            "BINARY_CHECK_STAGED=1 exec uv run --locked --no-sync --offline python tools/check.py --full\n"
        ) or not os.access(root / hook, os.X_OK):
            diagnostics.append(Diagnostic("CHECK_CONFIG", hook))
        expected = {"AGENTS.md", "CLAUDE.md"} | {
            f"tests/cases/{name.lower()}.{kind}.json"
            for name in REQUIRED_RULES
            for kind in ("bad", "control")
        }
        if not expected.issubset(files):
            diagnostics.append(
                Diagnostic("STAGE_GATE", "file roster: " + str(sorted(expected - set(files))))
            )
        if config.staged and not index_matches(root, files):
            diagnostics.append(
                Diagnostic("CHECK_CONFIG", "index snapshot differs from authored worktree")
            )

        if diagnostics:
            for d in diagnostics:
                print(format_diagnostic(d, files))
            print("bootstrap FAIL; subsequent stages unavailable")
            return 2
        observed_groups = {
            group: sum(record(v)["group"] == group for v in sequence(rules["rule"]))
            for group in GROUP_COUNTS
        }
        if observed_groups != GROUP_COUNTS:
            print(format_diagnostic(Diagnostic("RULE_PAIR", "group counts differ"), files))
            return 2
        print("0 bootstrap PASS")
        print("rules: " + ", ".join(f"{g}={n}" for g, n in GROUP_COUNTS.items()))
        if sys.argv[1:] == ["--self-test"]:
            proof, bad, control = self_test.run_corpus(root)
            for diagnostic in proof:
                print(format_diagnostic(diagnostic, files))
            print(
                f"{len(REQUIRED_RULES)} rules; {bad}/{len(REQUIRED_RULES)} bad rejected; {control}/{len(REQUIRED_RULES)} controls passed"
            )
            print("proof " + ("FAIL" if proof else "PASS") + "; runtime qualification unavailable")
            return int(bool(proof))
        all_diagnostics: list[Diagnostic] = []
        passed = ["bootstrap"]
        clock = Monotonic()
        last_stage = clock.now()
        paths = sorted(p for p in files if p.endswith(".py"))

        def stage(name: str, diagnostics: list[Diagnostic]) -> None:
            nonlocal last_stage
            all_diagnostics.extend(diagnostics)
            now = clock.now()
            print(
                f"{STAGES.index(name)} {name} {'FAIL' if diagnostics else 'PASS'} ({now - last_stage:.2f}s)"
            )
            last_stage = now
            for diagnostic in diagnostics:
                print(format_diagnostic(diagnostic, files))
            if not diagnostics:
                passed.append(name)

        stage("repository", repo_policy.validate_files(files, escapes))
        stage(
            "comments",
            comments.validate_exceptions(exceptions_data, files)
            + sorted({d for p, t in files.items() for d in comments.scan_file(p, t, exceptions)}),
        )
        with temporary_root(config.temporary) as temporary:
            ruff_format = self_test.run_command(
                ["ruff", "format", "--check", "--no-cache", *paths], root
            )
            ruff_lint = self_test.run_command(
                ["ruff", "check", "--no-cache", "--output-format", "json", *paths], root
            )
            if ruff_format.exit or ruff_lint.exit:
                print(ruff_format.output + ruff_lint.output)
            stage(
                "ruff",
                []
                if ruff_format.exit == ruff_lint.exit == 0
                else [Diagnostic("LINT", "owned Python")],
            )
            numerical_paths = [p for p in paths if p.startswith("tests/numerical/")]
            model_paths = [
                p
                for p in paths
                if p == "w1a1_eagle/model.py" or p.startswith("w1a1_eagle/numerical/")
            ]
            core_paths = [p for p in paths if p not in numerical_paths + model_paths]
            mypy_results = [
                self_test.run_command(
                    [
                        "mypy",
                        "--strict",
                        *mypy_cache("core"),
                        "--show-error-codes",
                        *core_paths,
                    ],
                    root,
                )
            ]
            profile = temporary / "numerical-types.ini"
            profile.write_text(
                "[mypy]\n"
                + "\n".join(
                    key
                    + " = "
                    + (
                        ",".join(str(v) for v in sequence(value))
                        if isinstance(value, list)
                        else str(value)
                    )
                    for key, value in MYPY.items()
                )
                + "\n[mypy-tests.numerical.*]\ndisallow_any_expr = false\ndisallow_untyped_calls = false\n"
            )
            selected = model_paths + numerical_paths
            if selected:
                mypy_results.append(
                    self_test.run_command(
                        [
                            "mypy",
                            *mypy_cache("numerical"),
                            "--python-executable",
                            config.environment["BINARY_NUMERICAL_PYTHON"],
                            "--show-error-codes",
                            "--config-file",
                            str(profile),
                            *selected,
                        ],
                        root,
                    )
                )
            for outcome in mypy_results:
                if outcome.exit or "Warning: unused section(s)" in outcome.output:
                    print(outcome.output)
            stage(
                "mypy",
                []
                if all(
                    outcome.exit == 0 and "Warning: unused section(s)" not in outcome.output
                    for outcome in mypy_results
                )
                else [Diagnostic("TYPES", "owned Python")],
            )
            imports = self_test.run_command(
                ["lint-imports", "--config", "pyproject.toml", "--no-cache"], root
            )
            if imports.exit:
                print(imports.output)
            stage(
                "architecture",
                boundaries.inspect_tree(files)
                + ([] if imports.exit == 0 else [Diagnostic("IMPORTS", "tools")]),
            )
            stage(
                "dependencies",
                dependencies.dependency_checks(
                    root, paths, temporary, self_test.run_command, exceptions
                )
                + dependencies.reconcile_tools(record(toolchain["uses"]), files),
            )
            stage("documentation", docs.validate_docs(files, evidence, rules))
            if fast:
                quick = self_test.run_command(
                    [
                        "pytest",
                        "-q",
                        "--strict-config",
                        "--strict-markers",
                        "-W",
                        "error",
                        "-o",
                        "addopts=",
                        "--ignore=tests/numerical",
                        "--ignore=tests/test_numerical.py",
                        "tests",
                    ],
                    root,
                )
                print(quick.output.strip())
                stage("cpu", [] if quick.exit == 0 else [Diagnostic("REQUIRED_TESTS", "tests")])
        if fast:
            verdict = "FAIL" if all_diagnostics else "PASS"
            print(
                f"quick {verdict}; partial: numerical test, required-test roster and rule proofs"
                " not run; commit gate is --full"
            )
            return int(bool(all_diagnostics))
        with temporary_root(config.temporary):
            cpu = self_test.run_command(
                [
                    "pytest",
                    "-q",
                    "-p",
                    "tools.pytest_gate",
                    "--strict-config",
                    "--strict-markers",
                    "-W",
                    "error",
                    "-o",
                    "addopts=",
                    "--ignore=tests/numerical",
                    "tests",
                ],
                root,
            )
            print(cpu.output.strip())
            stage("cpu", [] if cpu.exit == 0 else [Diagnostic("REQUIRED_TESTS", "tests")])
            proof, bad, control = self_test.run_corpus(root)
            print(
                f"specimens: {bad}/{len(REQUIRED_RULES)} bad rejected; {control}/{len(REQUIRED_RULES)} controls passed"
            )
            stage("proofs", proof)
        report: Record = policy_review.verification_report(files, passed)
        report["content_sha256"] = content_digest(files)
        current, current_escapes = inventory(root)
        if (
            content_digest(current) != content_digest(files)
            or current_escapes != escapes
            or config.staged
            and not index_matches(root, current)
        ):
            all_diagnostics.append(
                Diagnostic("CHECK_CONFIG", "authored content or index changed during verification")
            )
            print(
                format_diagnostic(
                    Diagnostic(
                        "CHECK_CONFIG", "authored content or index changed during verification"
                    ),
                    files,
                )
            )
        if not all_diagnostics:
            import json

            receipt = root / ".venv/foundation-verification.json"
            receipt.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(
            "10 qualification: native/runtime verify/source parity/benchmark UNAVAILABLE; foundation only"
        )
        print(
            "Authority: verification is CPU evidence; it grants no approval for runs or policy changes"
        )
        print("foundation " + ("FAIL" if all_diagnostics else "PASS"))
        return int(bool(all_diagnostics))
    except (OSError, KeyError, ValueError) as error:
        print(f"SETUP_ERROR: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

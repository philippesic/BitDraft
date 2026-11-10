import hashlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    os.execv(sys.executable, [sys.executable, "-m", "tools.check", *sys.argv[1:]])

from tools import boundaries, comments, dependencies, docs, policy_review, repo_policy, self_test
from tools.config import (
    CONFIG_DIGESTS,
    FOUNDATION_FILES,
    GROUP_COUNTS,
    MYPY,
    REQUIRED_CPU,
    REQUIRED_RULES,
    STAGES,
    needs_reexec,
    settings,
)
from tools.contracts import Diagnostic, Record, record, sequence
from tools.repo_files import (
    content_digest,
    index_matches,
    inventory,
    read_toml,
    temporary_root,
    validate_yaml,
)


def main() -> int:
    root = Path.cwd().resolve()
    config = settings(root)
    config.temporary.mkdir(parents=True, exist_ok=True)
    if needs_reexec(config):
        os.execve(
            sys.executable, [sys.executable, "-m", "tools.check", *sys.argv[1:]], config.environment
        )
    if sys.argv[1:] not in ([], ["--self-test"]):
        print("SETUP_ERROR: only --self-test is supported")
        return 2
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
        locked = {
            str(record(v)["name"]): str(record(v)["version"])
            for v in sequence(read_toml(root / "uv.lock")["package"])
            if "registry" in record(record(v)["source"])
        }
        if installed != toolchain["closure"] or any(
            locked.get(k) != v for k, v in installed.items()
        ):
            diagnostics.append(Diagnostic("PIN_TOOL", "installed closure"))
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            diagnostics.append(Diagnostic("PIN_TOOL", "supported platform"))
        for path, digest in CONFIG_DIGESTS.items():
            if hashlib.sha256(files[path].encode()).hexdigest() != digest:
                diagnostics.append(Diagnostic("CHECK_CONFIG", path))
        hook = self_test.run_command(
            ["pre-commit", "validate-config", ".pre-commit-config.yaml"], root
        )
        if hook.exit:
            diagnostics.append(Diagnostic("CHECK_CONFIG", ".pre-commit-config.yaml"))

        expected = set(FOUNDATION_FILES) | {
            f"tests/cases/{name.lower()}.{kind}.json"
            for name in REQUIRED_RULES
            for kind in ("bad", "control")
        }
        if not expected.issubset(files):
            diagnostics.append(
                Diagnostic("STAGE_GATE", "file roster: " + str(sorted(expected - set(files))))
            )
        if read_toml(root / "policy/cpu-tests.toml").get("required") != list(REQUIRED_CPU):
            diagnostics.append(Diagnostic("REQUIRED_TESTS", "policy/cpu-tests.toml"))
        if hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest() != toolchain.get(
            "lock_sha256"
        ):
            diagnostics.append(Diagnostic("PIN_LOCK", "uv.lock"))
        for path in (".pre-commit-config.yaml", ".github/workflows/checks.yml"):
            if not validate_yaml(files[path]):
                diagnostics.append(Diagnostic("CHECK_CONFIG", path))
        if config.staged and not index_matches(root, files):
            diagnostics.append(
                Diagnostic("CHECK_CONFIG", "index snapshot differs from authored worktree")
            )

        if diagnostics:
            for d in diagnostics:
                print(f"{d.code}: {d.path}")
            print("bootstrap FAIL; subsequent stages unavailable")
            return 2
        observed_groups = {
            group: sum(record(v)["group"] == group for v in sequence(rules["rule"]))
            for group in GROUP_COUNTS
        }
        if observed_groups != GROUP_COUNTS:
            print("RULE_PAIR: group counts differ")
            return 2
        print("0 bootstrap PASS")
        print("rules: " + ", ".join(f"{g}={n}" for g, n in GROUP_COUNTS.items()))
        if sys.argv[1:] == ["--self-test"]:
            proof, bad, control = self_test.run_corpus(
                root, config.temporary, record(evidence["fixtures"])
            )
            for diagnostic in proof:
                print(f"{diagnostic.code}: {diagnostic.path}")
            print(
                f"{len(REQUIRED_RULES)} rules; {bad}/{len(REQUIRED_RULES)} bad rejected; {control}/{len(REQUIRED_RULES)} controls passed"
            )
            print("proof " + ("FAIL" if proof else "PASS") + "; runtime qualification unavailable")
            return int(bool(proof))
        all_diagnostics: list[Diagnostic] = []
        passed = ["bootstrap"]
        paths = sorted(p for p in files if p.endswith(".py"))

        def stage(name: str, diagnostics: list[Diagnostic]) -> None:
            all_diagnostics.extend(diagnostics)
            print(f"{STAGES.index(name)} {name} {'FAIL' if diagnostics else 'PASS'}")
            for diagnostic in diagnostics:
                print(f"  {diagnostic.code}: {diagnostic.path}")
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
                        "--no-incremental",
                        "--cache-dir=/dev/null",
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
                            "--no-incremental",
                            "--cache-dir=/dev/null",
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
            stage("documentation", docs.validate_docs(files, evidence))
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
            proof, bad, control = self_test.run_corpus(
                root, config.temporary, record(evidence["fixtures"])
            )
            print(
                f"specimens: {bad}/{len(REQUIRED_RULES)} bad rejected; {control}/{len(REQUIRED_RULES)} controls passed"
            )
            stage("proofs", proof)
        report: Record = policy_review.verification_report(files, passed)
        report["content_sha256"] = content_digest(files)
        if not all_diagnostics:
            import json

            receipt = root / ".venv/foundation-verification.json"
            receipt.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(
            "10 qualification: native/runtime verify/source parity/benchmark UNAVAILABLE; foundation only"
        )
        print("REVIEW_REQUIRED: local verification cannot authenticate Philip's policy approval")
        print("foundation " + ("FAIL" if all_diagnostics else "PASS"))
        return int(bool(all_diagnostics))
    except (OSError, KeyError, ValueError) as error:
        print(f"SETUP_ERROR: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

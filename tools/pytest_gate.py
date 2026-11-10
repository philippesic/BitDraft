from pathlib import Path

import pytest

from tools.contracts import Diagnostic, Record, sequence, string
from tools.repo_files import read_toml


class Gate:
    def __init__(self, required: set[str]) -> None:
        self.required = required
        self.collected: list[str] = []
        self.outcomes: dict[str, str] = {}

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" or report.failed or report.skipped:
            self.outcomes[report.nodeid] = (
                "passed" if report.passed and not hasattr(report, "wasxfail") else "invalid"
            )

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if validate_run(self.required, self.collected, self.outcomes) or exitstatus:
            session.exitstatus = 1


def validate_run(
    required: set[str], collected: list[str], outcomes: dict[str, str]
) -> list[Diagnostic]:
    valid = (
        bool(required)
        and len(collected) == len(set(collected))
        and required.issubset(collected)
        and set(collected) == set(outcomes)
        and all(v == "passed" for v in outcomes.values())
    )
    return [] if valid else [Diagnostic("REQUIRED_TESTS", "tests")]


def pytest_configure(config: pytest.Config) -> None:
    data: Record = read_toml(Path(str(config.rootpath)) / "policy/cpu-tests.toml")
    config.pluginmanager.register(
        Gate({string(v) for v in sequence(data["required"])}), "foundation-gate"
    )

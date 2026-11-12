import hashlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias, cast

from tools.contracts import Record, record, sequence, string
from w1a1_eagle.config import numerical_python


@dataclass(frozen=True)
class Result:
    exit: int
    output: str


Runner: TypeAlias = Callable[[list[str], Path], Result]


@dataclass(frozen=True)
class Settings:
    root: Path
    temporary: Path
    environment: dict[str, str]
    staged: bool


def settings(root: Path) -> Settings:
    temporary = (
        root.resolve()
        / ".venv/check-temp"
        / ("check-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16])
    )
    environment = dict(os.environ)
    environment.update(
        {
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONUTF8": "1",
            "PYTHONPATH": str(root),
            "TMPDIR": str(temporary),
            "PYTEST_DEBUG_TEMPROOT": str(temporary),
            "BINARY_NUMERICAL_PYTHON": numerical_python(root),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_CACHE_DIR": str(root / ".venv/check-cache"),
        }
    )
    return Settings(root, temporary, environment, os.environ.get("BINARY_CHECK_STAGED") == "1")


def needs_reexec(config: Settings) -> bool:
    return any(
        os.environ.get(k) != config.environment[k]
        for k in (
            "LC_ALL",
            "TZ",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONHASHSEED",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTHONUTF8",
        )
    )


def policy_data(name: str) -> Record:
    return cast(
        Record,
        tomllib.loads((Path(__file__).resolve().parent.parent / "policy" / name).read_text()),
    )


RULES = tuple(record(value) for value in sequence(policy_data("rules.toml")["rule"]))
REQUIRED_RULES = tuple(string(value["id"]) for value in RULES)
GROUP_COUNTS = {
    string(value["group"]): sum(item["group"] == value["group"] for item in RULES)
    for value in RULES
}
COMMANDS = policy_data("commands.toml")
BOUNDARIES = policy_data("boundaries.toml")
ENTRY_MODULES = {
    string(value) for key in ("public", "private", "deferred") for value in sequence(COMMANDS[key])
}
STAGES = (
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
)
PROJECT = cast(
    Record, tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
)
TOOL_CONFIG = record(PROJECT["tool"])
RUFF = record(TOOL_CONFIG["ruff"])
MYPY = record(TOOL_CONFIG["mypy"])
PINS = {
    string(value).partition("==")[0]: string(value).partition("==")[2]
    for value in sequence(record(PROJECT["dependency-groups"])["check"])
}
TOOLCHAIN = policy_data("toolchain.toml")
PYTHON = string(TOOLCHAIN["python"])
UV = string(TOOLCHAIN["uv"])

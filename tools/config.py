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
    temporary = Path("/tmp") / (
        "binary-check-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
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
            "PRE_COMMIT_HOME": str(root / ".venv/check-cache/pre-commit"),
            "BINARY_NUMERICAL_PYTHON": numerical_python(root),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_CACHE_DIR": str(root / ".venv/check-cache"),
        }
    )
    return Settings(root, temporary, environment, os.environ.get("PRE_COMMIT") == "1")


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
PINS = {
    "ruff": "0.16.8",
    "mypy": "1.18.2",
    "pytest": "8.4.2",
    "hypothesis": "6.140.3",
    "import-linter": "2.7",
    "vulture": "2.14",
    "deptry": "0.24.0",
    "pre-commit": "4.3.0",
    "PyYAML": "6.0.3",
    "types-PyYAML": "6.0.12.20250915",
    "dulwich": "0.24.2",
}
RUFF: Record = {
    "target-version": "py311",
    "line-length": 100,
    "exclude": [],
    "lint": {
        "select": ["F401", "F821", "I001", "B006", "UP006"],
        "ignore": [],
        "per-file-ignores": {},
    },
}
MYPY: Record = {
    "python_version": "3.11",
    "strict": True,
    "disallow_any_expr": True,
    "disallow_any_explicit": True,
    "disallow_any_unimported": True,
    "warn_unreachable": True,
    "follow_imports": "normal",
    "ignore_missing_imports": False,
    "enable_error_code": ["ignore-without-code"],
    "warn_unused_configs": True,
    "disallow_any_generics": True,
    "disallow_subclassing_any": True,
    "disallow_untyped_calls": True,
    "disallow_untyped_defs": True,
    "disallow_incomplete_defs": True,
    "check_untyped_defs": True,
    "disallow_untyped_decorators": True,
    "warn_redundant_casts": True,
    "warn_unused_ignores": True,
    "warn_return_any": True,
    "implicit_reexport": False,
    "strict_equality": True,
    "strict_bytes": True,
    "extra_checks": True,
}

FOUNDATION_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "FINDINGS.md",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
    ".pre-commit-config.yaml",
    ".github/workflows/checks.yml",
    "Dockerfile.checks",
    "tools/__init__.py",
    "tools/check.py",
    "tools/config.py",
    "tools/repo_files.py",
    "tools/repo_policy.py",
    "tools/comments.py",
    "tools/boundaries.py",
    "tools/contracts.py",
    "tools/docs.py",
    "tools/dependencies.py",
    "tools/pytest_gate.py",
    "tools/self_test.py",
    "tools/policy_review.py",
    "policy/rules.toml",
    "policy/exceptions.toml",
    "policy/boundaries.toml",
    "policy/check-stages.toml",
    "policy/cases.toml",
    "policy/cpu-tests.toml",
    "policy/toolchain.toml",
    "policy/evidence.toml",
    "policy/retained-states.toml",
    "policy/validation.toml",
    "tests/__init__.py",
    "tests/test_contracts.py",
    "tests/test_architecture.py",
    "tests/test_policy.py",
    "tests/test_self_test.py",
    "docs/system-map.md",
    "docs/checks.md",
    "docs/data.md",
    "docs/training.md",
    "docs/runtime.md",
    "docs/benchmarking.md",
    "docs/port-overhaul.md",
    "docs/issue-analysis.md",
    "docs/remaining-stages.md",
    "docs/decisions/README.md",
    "docs/decisions/001-scope-and-names.md",
    "docs/decisions/002-boundaries-and-owners.md",
    "docs/decisions/003-data-and-exposure.md",
    "docs/decisions/004-clocks-and-selection.md",
    "docs/decisions/005-state-and-evidence.md",
    "docs/decisions/006-comment-ban-and-exceptions.md",
    "docs/decisions/007-checks-and-review.md",
    "docs/decisions/008-verify-and-parity.md",
)

REQUIRED_CPU = tuple(string(value) for value in sequence(policy_data("cpu-tests.toml")["required"]))

OTHER_TOOL_CONFIG: Record = {
    "uv": {
        "package": False,
        "python-downloads": "never",
        "default-groups": ["check"],
        "required-environments": ["sys_platform == 'linux' and platform_machine == 'x86_64'"],
        "no-build": True,
    },
    "pytest": {"ini_options": {"testpaths": ["tests"], "addopts": ""}},
    "importlinter": {
        "root_packages": ["tools", "tests", "w1a1_eagle"],
        "include_external_packages": True,
        "contracts": [
            {
                "name": "tools never import tests",
                "type": "forbidden",
                "source_modules": ["tools"],
                "forbidden_modules": ["tests"],
            },
            {
                "name": "contracts stay pure",
                "type": "forbidden",
                "source_modules": ["tools.contracts"],
                "forbidden_modules": [
                    "tools.config",
                    "tools.check",
                    "tools.repo_files",
                    "tools.self_test",
                ],
            },
        ],
    },
    "deptry": {
        "exclude": ["^\\.git/", "^\\.venv/", "^\\.uv-cache/"],
        "extend_exclude": [],
        "ignore": [],
        "per_rule_ignores": {},
        "known_first_party": ["tools", "tests", "w1a1_eagle"],
    },
}

CONFIG_DIGESTS = {
    "policy/rules.toml": "f54020182c172d32326adebd7c5e3ea96de376179eda283a96f2bf58dca7f5c2",
    "policy/commands.toml": "d0b9093bb88ce5e8225cefad890f0507ae4e357fba4122255241b31bc4a9518c",
    ".pre-commit-config.yaml": "30f513d6c67238996c9f8496b2d32ae39c39f3909da15adcc220ca79fc5ae69d",
    ".github/workflows/checks.yml": "b77dd147c5047bf7fab676f08303ab264fda7b50f073241c6c0d5b52e85cdb56",
    "Dockerfile.checks": "c8f0ad1105d976a3f757b74727bf2aed9f4bbfb5030a193f1328ba6f5a9b5696",
    "policy/boundaries.toml": "2222fafd65a0c3d355483f9d0ff3354550f495faefa282b80a327e5365ca024a",
    "policy/cases.toml": "d3094468c15e70cec936fb2b3b2eb6f99ee2900b39a507492f5b762b32ee6cfc",
    "policy/cpu-tests.toml": "936218d7b8f290fa9ad93493c4d9bf6f468fc4434ac5acd3cb7333d5085816d8",
    "policy/check-stages.toml": "b292836462fa970fb2330e93b913ea1a3ac8fe5eb8efde756b691184bf8165f1",
    "policy/validation.toml": "059cdf0120d2fc9c974512f6ec9943b79cceae47719d62ddc47c6a16392d7368",
}

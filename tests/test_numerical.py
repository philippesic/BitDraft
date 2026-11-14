import ast
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.contracts import sequence, string
from tools.repo_files import read_toml
from w1a1_eagle.config import cpu_environment, numerical_python


def numerical_classes() -> dict[str, set[str]]:
    classes: dict[str, set[str]] = {}
    for path in sorted(Path("tests/numerical").glob("test*.py")):
        module = ".".join(path.with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                tests = {
                    f"{module}.{node.name}.{item.name}"
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test")
                }
                if tests:
                    classes[f"{module}.{node.name}"] = tests
    return classes


def run_class(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [numerical_python(Path.cwd()), "-B", "-m", "unittest", "-v", name],
        env=cpu_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )


def test_numerical_profile() -> None:
    classes = numerical_classes()
    assert classes
    with ThreadPoolExecutor(len(classes)) as pool:
        results = dict(zip(classes, pool.map(run_class, classes)))
    completed: set[str] = set()
    for name, result in results.items():
        assert result.returncode == 0 and "OK" in result.stdout, result.stdout
        assert not any(
            word in result.stdout.lower() for word in ("skipped", "expected failures", "unexpected")
        ), result.stdout
        completed |= {
            line.split(" (", 1)[1].split(")", 1)[0]
            for line in result.stdout.splitlines()
            if line.endswith(" ... ok")
        }
    discovered = set().union(*classes.values())
    assert completed == discovered, sorted(completed ^ discovered)
    required = {
        string(v) for v in sequence(read_toml(Path("policy/numerical.toml"))["required_tests"])
    }
    names = {name.rsplit(".", 1)[1] for name in completed}
    assert required and required <= names, sorted(required - names)

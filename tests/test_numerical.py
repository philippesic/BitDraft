import subprocess
from pathlib import Path

from tools.contracts import sequence, string
from tools.repo_files import read_toml
from w1a1_eagle.config import cpu_environment, numerical_python


def test_numerical_profile() -> None:
    result = subprocess.run(
        [
            numerical_python(Path.cwd()),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-v",
            "-s",
            "tests/numerical",
        ],
        env=cpu_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout
    required = {
        string(v) for v in sequence(read_toml(Path("policy/numerical.toml"))["required_tests"])
    }
    completed = {
        line.split(" ", 1)[0] for line in result.stdout.splitlines() if line.endswith(" ... ok")
    }
    assert required and required.issubset(completed) and "OK" in result.stdout, result.stdout
    assert not any(
        word in result.stdout.lower() for word in ("skipped", "expected failures", "unexpected")
    ), result.stdout

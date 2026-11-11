import os
import signal
import subprocess
from pathlib import Path
from typing import cast

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, string

TOOLS: list[Record] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
    for name, description, properties, required in (
        ("list_files", "List project files.", {}, []),
        (
            "read_file",
            "Read a project file; optional inclusive line range starts at one.",
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            ["path"],
        ),
        (
            "replace_text",
            "Replace exactly one matching text block in a project file.",
            {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            ["path", "old", "new"],
        ),
        (
            "write_file",
            "Replace one project file with complete text.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        ("run_tests", "Run the protected project acceptance checks.", {}, []),
    )
]


class Sandbox:
    def __init__(self, root: Path, fixture: Record) -> None:
        self.store = Store(root)
        self.work = Store(root / "work")
        self.checks = Store(root / "checks")
        for name, value in record(fixture["files"]).items():
            if not isinstance(value, str):
                raise ValueError("fixture file must be text")
            self.write(name, value)
        self.checks.atomic(self.checks.root / "check.py", string(fixture["checks"]).encode())
        self.runs = 0
        self.followup_checks = fixture.get("followup_checks")

    def advance(self) -> None:
        if self.followup_checks is None:
            raise ValueError("fixture has no followup phase")
        self.checks.atomic(self.checks.root / "check.py", string(self.followup_checks).encode())
        self.followup_checks = None

    def path(self, name: str) -> Path:
        selected = Path(name)
        if selected.is_absolute() or ".." in selected.parts or not selected.parts:
            raise ValueError("path must stay in the project workspace")
        path = self.work.root / selected
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("symlink paths are forbidden")
        if self.work.root not in path.resolve().parents:
            raise ValueError("path escapes workspace")
        return path

    def write(self, name: str, content: str) -> None:
        if len(content.encode()) > 128 * 1024:
            raise ValueError("file exceeds tool size bound")
        path = self.path(name)
        Store(path.parent).atomic(path, content.encode())

    def invoke(self, name: str, arguments: Record) -> str:
        if name == "list_files" and not arguments:
            return "\n".join(
                str(path.relative_to(self.work.root))
                for path in sorted(self.work.root.rglob("*"))
                if path.is_file()
            )
        if name == "read_file" and {"path"} <= set(arguments) <= {"path", "start_line", "end_line"}:
            path = self.path(string(arguments["path"]))
            if path.stat().st_size > 128 * 1024:
                raise ValueError("file exceeds tool read bound")
            lines = path.read_text().splitlines(keepends=True)
            start = integer(arguments.get("start_line", 1))
            end = integer(arguments.get("end_line", max(1, len(lines))))
            if not 1 <= start <= end:
                raise ValueError("invalid inclusive line range")
            return "".join(lines[start - 1 : end]) if lines else "[Empty file.]"
        if name == "replace_text" and set(arguments) == {"path", "old", "new"}:
            path = self.path(string(arguments["path"]))
            if path.stat().st_size > 128 * 1024:
                raise ValueError("file exceeds tool read bound")
            text = path.read_text()
            old = string(arguments["old"])
            if not old or text.count(old) != 1:
                raise ValueError("old text must match exactly once")
            new = arguments["new"]
            if not isinstance(new, str):
                raise ValueError("replacement must be text")
            self.write(string(arguments["path"]), text.replace(old, new, 1))
            return "Text replaced."
        if name == "write_file" and set(arguments) == {"path", "content"}:
            content = arguments["content"]
            if not isinstance(content, str):
                raise ValueError("file content must be text")
            self.write(string(arguments["path"]), content)
            return "File written."
        if name == "run_tests" and not arguments:
            return str(self.test()["output"])
        raise ValueError("unknown tool or invalid arguments")

    def test(self) -> Record:
        self.runs += 1
        argv = [
            "/usr/bin/prlimit",
            "--as=1073741824",
            "--cpu=20",
            "--nproc=1024",
            "--fsize=4194304",
            "--nofile=128",
            "--",
            "/usr/bin/bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(self.work.root),
            "/work",
            "--ro-bind",
            str(self.checks.root),
            "/checks",
            "--chdir",
            "/work",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "/usr/bin/python3",
            "-I",
            "-B",
            "/checks/check.py",
        ]
        path = self.store.root / ("test-" + str(self.runs) + ".log")
        timed_out = False
        with path.open("wb") as output:
            process: subprocess.Popen[bytes] = subprocess.Popen(
                argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        text = path.read_bytes()[:16384].decode(errors="replace")
        return {
            "passed": cast(int, process.returncode) == 0
            and "ACCEPTANCE_CHECKS_PASSED" in text
            and not timed_out,
            "returncode": process.returncode,
            "timed_out": timed_out,
            "output": text,
        }

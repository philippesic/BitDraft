import json
import re
from pathlib import Path
from typing import cast

from tools.config import PINS, Runner
from tools.contracts import Diagnostic, Record, record, sequence


def reconcile_tools(uses: Record, files: dict[str, str]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    if set(uses) != set(PINS):
        diagnostics.append(Diagnostic("UNUSED_TOOL", "policy/toolchain.toml"))
    for distribution, value in uses.items():
        entry = record(value)
        path, needle = str(entry["path"]), str(entry["needle"])
        if path not in files or needle not in files[path] or not distribution:
            diagnostics.append(Diagnostic("UNUSED_TOOL", "policy/toolchain.toml"))
    return sorted(set(diagnostics))


def dependency_checks(
    root: Path, paths: list[str], temporary: Path, run: Runner, exceptions: list[Record]
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    used: set[tuple[str, str, str]] = set()
    result = run(["vulture", *paths, "--min-confidence", "60"], root)
    if result.exit not in (0, 3):
        return [Diagnostic("SETUP_ERROR", "vulture")]
    for line in result.output.splitlines():
        match = re.match(r"(.+?):(\d+): (.+)", line)
        if match:
            path, lineno, message = match.groups()
            symbol_match = re.search(r"'([^']+)'", message)
            symbol = symbol_match.group(1) if symbol_match else ""
            used.add(("VULTURE", path, symbol))
            if not any(
                e.get("rule_id") == "VULTURE"
                and e.get("path") == path
                and e.get("symbol") == symbol
                for e in exceptions
            ):
                diagnostics.append(Diagnostic("DEAD_CODE", f"{path}:{lineno}:{symbol}"))
        elif line.strip():
            diagnostics.append(Diagnostic("SETUP_ERROR", "vulture-output"))
    output = temporary / "deptry.json"
    result = run(
        ["deptry", ".", "--config", "pyproject.toml", "--no-ansi", "--json-output", str(output)],
        root,
    )
    if result.exit not in (0, 1) or not output.exists():
        return diagnostics + [Diagnostic("SETUP_ERROR", "deptry")]
    raw: object = cast(object, json.loads(output.read_text()))
    for value in sequence(raw):
        issue = record(value)
        code = str(record(issue["error"])["code"])
        location = record(issue["location"])
        path = str(location["file"])
        module = str(issue["module"])
        used.add((code, path, module))
        if not any(
            e.get("rule_id") == code and e.get("path") == path and e.get("symbol") == module
            for e in exceptions
        ):
            diagnostics.append(Diagnostic(code.upper(), path))
    for entry in exceptions:
        if (
            entry["rule_id"] in {"VULTURE", "DEP004"}
            and (str(entry["rule_id"]), str(entry["path"]), str(entry["symbol"])) not in used
        ):
            diagnostics.append(Diagnostic("EXCEPTION_STALE", str(entry["path"])))
    return diagnostics

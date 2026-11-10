import re

from tools.config import MYPY, OTHER_TOOL_CONFIG, REQUIRED_RULES, RUFF, STAGES
from tools.contracts import Diagnostic, Record, record, sequence


def validate_files(files: dict[str, str], escapes: list[str] | None = None) -> list[Diagnostic]:
    result: set[Diagnostic] = set()
    for path, text in files.items():
        if path.rsplit("/", 1)[-1].lower() in {"agents.md", "claude.md"} and path not in {
            "AGENTS.md",
            "CLAUDE.md",
        }:
            result.add(Diagnostic("INSTRUCTIONS_ROOT", path))
        if path == "AGENTS.md" and (len(text.splitlines()) > 80 or len(text.encode()) > 4000):
            result.add(Diagnostic("INSTRUCTIONS_BUDGET", path))
        if path == "CLAUDE.md" and text != "@AGENTS.md\n":
            result.add(Diagnostic("CLAUDE_POINTER", path))
        if (
            path.split("/")[0]
            in {
                "runs",
                "results",
                "checkouts",
                "models",
                "captures",
                "checkpoints",
                "src",
                "w1a1_eagle",
                "benchmarks",
                "native",
            }
            or path.endswith((".pt", ".gguf", ".safetensors", ".so", ".cu"))
            or re.search(r"(?i)generated (?:code|source)|do not edit.*generated", text[:500])
        ):
            result.add(Diagnostic("REPO_OUTPUTS", path))
        if "\x00" in text or len(text.encode()) > (1048576 if path == "uv.lock" else 262144):
            result.add(Diagnostic("REPO_LARGE", path))
        if re.search(
            r"(?im)-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bsk-[A-Za-z0-9]{32,}|\bAKIA[A-Z0-9]{16}\b|^\s*(?:password|api_key|secret_token)\s*=\s*[\"'][^\"']{8,}",
            text,
        ):
            result.add(Diagnostic("REPO_SECRET", path))
        if re.search(r"(?i)(?:^|/)(?:STATUS|HISTORY|PLAN|SCRATCH)(?:[._-]|$)", path):
            result.add(Diagnostic("REPO_DIARY", path))
    if sum(len(t.encode()) for p, t in files.items() if p != "uv.lock") > 2097152:
        result.add(Diagnostic("REPO_LARGE", "."))
    for path in escapes or []:
        result.add(Diagnostic("REPO_IGNORED", path))
    for line in files.get(".gitignore", "").splitlines():
        if line.strip() and line not in {
            ".venv/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            ".hypothesis/",
            ".uv-cache/",
            ".env",
            "*.pem",
            "*.key",
            "artifacts/",
        }:
            result.add(Diagnostic("REPO_IGNORED", ".gitignore"))
    return sorted(result)


def validate_policy(project: Record, stages: Record, rules: Record) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    try:
        tool = record(project["tool"])
        if (
            any(tool.get(k) != v for k, v in OTHER_TOOL_CONFIG.items())
            or record(project["project"]).get("dependencies") != []
        ):
            diagnostics.append(Diagnostic("CHECK_CONFIG", "pyproject.toml"))
        if record(tool["ruff"]) != RUFF:
            diagnostics.append(Diagnostic("CHECK_CONFIG", "pyproject.toml"))
        if record(tool["mypy"]) != MYPY:
            diagnostics.append(Diagnostic("TYPE_CONFIG", "pyproject.toml"))
        if stages["active"] != list(STAGES):
            diagnostics.append(Diagnostic("CHECK_CONFIG", "policy/check-stages.toml"))
        if stages["status"] != [
            "native unavailable",
            "runtime verify unavailable",
            "source parity unavailable",
            "benchmark unavailable",
        ]:
            diagnostics.append(Diagnostic("STAGE_GATE", "policy/check-stages.toml"))
        roster = [record(v) for v in sequence(rules["rule"])]
        if any(
            r.get("validator") != str(r.get("id", "")).lower()
            or r.get("diagnostic") != r.get("id")
            or r.get("exit") != 1
            or r.get("bad") != "tests/cases/" + str(r.get("id", "")).lower() + ".bad.json"
            or r.get("control") != "tests/cases/" + str(r.get("id", "")).lower() + ".control.json"
            for r in roster
        ):
            diagnostics.append(Diagnostic("RULE_PAIR", "policy/rules.toml"))

        if tuple(r["id"] for r in roster) != REQUIRED_RULES or any(
            not all(r.get(k) for k in ("validator", "evidence", "bad", "control", "diagnostic"))
            for r in roster
        ):
            diagnostics.append(Diagnostic("RULE_PAIR", "policy/rules.toml"))
    except (KeyError, ValueError):
        diagnostics.append(Diagnostic("CHECK_CONFIG", "pyproject.toml"))
    return diagnostics

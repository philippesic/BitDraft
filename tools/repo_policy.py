import re

from tools.config import REQUIRED_RULES, STAGES
from tools.contracts import Diagnostic, Record, record, sequence

TASK_SECTIONS = ("Goal", "Authorized", "Done when", "State", "Next")


def sections(text: str) -> list[str]:
    result: list[str] = []
    for match in re.finditer(r"(?m)^## (.+)$", text):
        heading: str = match.group(1)
        result.append(heading)
    return result


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
                "checkouts",
                "models",
                "captures",
                "checkpoints",
            }
            or path.startswith("results/")
            and not path.endswith((".md", ".json", ".csv", ".txt", ".svg"))
            or path.endswith((".pt", ".gguf", ".safetensors", ".so"))
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
        if (
            path != "tasks/README.md"
            and re.fullmatch(r"tasks/[^/]+\.md", path)
            and (
                not text.startswith("# ")
                or sections(text)[: len(TASK_SECTIONS)] != list(TASK_SECTIONS)
            )
        ):
            result.add(Diagnostic("REPO_TASK", path))
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
            ".host.toml",
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
        mypy = record(tool["mypy"])
        strict_flags = (
            "strict",
            "disallow_any_expr",
            "disallow_any_explicit",
            "disallow_any_unimported",
            "warn_unreachable",
            "warn_unused_configs",
            "disallow_any_generics",
            "disallow_subclassing_any",
            "disallow_untyped_calls",
            "disallow_untyped_defs",
            "disallow_incomplete_defs",
            "check_untyped_defs",
            "disallow_untyped_decorators",
            "warn_redundant_casts",
            "warn_unused_ignores",
            "warn_return_any",
            "strict_equality",
            "strict_bytes",
            "extra_checks",
        )
        if (
            any(mypy.get(flag) is not True for flag in strict_flags)
            or mypy.get("ignore_missing_imports") is not False
            or mypy.get("implicit_reexport") is not False
            or mypy.get("follow_imports") != "normal"
            or "ignore-without-code" not in sequence(mypy.get("enable_error_code", []))
            or any(mypy.get(key) for key in ("overrides", "disable_error_code", "exclude"))
        ):
            diagnostics.append(Diagnostic("TYPE_CONFIG", "pyproject.toml"))
        ruff = record(tool["ruff"])
        lint = record(ruff["lint"])
        uv = record(tool["uv"])
        deptry = record(tool["deptry"])
        imports = record(tool["importlinter"])
        if (
            record(project["project"]).get("dependencies") != []
            or not {"F401", "F821", "I001", "B006", "UP006"}.issubset(sequence(lint["select"]))
            or any(
                lint.get(key)
                for key in (
                    "ignore",
                    "extend-ignore",
                    "per-file-ignores",
                    "extend-per-file-ignores",
                )
            )
            or any(ruff.get(key) for key in ("exclude", "extend-exclude"))
            or uv.get("package") is not False
            or uv.get("python-downloads") != "never"
            or uv.get("no-build") is not True
            or uv.get("default-groups") != ["check"]
            or any(deptry.get(key) for key in ("ignore", "per_rule_ignores", "extend_exclude"))
            or deptry.get("exclude") != ["^\\.git/", "^\\.venv/", "^\\.uv-cache/"]
            or set(sequence(deptry["known_first_party"])) != {"tools", "tests", "w1a1_eagle"}
            or set(sequence(imports["root_packages"])) != {"tools", "tests", "w1a1_eagle"}
            or imports.get("include_external_packages") is not True
        ):
            diagnostics.append(Diagnostic("CHECK_CONFIG", "pyproject.toml"))
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

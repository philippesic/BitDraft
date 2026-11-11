import ast
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from tools.contracts import Diagnostic, Record, record, sequence, string


def anchors(text: str) -> set[str]:
    result: set[str] = set()
    for match in re.finditer(r"(?m)^#{1,6} (.+)$", text):
        heading: str = match.group(1)
        clean: str = re.sub(r"[^\w\- ]", "", heading.lower())
        result.add(clean.replace(" ", "-"))
    return result


def validate_docs(
    files: dict[str, str], evidence: Record, rules: Record | None = None
) -> list[Diagnostic]:
    diagnostics: set[Diagnostic] = set()
    registered = {record(v)["url"] for v in sequence(evidence.get("object", []))}
    for path, text in files.items():
        if not path.endswith(".md"):
            continue
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1)
            parsed = urlparse(target)
            if parsed.scheme in {"https", "http"}:
                if "binary-eagle-decoding" in target and (
                    re.search(r"/blob/[0-9a-f]{40}/", target) is None or target not in registered
                ):
                    diagnostics.add(Diagnostic("EVIDENCE_PIN", path))
                continue
            local, _, anchor = unquote(target).partition("#")
            parts: list[str] = list(PurePosixPath(path).parent.parts)
            for part in PurePosixPath(local).parts:
                if part == "..":
                    if not parts:
                        break
                    parts.pop()
                elif part != ".":
                    parts.append(part)
            destination = "/".join(parts) if local else path
            if (
                destination not in files
                or anchor
                and anchor not in anchors(files.get(destination, ""))
            ):
                diagnostics.add(Diagnostic("DOC_LINK", path))
    for value in sequence((rules or {}).get("rule", [])):
        destination, _, anchor = string(record(value)["evidence"]).partition("#")
        if destination not in files or anchor and anchor not in anchors(files[destination]):
            diagnostics.add(Diagnostic("DOC_LINK", "policy/rules.toml"))
    mapped: list[str] = []
    for match in re.finditer(
        r"(?m)^\| (\w+) \| (active|pending) \| ([^ |]+) \| ([^ |]+) \|",
        files.get("docs/architecture.md", ""),
    ):
        flow: str = match.group(1)
        state: str = match.group(2)
        mapped_path: str = match.group(3)
        symbol: str = match.group(4)
        if state == "pending":
            continue
        mapped.append(flow)
        if mapped_path not in files:
            diagnostics.add(Diagnostic("MAP_ENTRY", "docs/architecture.md"))
        elif mapped_path.endswith(".py"):
            names = {
                n.name
                for n in ast.walk(ast.parse(files[mapped_path]))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            if symbol not in names:
                diagnostics.add(Diagnostic("MAP_ENTRY", "docs/architecture.md"))
    if len(mapped) != len(set(mapped)) or not {"check", "policy", "selftest", "setup"}.issubset(
        mapped
    ):
        diagnostics.add(Diagnostic("MAP_ENTRY", "docs/architecture.md"))
    return sorted(diagnostics)

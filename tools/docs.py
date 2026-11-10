import ast
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from tools.contracts import Diagnostic, Record, record, sequence


def anchors(text: str) -> set[str]:
    result: set[str] = set()
    for match in re.finditer(r"(?m)^#{1,6} (.+)$", text):
        heading: str = match.group(1)
        clean: str = re.sub(r"[^\w\- ]", "", heading.lower())
        result.add(clean.replace(" ", "-"))
    return result


def validate_docs(files: dict[str, str], evidence: Record) -> list[Diagnostic]:
    diagnostics: set[Diagnostic] = set()
    registered = {record(v)["url"] for v in sequence(evidence.get("object", []))}
    revision = str(evidence.get("old_revision", ""))
    for path, text in files.items():
        if not path.endswith(".md"):
            continue
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1)
            parsed = urlparse(target)
            if parsed.scheme in {"https", "http"}:
                if "binary-eagle-decoding" in target and (
                    "/blob/" not in target
                    or f"/blob/{revision}/" not in target
                    or target not in registered
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
    mapped: list[str] = []
    for match in re.finditer(
        r"(?m)^\| (\w+) \| (active|pending) \| ([^ |]+) \| ([^ |]+) \|",
        files.get("docs/system-map.md", ""),
    ):
        flow: str = match.group(1)
        state: str = match.group(2)
        mapped_path: str = match.group(3)
        symbol: str = match.group(4)
        if state == "pending":
            continue
        mapped.append(flow)
        if mapped_path not in files:
            diagnostics.add(Diagnostic("MAP_ENTRY", "docs/system-map.md"))
        elif mapped_path.endswith(".py"):
            names = {
                n.name
                for n in ast.walk(ast.parse(files[mapped_path]))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            if symbol not in names:
                diagnostics.add(Diagnostic("MAP_ENTRY", "docs/system-map.md"))
    if len(mapped) != len(set(mapped)) or not {"check", "policy", "selftest", "setup"}.issubset(
        mapped
    ):
        diagnostics.add(Diagnostic("MAP_ENTRY", "docs/system-map.md"))
    return sorted(diagnostics)

import ast
import hashlib
import io
import re
import tokenize
from pathlib import PurePosixPath

from tools.contracts import Diagnostic, Record, record, sequence, string

TEXT_SUFFIXES = {".md", ".json", ".lock", ".csv", ".txt"}
HASH_SUFFIXES = {".sh", ".bash", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".cmake"}
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}


def machine_line(text: str) -> bool:
    return bool(
        re.match(
            r"^#!\S|^#\s*(type:\s*ignore\[[\w,-]+\]|noqa:\s*[A-Z0-9, ]+|pragma:|coding[:=])|^//\s*SPDX-License-Identifier:",
            text,
        )
    )


def lexical_comments(text: str, native: bool = False, semicolon: bool = False) -> list[str]:
    pattern = (
        r'''(?:R"(?P<delim>[^ ()\\\t\n]{0,16})\([\s\S]*?\)(?P=delim)"|"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27)|(?P<comment>/\*[\s\S]*?\*/|//[^\n]*)'''
        if native
        else r'''(?:"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27)|(?P<comment>\#[^\n]*)'''
    )
    pattern = r"\\[^\n]|\[(?P<bracket>=*)\[[\s\S]*?\](?P=bracket)\]|" + pattern
    if semicolon:
        pattern = pattern.replace(r"\#[^\n]*", r"[\#;][^\n]*")
    return [
        m.group("comment") for m in re.finditer(pattern, text) if m.group("comment") is not None
    ]


def mask_data(text: str, suffix: str) -> str:
    if suffix not in {".yaml", ".yml", ".sh", ".bash"}:
        return text
    output: list[str] = []
    block_indent: int | None = None
    terminator: str | None = None
    for line in text.splitlines(keepends=True):
        indent = len(line) - len(line.lstrip(" "))
        if terminator is not None:
            output.append("\n")
            if line.strip() == terminator:
                terminator = None
            continue
        if block_indent is not None:
            if not line.strip() or indent > block_indent:
                output.append("\n")
                continue
            block_indent = None
        output.append(line)
        if suffix in {".yaml", ".yml"} and re.search(
            r":\s*[|>][-+0-9]*\s*(?:#.*)?$", line.rstrip()
        ):
            block_indent = indent
        if suffix in {".sh", ".bash"}:
            match = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
            if match:
                terminator = match.group(1)
    return "".join(output)


def scan_file(path: str, text: str, exceptions: list[Record]) -> list[Diagnostic]:
    suffix = PurePosixPath(path).suffix
    name = PurePosixPath(path).name
    comments: list[str] = []
    code = "COMMENT_OTHER"
    if suffix == ".py":
        code = "COMMENT_PYTHON"
        try:
            comments = [
                t.string
                for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type == tokenize.COMMENT
            ]
            tree = ast.parse(text)
        except (SyntaxError, tokenize.TokenError, IndentationError):
            return [Diagnostic("COMMENT_UNKNOWN", path)]
        if any(
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
            for n in ast.walk(tree)
        ):
            return [Diagnostic("COMMENT_DOCSTRING", path)]
    elif suffix in NATIVE_SUFFIXES:
        code = "COMMENT_NATIVE"
        comments = lexical_comments(text, native=True)
    elif (
        suffix in HASH_SUFFIXES
        or name.startswith("Dockerfile")
        or name in {".gitignore", ".dockerignore", ".editorconfig", "CMakeLists.txt"}
    ):
        code = {
            ".sh": "COMMENT_SHELL",
            ".bash": "COMMENT_SHELL",
            ".toml": "COMMENT_TOML",
            ".yaml": "COMMENT_YAML",
            ".yml": "COMMENT_YAML",
        }.get(suffix, "COMMENT_OTHER")
        comments = lexical_comments(
            mask_data(text, suffix), semicolon=suffix in {".ini", ".cfg"} or name == ".editorconfig"
        )
    elif suffix in {".patch", ".diff"}:
        diagnostics: list[Diagnostic] = []
        target = ""
        context: list[str] = []
        for line in text.splitlines():
            if line.startswith("+++ b/"):
                target = line[6:]
            elif line.startswith("@@"):
                if not re.match(r"@@ -1(?:,\d+)? \+1(?:,\d+)? @@", line):
                    return [Diagnostic("COMMENT_PATCH", path)]
            elif line.startswith(("+", " ")):
                context.append(line[1:])
        if not target:
            return [Diagnostic("COMMENT_PATCH", path)]
        if scan_file(target, "\n".join(context) + "\n", exceptions):
            diagnostics.append(Diagnostic("COMMENT_PATCH", path))
        return diagnostics
    elif suffix == ".md":
        diagnostics = []
        for match in re.finditer(r"(?m)^```([^\n]*)\n([\s\S]*?)^```", text):
            lang, body = match.group(1), match.group(2)
            extension = {
                "python": ".py",
                "sh": ".sh",
                "bash": ".sh",
                "toml": ".toml",
                "yaml": ".yaml",
                "json": ".json",
                "text": ".txt",
                "": ".txt",
            }.get(lang)
            if extension is None:
                diagnostics.append(Diagnostic("COMMENT_UNKNOWN", path))
            else:
                diagnostics.extend(
                    Diagnostic(d.code, path)
                    for d in scan_file("fence" + extension, body, exceptions)
                )
        return sorted(set(diagnostics))
    elif suffix in TEXT_SUFFIXES or name == ".python-version":
        return []
    else:
        return [Diagnostic("COMMENT_UNKNOWN", path)]
    diagnostics = []
    for comment in comments:
        digest = hashlib.sha256(comment.encode()).hexdigest()
        selectors = {digest} | {
            hashlib.sha256(line.strip().encode()).hexdigest()
            for line in text.splitlines()
            if line.strip().endswith(comment)
        }
        matched = any(e.get("path") == path and e.get("sha256") in selectors for e in exceptions)
        if machine_line(comment):
            if not matched:
                diagnostics.append(Diagnostic("EXCEPTION_UNLISTED", path))
        else:
            diagnostics.append(Diagnostic(code, path))
    return sorted(set(diagnostics))


def validate_exceptions(data: Record, files: dict[str, str]) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    seen: set[tuple[str, str]] = set()
    try:
        for value in sequence(data["exception"]):
            entry = record(value)
            path, digest = string(entry["path"]), string(entry["sha256"])
            if any(c in path for c in "*?[") or not all(
                string(entry[k])
                for k in ("rule_id", "reason", "decision_id", "owner", "review_condition")
            ):
                diagnostics.append(Diagnostic("EXCEPTION_BROAD", "policy/exceptions.toml"))
                continue
            match = [
                line.strip()
                for line in files.get(path, "").splitlines()
                if hashlib.sha256(line.strip().encode()).hexdigest() == digest
            ]
            if (
                (path, digest) in seen
                or not match
                or string(entry.get("expiry", "9999-12-31")) < string(data["review_date"])
            ):
                diagnostics.append(Diagnostic("EXCEPTION_STALE", "policy/exceptions.toml"))
            elif (
                not machine_line(match[0])
                and not (entry["rule_id"] == "TYPE_DIRECTIVE" and "# type: ignore[" in match[0])
                and entry["rule_id"]
                not in {
                    "VULTURE",
                    "DEP004",
                    "PYTEST_ENTRY",
                }
            ):
                diagnostics.append(Diagnostic("EXCEPTION_BROAD", "policy/exceptions.toml"))
            seen.add((path, digest))
    except (KeyError, ValueError, TypeError):
        diagnostics.append(Diagnostic("EXCEPTION_BROAD", "policy/exceptions.toml"))
    return sorted(set(diagnostics))

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


def lexical_spans(
    text: str, native: bool = False, semicolon: bool = False
) -> list[tuple[str, int, int]]:
    pattern = (
        r'''(?:R"(?P<delim>[^ ()\\\t\n]{0,16})\([\s\S]*?\)(?P=delim)"|"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27)|(?P<comment>/\*[\s\S]*?\*/|//[^\n]*)'''
        if native
        else r'''(?:"""[\s\S]*?"""|\x27\x27\x27[\s\S]*?\x27\x27\x27|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27)|(?P<comment>\#[^\n]*)'''
    )
    pattern = r"\\[^\n]|\[(?P<bracket>=*)\[[\s\S]*?\](?P=bracket)\]|" + pattern
    if semicolon:
        pattern = pattern.replace(r"\#[^\n]*", r"[\#;][^\n]*")
    return [
        (m.group("comment"), m.start("comment"), m.end("comment"))
        for m in re.finditer(pattern, text)
        if m.group("comment") is not None
    ]


def lexical_comments(text: str, native: bool = False, semicolon: bool = False) -> list[str]:
    return [value for value, _, _ in lexical_spans(text, native, semicolon)]


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


def scan_patch(
    path: str, text: str, exceptions: list[Record], bases: dict[str, str]
) -> list[Diagnostic]:
    lines = text.splitlines()
    index = 0
    seen = False
    try:
        while index < len(lines):
            if not lines[index].startswith("--- "):
                index += 1
                continue
            old_name = lines[index][4:]
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise ValueError("missing patch post-image")
            new_name = lines[index][4:]
            target = (
                new_name.removeprefix("b/")
                if new_name != "/dev/null"
                else old_name.removeprefix("a/")
            )
            if PurePosixPath(target).is_absolute() or ".." in PurePosixPath(target).parts:
                raise ValueError("patch path escape")
            index += 1
            hunks: list[tuple[int, int, int, int, list[str]]] = []
            while index < len(lines) and not lines[index].startswith(("--- ", "diff --git ")):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index])
                if not match:
                    raise ValueError("unsupported patch framing")
                old_start, old_count, new_start, new_count = (
                    int(match[1]),
                    int(match[2] or "1"),
                    int(match[3]),
                    int(match[4] or "1"),
                )
                index += 1
                body: list[str] = []
                while index < len(lines) and not lines[index].startswith(
                    ("@@", "--- ", "diff --git ")
                ):
                    if lines[index] == "\\ No newline at end of file":
                        raise ValueError(
                            "non-newline post-images need explicit byte reconstruction"
                        )
                    if not lines[index] or lines[index][0] not in " +-":
                        raise ValueError("malformed patch body")
                    body.append(lines[index])
                    index += 1
                if (
                    sum(line[0] in " -" for line in body) != old_count
                    or sum(line[0] in " +" for line in body) != new_count
                ):
                    raise ValueError("hunk counts disagree")
                hunks.append((old_start, old_count, new_start, new_count, body))
            if not hunks:
                raise ValueError("empty file patch")
            base = bases.get(target)
            if base is None:
                if len(hunks) != 1 or hunks[0][0] not in (0, 1) or hunks[0][2] not in (0, 1):
                    raise ValueError("noninitial hunks require a pinned complete base")
                base = "".join(line[1:] + "\n" for line in hunks[0][4] if line[0] in " -")
            original = base.splitlines(keepends=True)
            result: list[str] = []
            added: set[int] = set()
            cursor = 0
            for old_start, old_count, new_start, new_count, body in hunks:
                offset = old_start - 1 if old_count else old_start
                if offset < cursor or offset > len(original):
                    raise ValueError("overlapping or out-of-range hunk")
                result.extend(original[cursor:offset])
                cursor = offset
                if new_count and new_start != len(result) + 1:
                    raise ValueError("post-image hunk offset disagrees")
                for line in body:
                    if line[0] in " -":
                        if cursor >= len(original) or original[cursor] != line[1:] + "\n":
                            raise ValueError("pinned base disagrees with patch")
                        cursor += 1
                    if line[0] in " +":
                        result.append(line[1:] + "\n")
                        if line[0] == "+":
                            added.add(len(result))
            result.extend(original[cursor:])
            post = "".join(result)
            suffix = PurePosixPath(target).suffix
            if suffix == ".py":
                tree = ast.parse(post)
                tokens = tokenize.generate_tokens(io.StringIO(post).readline)
                if any(
                    token.type == tokenize.COMMENT
                    and added.intersection(range(token.start[0], token.end[0] + 1))
                    for token in tokens
                ):
                    raise ValueError("added Python comment")
                if any(
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and added.intersection(range(node.lineno, (node.end_lineno or node.lineno) + 1))
                    for node in ast.walk(tree)
                ):
                    raise ValueError("added Python documentation string")
            elif suffix in NATIVE_SUFFIXES or suffix in HASH_SUFFIXES:
                for comment, start, end in lexical_spans(
                    post, native=suffix in NATIVE_SUFFIXES, semicolon=suffix in {".ini", ".cfg"}
                ):
                    first = post.count("\n", 0, start) + 1
                    last = post.count("\n", 0, end) + 1
                    if added.intersection(range(first, last + 1)):
                        if not machine_line(comment) or scan_file(
                            target, comment + "\n", exceptions
                        ):
                            raise ValueError("added comment or unregistered machine line")
            else:
                raise ValueError("unsupported patch language")
            seen = True
        if not seen:
            raise ValueError("no patch files")
    except (ValueError, SyntaxError, tokenize.TokenError, IndentationError):
        return [Diagnostic("COMMENT_PATCH", path)]
    return []


def scan_file(path: str, text: str, exceptions: list[Record]) -> list[Diagnostic]:
    suffix = ".sh" if path.startswith(".githooks/") else PurePosixPath(path).suffix
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
        return scan_patch(path, text, exceptions, {})
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

import fnmatch
import io
import re
import tokenize

from tools.comments import machine_line
from tools.contracts import Diagnostic


def format_diagnostic(diagnostic: Diagnostic, files: dict[str, str]) -> str:
    location = diagnostic.path
    if diagnostic.code == "COMMENT_PYTHON" and diagnostic.path in files:
        try:
            for token in tokenize.generate_tokens(io.StringIO(files[diagnostic.path]).readline):
                if token.type == tokenize.COMMENT and not machine_line(token.string):
                    location += f":{token.start[0]}:{token.start[1] + 1}"
                    break
        except (tokenize.TokenError, IndentationError):
            location = diagnostic.path
    for line in files.get("docs/checks.md", "").splitlines():
        columns = line.split("|")
        if len(columns) != 4:
            continue
        patterns = [match.group(1) for match in re.finditer(r"`([^`]+)`", columns[1])]
        if any(fnmatch.fnmatchcase(diagnostic.code, pattern) for pattern in patterns):
            repair = columns[2].strip()
            return f"{diagnostic.code}: {location}\n  Fix: {repair}"
    return (
        f"{diagnostic.code}: {location}\n"
        "  Fix: inspect this failure and docs/checks.md#fixes; report an undocumented diagnostic."
    )

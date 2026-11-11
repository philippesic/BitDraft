import hashlib

from tools.contracts import Diagnostic, Record


def content_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, text in sorted(files.items()):
        digest.update(path.encode() + b"\0" + text.encode() + b"\0")
    return digest.hexdigest()


def policy_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, text in sorted(files.items()):
        if path.startswith(("policy/", "tools/", "tests/", ".githooks/")) or path in {
            "pyproject.toml",
            "uv.lock",
            "AGENTS.md",
            "CLAUDE.md",
        }:
            digest.update(path.encode() + b"\0" + text.encode() + b"\0")
    return digest.hexdigest()


def verification_report(files: dict[str, str], results: list[str]) -> Record:
    return {
        "policy_sha256": policy_digest(files),
        "stages": results,
        "approval": "REVIEW_REQUIRED",
        "scope": "local CPU infrastructure and source fixtures; no native or scientific qualification",
        "content_sha256": content_digest(files),
    }


def validate_report(files: dict[str, str], report: Record) -> list[Diagnostic]:
    valid = (
        report.get("policy_sha256") == policy_digest(files)
        and report.get("content_sha256") == content_digest(files)
        and report.get("approval") == "REVIEW_REQUIRED"
        and report.get("stages")
        == [
            "bootstrap",
            "repository",
            "comments",
            "ruff",
            "mypy",
            "architecture",
            "dependencies",
            "documentation",
            "cpu",
            "proofs",
        ]
    )
    return [] if valid else [Diagnostic("POLICY_REVIEW", "policy/validation.toml")]

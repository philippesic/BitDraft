import ast
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from benchmarks.native import file_reference
from benchmarks.project_recipes import check_script
from benchmarks.protection import source_names
from benchmarks.sandbox import TOOLS
from benchmarks.sessions import (
    PERMISSIVE,
    change_checks,
    clean_message,
    commit_order,
    components,
    json_lines,
    parses,
    safe_path,
    text_value,
)
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import artifact, canonical, identity, read, reference

SOURCE = "session-long-project"
LICENSE_FILE = re.compile(r"^(?:licen[cs]e|copying)(?:[-._][a-z0-9.-]+)?$", re.IGNORECASE)
COPYLEFT = (
    "gnu general public license",
    "gnu lesser general public license",
    "gnu library general public license",
    "gnu affero general public license",
    "mozilla public license",
    "eclipse public license",
    "european union public licence",
    "sharealike",
)
CHANGED_BYTES = 48 * 1024
WORKSPACE_FILE_BYTES = 128 * 1024
LICENSE_BYTES = 64 * 1024
ARCHIVE_BYTES = 256 * 1024 * 1024
SHINGLE = 13
SHARED_SHINGLES = 12
CONTAINMENT = 0.35
EXACT_TOKENS = 25
CODE_TOKEN = re.compile(r"\w+|[^\w\s]")


def license_kind(text: str) -> str | None:
    folded = " ".join(text.lower().split())
    if any(marker in folded for marker in COPYLEFT):
        return None
    if "apache license" in folded and "version 2.0" in folded:
        return "apache-2.0"
    if "permission is hereby granted, free of charge" in folded:
        return "mit"
    if "redistribution and use in source and binary forms" in folded:
        if "endorse or promote" in folded:
            return "bsd-3-clause"
        return "bsd-2-clause"
    if "permission to use, copy, modify, and/or distribute this software for any purpose" in (
        folded
    ) or "permission to use, copy, modify, and distribute this software for any purpose" in (
        folded
    ):
        return "isc"
    return None


def project_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", value.rsplit("/", 1)[-1].lower())
    for prefix in ("python", "py"):
        if name.startswith(prefix) and len(name) - len(prefix) >= 3:
            name = name[len(prefix) :]
    return name


@dataclass(frozen=True)
class Protected:
    families: frozenset[str]
    names: frozenset[str]
    commits: frozenset[str]

    def overlap(self, repositories: list[str], commits: list[str]) -> str | None:
        if any(name.lower() in self.families for name in repositories):
            return "protected_family"
        if any(project_name(name) in self.names for name in repositories):
            return "protected_name"
        if any(commit in self.commits for commit in commits):
            return "protected_commit"
        return None

    def path_overlap(self, path: str) -> bool:
        parts = PurePosixPath(path).with_suffix("").parts
        return any(re.sub(r"[^a-z0-9]", "", part.lower()) in self.names for part in parts)


def protected_projects(definition: Record, lineage: Record) -> Protected:
    families: set[str] = set()
    commits: set[str] = set()
    for value in sequence(definition["projects"]):
        project = record(value)
        families.add(string(project["repository"]).lower())
        commits.add(string(project["revision"]))
        commits.update(string(entry).split()[0] for entry in sequence(project["first_commits"]))
    for value in sequence(lineage["projects"]):
        project = record(value)
        families.add(string(project["repository"]).lower())
        families.update(string(name).lower() for name in sequence(project["known_family"]))
        for match in sequence(project["raw_source_history_matches"]):
            entry = record(match)
            commits.add(string(entry["commit"]))
            families.update(string(name).lower() for name in sequence(entry["repositories"]))
    names = frozenset(project_name(name) for name in families)
    return Protected(frozenset(families), names, frozenset(commits))


def code_tokens(text: str) -> list[str]:
    return [match.group(0) for match in CODE_TOKEN.finditer(text)]


def shingles(tokens: list[str]) -> set[int]:
    return {hash(tuple(tokens[index : index + SHINGLE])) for index in range(len(tokens) - 12)}


class ProtectedCode:
    def __init__(self, texts: list[str]) -> None:
        self.index: dict[int, list[int]] = {}
        self.sizes: list[int] = []
        self.exact: set[str] = set()
        for number, text in enumerate(texts):
            tokens = code_tokens(text)
            values = shingles(tokens)
            self.sizes.append(len(values))
            for value in values:
                self.index.setdefault(value, []).append(number)
            if len(tokens) >= EXACT_TOKENS:
                self.exact.add(identity({"tokens": tokens}))

    def matches(self, text: str) -> bool:
        tokens = code_tokens(text)
        if len(tokens) >= EXACT_TOKENS and identity({"tokens": tokens}) in self.exact:
            return True
        values = shingles(tokens)
        shared: Counter[int] = Counter()
        for value in values:
            shared.update(self.index.get(value, ()))
        return any(
            count >= SHARED_SHINGLES and count >= CONTAINMENT * min(len(values), self.sizes[number])
            for number, count in shared.items()
        )


def snapshot(archive: Path) -> tuple[dict[str, str], dict[str, str]]:
    files: dict[str, str] = {}
    licenses: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or parts[0] in {"/", ".."}:
                continue
            name = "/".join(parts[1:])
            license_file = len(parts) == 2 and LICENSE_FILE.match(parts[1]) is not None
            if license_file and member.size > LICENSE_BYTES:
                licenses[name] = ""
                continue
            if not (license_file or safe_path(name) and member.size <= WORKSPACE_FILE_BYTES):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            try:
                text = extracted.read().decode()
            except UnicodeDecodeError:
                if license_file:
                    licenses[name] = ""
                continue
            (licenses if license_file else files)[name] = text
    return files, licenses


def distance(left: str, right: str) -> int:
    first, second = PurePosixPath(left).parent.parts, PurePosixPath(right).parent.parts
    common = 0
    for one, two in zip(first, second):
        if one != two:
            break
        common += 1
    return len(first) + len(second) - 2 * common


def module_names(path: str) -> list[str]:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[index:]) for index in range(len(parts))]


def imported(path: str, text: str) -> list[str]:
    names: list[str] = []
    package = list(PurePosixPath(path).parent.parts)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level - 1 > len(package):
                    continue
                base = package[: len(package) - (node.level - 1)]
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            if module:
                names.append(module)
            names.extend(
                (module + "." if module else "") + alias.name
                for alias in node.names
                if alias.name != "*"
            )
    return names


def resolved(path: str, text: str, index: dict[str, list[str]]) -> list[str]:
    targets: list[str] = []
    for name in imported(path, text):
        matches = index.get(name, [])
        if not matches:
            continue
        nearest = min((distance(path, other), other) for other in matches)[1]
        if nearest != path and nearest not in targets:
            targets.append(nearest)
    return targets


def relevance(name: str, files: dict[str, str]) -> list[str]:
    index: dict[str, list[str]] = {}
    for path in sorted(files):
        for dotted in module_names(path):
            index.setdefault(dotted, []).append(path)
    direct = resolved(name, files[name], index)
    reverse = [
        path
        for path in sorted(files)
        if path != name and name in resolved(path, files[path], index)
    ]
    rest = [path for _, path in sorted((distance(name, path), path) for path in files)]
    ordered: list[str] = []
    for path in [name, *direct, *reverse, *rest]:
        if path not in ordered:
            ordered.append(path)
    return ordered


@dataclass(frozen=True)
class Settings:
    minimum_tokens: int
    maximum_tokens: int
    context_bytes: int
    workspace_files: int
    workspace_bytes: int

    @staticmethod
    def parse(raw: Record) -> "Settings":
        bounds = [integer(value) for value in sequence(raw["initial_tokens"])]
        settings = Settings(
            bounds[0] if len(bounds) == 2 else 0,
            bounds[1] if len(bounds) == 2 else 0,
            integer(raw["context_bytes"]),
            integer(raw["workspace_files"]),
            integer(raw["workspace_bytes"]),
        )
        if (
            len(bounds) != 2
            or not 1024 <= settings.minimum_tokens < settings.maximum_tokens <= 28672
            or not CHANGED_BYTES <= settings.context_bytes <= 256 * 1024
            or not 2 <= settings.workspace_files <= 1000
            or not settings.context_bytes <= settings.workspace_bytes <= 16 * 1024 * 1024
        ):
            raise ValueError("invalid long-project context or workspace bounds")
        return settings


def source_prompt(name: str, message: str, context: list[str], files: dict[str, str]) -> str:
    blocks = "".join(
        "### "
        + path
        + "\n```python\n"
        + files[path]
        + ("" if files[path].endswith("\n") else "\n")
        + "```\n\n"
        for path in reversed(context)
    )
    return (
        "Project source for reference; the full project is in the workspace.\n\n"
        + blocks
        + "Change request:\n"
        + message
        + "\n\nThe change belongs in "
        + name
        + ". The source above is current. Make the edit with the file tools and run the tests."
    )


def long_task(
    candidate: Record,
    files: dict[str, str],
    licenses: dict[str, str],
    settings: Settings,
    count: Callable[[list[Record]], int],
    system: str,
    cap: int,
    code: ProtectedCode,
    protected: Protected,
) -> tuple[Record | None, str]:
    row = record(candidate["row"])
    name, old, new = (text_value(row, key) for key in ("old_file", "old_contents", "new_contents"))
    kinds = {path: license_kind(text) for path, text in sorted(licenses.items())}
    if not kinds:
        return None, "no_license_file"
    if any(kind is None for kind in kinds.values()) or string(row["license"]) not in PERMISSIVE:
        return None, "license_not_permissive"
    if files.get(name) != new:
        return None, "snapshot_disagrees"
    ranked = [
        path for _, path in sorted((distance(name, path), path) for path in files if path != name)
    ]
    workspace: dict[str, str] = {name: old}
    size = len(old.encode())
    for path in ranked:
        if len(workspace) == settings.workspace_files:
            break
        text = files[path]
        if size + len(text.encode()) > settings.workspace_bytes or not parses(text):
            continue
        workspace[path] = text
        size += len(text.encode())
    if any(protected.path_overlap(path) for path in workspace):
        return None, "protected_path"
    if any(code.matches(text) for text in workspace.values()):
        return None, "protected_content"
    message = clean_message(text_value(row, "message"))
    context: list[str] = []
    used = 0
    for path in relevance(name, workspace):
        added = len(workspace[path].encode())
        if used + added <= settings.context_bytes:
            context.append(path)
            used += added
    if name not in context:
        return None, "changed_file_exceeds_context"

    def messages() -> list[Record]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": source_prompt(name, message, context, workspace)},
        ]

    tokens = count(messages())
    while tokens > settings.maximum_tokens and len(context) > 1:
        context.pop()
        tokens = count(messages())
    if not settings.minimum_tokens <= tokens <= settings.maximum_tokens:
        return None, "context_tokens_out_of_band"
    commit = string(candidate["commit"])
    key = commit + ":" + name
    component = string(candidate["component"])
    repositories = [string(value) for value in sequence(candidate["repositories"])]
    archive = record(candidate["archive"])
    task: Record = {
        "id": SOURCE + "-" + identity({"key": key})[:24],
        "source": SOURCE,
        "source_key": key,
        "group": "commitpack:" + component,
        "aliases": sorted(
            {
                "commitpack:" + component,
                key,
                identity({"text": old}),
                *("repository:" + value.lower() for value in repositories),
            }
        ),
        "role": "train",
        "category": "code",
        "subtype": "agentic_session",
        "license": string(row["license"]),
        "prior_use": "unknown",
        "messages": messages(),
        "output_cap": cap,
        "fixture": {
            "schema": 1,
            "files": workspace,
            "checks": check_script(change_checks(name, old)),
        },
        "project": {
            "repository": string(candidate["repository"]),
            "repositories": repositories,
            "commit": commit,
            "url": string(candidate["url"]),
            "archive_sha256": string(archive["sha256"]),
            "licenses": {path: kind for path, kind in kinds.items()},
            "license_files": {path: identity({"text": text}) for path, text in licenses.items()},
            "initial_tokens": tokens,
            "context_files": list(reversed(context)),
            "workspace_files": len(workspace),
            "workspace_bytes": size,
        },
    }
    return task, "accepted"


def candidate_row(rows: list[Record]) -> Record | None:
    for row in sorted(rows, key=commit_order):
        name, old, new = (
            text_value(row, key) for key in ("old_file", "old_contents", "new_contents")
        )
        message = clean_message(text_value(row, "message"))
        if (
            string(row["license"]) in PERMISSIVE
            and safe_path(name)
            and text_value(row, "new_file") == name
            and 1024 <= len(old.encode()) <= CHANGED_BYTES
            and 0 < len(new.encode()) <= CHANGED_BYTES
            and old != new
            and 16 <= len(message) <= 600
            and parses(old)
            and parses(new)
        ):
            return row
    return None


def prior_session_components(
    tasks: list[Record], grouped: dict[str, list[Record]]
) -> tuple[set[str], int]:
    owner = {string(row["commit"]): name for name, rows in grouped.items() for row in rows}
    found: set[str] = set()
    for task in tasks:
        commit = string(task["source_key"]).split(":", 1)[0]
        if commit not in owner:
            raise ValueError("prior TRAIN session commit is missing from the declared source")
        found.add(owner[commit])
    return found, len(tasks)


def plan(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw)
        != {
            "schema",
            "role",
            "source",
            "prior_source",
            "prior_sessions",
            "definition",
            "lineage",
            "seed",
            "candidates",
        }
        or raw["schema"] != 1
        or raw["role"] != "train"
        or output.exists()
    ):
        raise ValueError("explicit TRAIN long-project plan and a new output required")
    root = specification.resolve().parent
    source, prior = artifact(raw["source"], root), artifact(raw["prior_source"], root)
    definition, lineage = artifact(raw["definition"], root), artifact(raw["lineage"], root)
    session_refs = [artifact(value, root) for value in sequence(raw["prior_sessions"])]
    for ref in (source, prior, definition, lineage, *session_refs):
        ref.verify()
    protected = protected_projects(read(definition.path), read(lineage.path))
    rows = json_lines(source.path)
    grouped = components(rows)
    seen = {name for row in json_lines(prior.path) for name in string(row["repos"]).split(",")}
    prior_tasks = [
        record(task)
        for ref in session_refs
        for task in sequence(read(ref.path, 256 * 1024 * 1024)["tasks"])
    ]
    sessions, session_tasks = prior_session_components(prior_tasks, grouped)
    seed = integer(raw["seed"])
    wanted = integer(raw["candidates"])
    if wanted < 1:
        raise ValueError("long-project plan needs a positive candidate count")
    ranked = sorted((identity({"group": name, "seed": seed}), name) for name in grouped)
    refused: Counter[str] = Counter()
    candidates: list[Record] = []
    for _, name in ranked:
        members = grouped[name]
        repositories = sorted({repo for row in members for repo in string(row["repos"]).split(",")})
        reason = protected.overlap(repositories, [string(row["commit"]) for row in members])
        if reason is not None:
            refused[reason] += 1
            continue
        if any(repo in seen for repo in repositories):
            refused["curated_pool_repository"] += 1
            continue
        if name in sessions:
            refused["existing_train_session_project"] += 1
            continue
        if int(identity({"group": "commitpack:" + name})[:8], 16) % 5 == 0:
            refused["dev_marked_group"] += 1
            continue
        row = candidate_row(members)
        if row is None:
            refused["no_permissive_parseable_change"] += 1
            continue
        commit = string(row["commit"])
        candidates.append(
            {
                "component": name,
                "repositories": repositories,
                "commit": commit,
                "file": text_value(row, "old_file"),
                "urls": [
                    "https://codeload.github.com/" + repo + "/tar.gz/" + commit
                    for repo in sorted(string(row["repos"]).split(","))
                ],
                "row": row,
            }
        )
        if len(candidates) == wanted:
            break
    if len(candidates) != wanted:
        raise ValueError("source cannot supply the requested long-project candidates")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(
        store.root / "candidates.json",
        canonical(
            {
                "schema": 1,
                "seed": seed,
                "source": reference(source),
                "candidates": candidates,
            }
        ),
    )
    result: Record = {
        "status": "planned",
        "role": "train",
        "candidates": file_reference(store.root / "candidates.json"),
        "count": len(candidates),
        "components": len(grouped),
        "examined": len(candidates) + sum(refused.values()),
        "refused": dict(sorted(refused.items())),
        "prior_session_tasks": session_tasks,
        "prior_session_projects": len(sessions),
        "protected_families": sorted(protected.families),
        "protected_names": sorted(protected.names),
        "source": reference(source),
        "prior_source": reference(prior),
        "prior_sessions": [reference(ref) for ref in session_refs],
        "definition": reference(definition),
        "lineage": reference(lineage),
        "code_sha256": code,
        "scope": "seeded TRAIN candidate order of permissive CommitPackFT single-file commits, one per repository family; DEV agentic, confirmation, curated-pool and existing session families excluded before any download",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result


def prepare(specification: Path, output: Path) -> Record:
    from w1a1_eagle.numerical.prompting import Tokenizer

    raw = read(specification)
    if (
        set(raw)
        != {
            "schema",
            "role",
            "plan",
            "snapshots",
            "tokenizer",
            "target",
            "harnesses",
            "exclusions",
            "heldout_exclusions",
            "additional_exclusions",
            "upstream",
            "initial_tokens",
            "context_bytes",
            "workspace_files",
            "workspace_bytes",
            "output_cap",
            "tasks",
        }
        or raw["schema"] != 1
        or raw["role"] != "train"
        or output.exists()
    ):
        raise ValueError("explicit TRAIN long-project preparation and a new output required")
    settings = Settings.parse(raw)
    root = specification.resolve().parent
    refs = {
        key: artifact(raw[key], root)
        for key in ("plan", "snapshots", "tokenizer", "target", "harnesses", "upstream")
    }
    for ref in refs.values():
        ref.verify()
    plan_result = read(refs["plan"].path)
    if plan_result.get("status") != "planned" or plan_result.get("role") != "train":
        raise ValueError("long-project preparation needs a completed TRAIN plan")
    candidates_ref = artifact(plan_result["candidates"], refs["plan"].path.parent)
    definition = artifact(plan_result["definition"], refs["plan"].path.parent)
    lineage = artifact(plan_result["lineage"], refs["plan"].path.parent)
    for ref in (candidates_ref, definition, lineage):
        ref.verify()
    cap = integer(raw["output_cap"])
    wanted = integer(raw["tasks"])
    if not 1 <= cap <= 2048 or wanted < 1:
        raise ValueError("invalid long-project output cap or task count")
    system = string(read(refs["harnesses"].path)["tools"])
    protected_ids: set[str] = {"heldout24"}
    exclusion_refs = [
        artifact(raw["exclusions"], root),
        artifact(raw["heldout_exclusions"], root),
        *(artifact(value, root) for value in sequence(raw["additional_exclusions"])),
    ]
    for ref in exclusion_refs:
        ref.verify()
        protected_ids.update(string(value) for value in sequence(read(ref.path)["identities"]))
    manifest = read(refs["snapshots"].path, 64 * 1024 * 1024)
    if record(manifest["candidates"])["sha256"] != candidates_ref.sha256:
        raise ValueError("snapshot manifest belongs to different candidates")
    fetched = {
        string(record(value)["commit"]): record(value) for value in sequence(manifest["snapshots"])
    }
    protected = protected_projects(read(definition.path), read(lineage.path))
    texts: list[str] = []
    for value in sequence(read(refs["upstream"].path)["files"]):
        ref = artifact(value, refs["upstream"].path.parent)
        ref.verify()
        texts.append(ref.path.read_text(errors="replace"))
    code = ProtectedCode(texts)
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code_sha = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code_sha, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    tokenizer = Tokenizer(refs["tokenizer"].path, refs["target"].path, store.root / "tokenizer.log")
    tasks: list[Record] = []
    refused: Counter[str] = Counter()
    try:

        def count(messages: list[Record]) -> int:
            return len(tokenizer.tokenize(tokenizer.render(messages, TOOLS)))

        for value in sequence(read(candidates_ref.path, 512 * 1024 * 1024)["candidates"]):
            if len(tasks) == wanted:
                break
            candidate = record(value)
            commit = string(candidate["commit"])
            reason = protected.overlap(
                [string(name) for name in sequence(candidate["repositories"])], [commit]
            )
            if reason is not None:
                raise ValueError("long-project candidate overlaps a protected project: " + reason)
            if commit not in fetched:
                raise ValueError("snapshot manifest stops before the requested tasks; fetch more")
            entry = fetched[commit]
            if entry["status"] != "fetched":
                refused["snapshot_unavailable"] += 1
                continue
            archive = artifact(entry["archive"], refs["snapshots"].path.parent)
            archive.verify()
            if archive.size > ARCHIVE_BYTES:
                refused["archive_too_large"] += 1
                continue
            try:
                files, licenses = snapshot(archive.path)
            except (tarfile.TarError, EOFError, OSError):
                refused["archive_unreadable"] += 1
                continue
            task, outcome = long_task(
                candidate
                | {
                    "repository": entry["repository"],
                    "url": entry["url"],
                    "archive": reference(archive),
                },
                files,
                licenses,
                settings,
                count,
                system,
                cap,
                code,
                protected,
            )
            if task is None:
                refused[outcome] += 1
                continue
            if source_names(task) & protected_ids:
                refused["protected_identity"] += 1
                continue
            tasks.append(task)
        template = tokenizer.template_sha256
    finally:
        tokenizer.close()
    if len(tasks) != wanted:
        raise ValueError("long-project snapshots cannot supply the requested TRAIN session tasks")
    tokens = [integer(record(task["project"])["initial_tokens"]) for task in tasks]
    store.atomic(store.root / "tasks.json", canonical({"schema": 1, "tasks": tasks}))
    result: Record = {
        "status": "prepared",
        "role": "train",
        "source": SOURCE,
        "tasks": file_reference(store.root / "tasks.json"),
        "count": len(tasks),
        "examined": len(tasks) + sum(refused.values()),
        "refused_examined": dict(sorted(refused.items())),
        "initial_tokens": {
            "minimum": min(tokens),
            "maximum": max(tokens),
            "mean": sum(tokens) / len(tokens),
        },
        "tasks_per_family": 1,
        "output_cap": cap,
        "template_sha256": template,
        "plan": reference(refs["plan"]),
        "snapshots": reference(refs["snapshots"]),
        "harnesses": reference(refs["harnesses"]),
        "target_sha256": refs["target"].sha256,
        "exclusions": [reference(ref) for ref in exclusion_refs],
        "upstream": reference(refs["upstream"]),
        "code_sha256": code_sha,
        "scope": "TRAIN tool-session tasks in full permissive repository snapshots with relevant source in the initial context; licence, DEV/confirmation family, path and code-content exclusions applied; target-only success is not required",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

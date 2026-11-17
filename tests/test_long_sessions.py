import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks import long_projects
from benchmarks.long_projects import (
    Protected,
    ProtectedCode,
    Settings,
    license_kind,
    long_task,
    protected_projects,
    snapshot,
)
from benchmarks.sandbox import Sandbox
from benchmarks.sessions import components, prepared_tasks, project_task, session_tasks
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.records import canonical, identity, read

MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy"
GPL = "GNU GENERAL PUBLIC LICENSE Version 3"
PROTECTED = Protected(
    frozenset({"hukkin/tomli", "jmoiron/humanize", "python-humanize/humanize"}),
    frozenset({"tomli", "humanize"}),
    frozenset({"c" * 40}),
)
LIBRARY = "".join(
    "def parse_value_" + str(index) + "(text, position):\n"
    "    value = text[position:].strip()\n"
    "    if not value:\n"
    "        raise ValueError('empty value at ' + str(position))\n"
    "    return value, position + len(value)\n\n"
    for index in range(12)
)
SETTINGS = Settings(100, 4000, 64 * 1024, 20, 512 * 1024)


def reference(path: Path) -> Record:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_json(path: Path, value: object) -> Record:
    path.write_bytes(canonical(value))
    return reference(path)


def count(messages: list[Record]) -> int:
    return len(string(record(messages[-1])["content"])) // 4


def project_files() -> dict[str, str]:
    helpers = "".join("HELPER_" + str(index) + " = " + str(index) + "\n" for index in range(400))
    return {
        "app/core.py": "from app import helpers\n\n\ndef total(values):\n    return sum(values)\n"
        + helpers,
        "app/helpers.py": helpers,
        "app/views.py": "from app.core import total\n\n\ndef render(values):\n    return str(total(values))\n",
        "docs/conf.py": "project = 'app'\n",
        "setup.py": "from setuptools import setup\nsetup(name='app')\n",
    }


def candidate(files: dict[str, str]) -> Record:
    new = files["app/core.py"]
    old = new.replace("return sum(values)", "return sum(values) + 0", 1)
    return {
        "component": "owner/app",
        "repositories": ["owner/app"],
        "repository": "owner/app",
        "commit": "a" * 40,
        "url": "https://codeload.github.com/owner/app/tar.gz/" + "a" * 40,
        "archive": {"path": "archive.tar.gz", "sha256": "0" * 64, "bytes": 1},
        "row": {
            "commit": "a" * 40,
            "old_file": "app/core.py",
            "new_file": "app/core.py",
            "old_contents": old,
            "new_contents": new,
            "message": "Simplify the total\n\nSigned-off-by: Someone <someone@example.com>",
            "license": "mit",
            "repos": "owner/app",
        },
    }


def build(
    files: dict[str, str], licenses: dict[str, str], protected: Protected = PROTECTED
) -> tuple[Record | None, str]:
    return long_task(
        candidate(files),
        files,
        licenses,
        SETTINGS,
        count,
        "system",
        1024,
        ProtectedCode([LIBRARY]),
        protected,
    )


def test_license_classification_keeps_only_permissive_texts() -> None:
    assert license_kind("MIT License\n\n" + MIT) == "mit"
    assert license_kind("Apache License\nVersion 2.0, January 2004") == "apache-2.0"
    assert (
        license_kind(
            "Redistribution and use in source and binary forms ... "
            "Neither the name ... may be used to endorse or promote products"
        )
        == "bsd-3-clause"
    )
    assert license_kind("Redistribution and use in source and binary forms") == "bsd-2-clause"
    assert (
        license_kind(
            "Permission to use, copy, modify, and/or distribute this software for any purpose"
        )
        == "isc"
    )
    assert license_kind(GPL) is None
    assert license_kind(MIT + "\nOr, at your option, the GNU Lesser General Public License") is None
    assert license_kind("All rights reserved.") is None


def test_long_task_accepts_permissive_snapshot_and_checks_the_change(tmp_path: Path) -> None:
    files = project_files()
    task, outcome = build(files, {"LICENSE": MIT})
    assert outcome == "accepted" and task is not None
    project = record(task["project"])
    assert project["licenses"] == {"LICENSE": "mit"}
    assert sequence(project["context_files"])[-1] == "app/core.py"
    assert sequence(project["context_files"])[-2] == "app/helpers.py"
    assert SETTINGS.minimum_tokens <= integer(project["initial_tokens"]) <= 4000
    prompt = string(record(sequence(task["messages"])[1])["content"])
    assert "Change request:\nSimplify the total\n\nThe change belongs in app/core.py." in prompt
    assert "@" not in prompt and task["group"] == "commitpack:owner/app"
    assert task["source"] == "session-long-project" and task["role"] == "train"
    fixture = record(task["fixture"])
    assert (
        record(fixture["files"])["app/core.py"] == record(candidate(files)["row"])["old_contents"]
    )
    sandbox = Sandbox(tmp_path / "session", fixture)
    assert sandbox.test()["passed"] is False
    sandbox.write("app/core.py", files["app/core.py"])
    assert sandbox.test()["passed"] is True


def test_long_task_refuses_licence_snapshot_and_context_failures() -> None:
    files = project_files()
    assert build(files, {}) == (None, "no_license_file")
    assert build(files, {"LICENSE": GPL}) == (None, "license_not_permissive")
    assert build(files, {"LICENSE": MIT, "COPYING": GPL}) == (None, "license_not_permissive")
    assert build(files, {"LICENSE": MIT, "LICENSE-BIN": ""}) == (None, "license_not_permissive")
    changed = files | {"app/core.py": files["app/core.py"] + "# later\n"}
    disagrees = long_task(
        candidate(files),
        changed,
        {"LICENSE": MIT},
        SETTINGS,
        count,
        "system",
        1024,
        ProtectedCode([LIBRARY]),
        PROTECTED,
    )
    assert disagrees == (None, "snapshot_disagrees")
    small = long_task(
        candidate(files),
        files,
        {"LICENSE": MIT},
        Settings(3900, 4000, 64 * 1024, 20, 512 * 1024),
        count,
        "system",
        1024,
        ProtectedCode([LIBRARY]),
        PROTECTED,
    )
    assert small == (None, "context_tokens_out_of_band")


def test_long_task_refuses_vendored_or_copied_protected_code() -> None:
    files = project_files()
    vendored = files | {"app/_vendor/tomli/_parser.py": "VALUE = 1\n"}
    assert build(vendored, {"LICENSE": MIT}) == (None, "protected_path")
    renamed = files | {"app/humanize.py": "VALUE = 1\n"}
    assert build(renamed, {"LICENSE": MIT}) == (None, "protected_path")
    copied = files | {"app/parsing.py": LIBRARY + "EXTRA = 1\n"}
    assert build(copied, {"LICENSE": MIT}) == (None, "protected_content")
    reformatted = files | {"app/parsing.py": "\n\n" + LIBRARY.replace("    ", "  ")}
    assert build(reformatted, {"LICENSE": MIT}) == (None, "protected_content")
    unrelated = files | {"app/parsing.py": LIBRARY.replace("parse_value", "read_item")}
    assert build(unrelated, {"LICENSE": MIT})[1] == "accepted"


def test_protected_families_cover_forks_renames_and_commits() -> None:
    protected = protected_projects(
        {
            "projects": [
                {
                    "repository": "hukkin/tomli",
                    "revision": "d" * 40,
                    "first_commits": ["e" * 40 + " 2021-05-23T17:27:13+03:00"],
                }
            ]
        },
        {
            "projects": [
                {
                    "repository": "python-humanize/humanize",
                    "known_family": ["jmoiron/humanize", "python-humanize/humanize"],
                    "raw_source_history_matches": [
                        {"commit": "f" * 40, "repositories": ["someone/humanize-fork"]}
                    ],
                }
            ]
        },
    )
    assert protected.overlap(["Hukkin/Tomli"], []) == "protected_family"
    assert protected.overlap(["other/python-tomli"], []) == "protected_name"
    assert protected.overlap(["someone/humanize-fork"], []) == "protected_family"
    assert protected.overlap(["owner/app"], ["e" * 40]) == "protected_commit"
    assert protected.overlap(["owner/app"], ["f" * 40]) == "protected_commit"
    assert protected.overlap(["owner/app"], ["d" * 40]) == "protected_commit"
    assert protected.overlap(["owner/tomlkit"], ["0" * 40]) is None


def test_snapshot_reads_python_and_root_licences_only(tmp_path: Path) -> None:
    archive = tmp_path / "snapshot.tar.gz"
    members = {
        "repo-abc/LICENSE": MIT,
        "repo-abc/app/core.py": "VALUE = 1\n",
        "repo-abc/app/.hidden/x.py": "VALUE = 2\n",
        "repo-abc/app/LICENSE": GPL,
        "repo-abc/README.md": "text\n",
    }
    with tarfile.open(archive, "w:gz") as bundle:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    files, licenses = snapshot(archive)
    assert files == {"app/core.py": "VALUE = 1\n"}
    assert licenses == {"LICENSE": MIT}


def plan_inputs(tmp_path: Path) -> Record:
    rows: list[Record] = []
    for index, repository in enumerate(
        ["owner/one", "owner/two", "owner/three", "hukkin/tomli", "fork/tomli", "seen/pool"]
        + ["owner/session", "owner/many", "owner/four"]
    ):
        for revision in range(2 if repository == "owner/four" else 1):
            body = "def value():\n    return " + str(index) + "\n" + "# padding\n" * 120
            rows.append(
                {
                    "commit": str(index) * 39 + str(revision),
                    "old_file": "pkg/module.py",
                    "new_file": "pkg/module.py",
                    "old_contents": body,
                    "new_contents": body + "VALUE = 1\n",
                    "message": "Add a module value for callers",
                    "license": "mit",
                    "repos": repository,
                }
            )
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prior = tmp_path / "prior.jsonl"
    prior_row: Record = {"repos": "seen/pool"}
    prior.write_text(json.dumps(prior_row) + "\n")
    sessions = {"tasks": [{"source_key": "6" * 39 + "0:pkg/module.py"}]}
    definition = {
        "projects": [{"repository": "hukkin/tomli", "revision": "d" * 40, "first_commits": []}]
    }
    lineage = {
        "projects": [
            {
                "repository": "hukkin/tomli",
                "known_family": ["hukkin/tomli"],
                "raw_source_history_matches": [],
            }
        ]
    }
    return {
        "schema": 1,
        "role": "train",
        "source": reference(source),
        "prior_source": reference(prior),
        "prior_sessions": [write_json(tmp_path / "sessions.json", sessions)],
        "definition": write_json(tmp_path / "definition.json", definition),
        "lineage": write_json(tmp_path / "lineage.json", lineage),
        "seed": 3,
        "candidates": 4,
    }


def unfrozen(*values: object) -> None:
    return None


def fixed_identity(root: Path) -> str:
    return "code"


def test_plan_is_deterministic_one_per_family_and_excludes_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(long_projects, "freeze_source", unfrozen)
    monkeypatch.setattr(long_projects, "source_identity", fixed_identity)
    raw = plan_inputs(tmp_path)
    specification = tmp_path / "plan.json"
    specification.write_bytes(canonical(raw))
    first = long_projects.plan(specification, tmp_path / "first")
    second = long_projects.plan(specification, tmp_path / "second")
    assert (tmp_path / "first" / "candidates.json").read_bytes() == (
        tmp_path / "second" / "candidates.json"
    ).read_bytes()
    chosen = [
        record(value)
        for value in sequence(read(tmp_path / "first" / "candidates.json")["candidates"])
    ]
    names = [string(value["component"]) for value in chosen]
    assert sorted(names) == ["owner/four", "owner/one", "owner/three", "owner/two"]
    assert first["prior_session_projects"] == 1 and second["count"] == 4
    assert integer(first["examined"]) == 4 + sum(
        integer(value) for value in record(first["refused"]).values()
    )
    raw["candidates"] = 5
    specification.write_bytes(canonical(raw))
    with pytest.raises(ValueError, match="cannot supply"):
        long_projects.plan(specification, tmp_path / "third")


def prepared_spec(tmp_path: Path, tasks: list[Record]) -> Record:
    harness = write_json(tmp_path / "harnesses.json", {"tools": "system"})
    tasks_ref = write_json(tmp_path / "tasks.json", {"schema": 1, "tasks": tasks})
    prepared = write_json(
        tmp_path / "prepared.json",
        {
            "status": "prepared",
            "role": "train",
            "source": "session-long-project",
            "tasks": tasks_ref,
            "harnesses": harness,
            "output_cap": 1024,
            "target_sha256": "f" * 64,
        },
    )
    exclusions = write_json(tmp_path / "exclusions.json", {"identities": ["protected-id"]})
    heldout = write_json(tmp_path / "heldout.json", {"identities": []})
    return {
        "prepared": prepared,
        "harnesses": harness,
        "exclusions": exclusions,
        "heldout_exclusions": heldout,
        "output_cap": 1024,
        "target": {"path": "target.gguf", "sha256": "f" * 64, "bytes": 1},
        "tasks": 2,
        "skip": 1,
    }


def prepared_task(name: str) -> Record:
    return {
        "id": name,
        "source": "session-long-project",
        "source_key": name + "-key",
        "group": "commitpack:" + name,
        "aliases": [],
        "role": "train",
    }


def test_prepared_tasks_slice_and_refuse_protected_sources(tmp_path: Path) -> None:
    tasks = [prepared_task("first"), prepared_task("second"), prepared_task("third")]
    raw = prepared_spec(tmp_path, tasks)
    selected, summary = session_tasks(raw, tmp_path)
    assert [task["id"] for task in selected] == ["second", "third"]
    assert summary["skip"] == 1 and summary["prepared_tasks"] == 3
    raw["tasks"] = 3
    with pytest.raises(ValueError, match="cannot supply"):
        prepared_tasks(raw, tmp_path)
    tasks[2]["aliases"] = ["protected-id"]
    raw = prepared_spec(tmp_path, tasks)
    with pytest.raises(ValueError, match="protected"):
        prepared_tasks(raw, tmp_path)
    raw = prepared_spec(tmp_path, tasks) | {"output_cap": 512}
    with pytest.raises(ValueError, match="different harness, output cap or target"):
        prepared_tasks(raw, tmp_path)
    raw = prepared_spec(tmp_path, tasks) | {
        "target": {"path": "target.gguf", "sha256": "e" * 64, "bytes": 1}
    }
    with pytest.raises(ValueError, match="different harness, output cap or target"):
        prepared_tasks(raw, tmp_path)


def test_sessions_without_prepared_tasks_keep_the_commitpack_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks import sessions

    seen: list[Record] = []

    def builder(raw: Record, root: Path) -> tuple[list[Record], Record]:
        seen.append(raw)
        return [], {}

    monkeypatch.setattr(sessions, "build_tasks", builder)
    assert sessions.session_tasks({"tasks": 1}, Path(".")) == ([], {})
    assert seen == [{"tasks": 1}]
    target = "def total(values):\n    return sum(values)\n" * 30
    rows: list[Record] = [
        {
            "commit": "b" * 40,
            "old_file": "pkg/util.py",
            "new_file": "pkg/util.py",
            "old_contents": "",
            "new_contents": "VALUE = 1\n",
            "message": "Add util",
            "license": "mit",
            "repos": "owner/project,fork/project",
        },
        {
            "commit": "a" * 40,
            "old_file": "pkg/core.py",
            "new_file": "pkg/core.py",
            "old_contents": target,
            "new_contents": target + "\n",
            "message": "Handle empty input\n\nSigned-off-by: Someone <someone@example.com>",
            "license": "mit",
            "repos": "fork/project",
        },
    ]
    task = project_task("fork/project", components(rows)["fork/project"], "system", 1024)
    assert identity(task) == "5db6ee33b53a840a2e20f070527134938ce8cdb632324fc1d791d5eaca23ccd1"

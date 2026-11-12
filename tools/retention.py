import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import cast

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import artifact, read, reference


def inspect(plan: Path) -> tuple[Record, list[Artifact], list[Artifact]]:
    raw = read(plan, 512 * 1024 * 1024)
    if (
        set(raw)
        != {
            "schema",
            "owner",
            "authorization",
            "result",
            "limitations",
            "root",
            "protected",
            "retained",
            "remove",
        }
        or type(raw["schema"]) is not int
        or raw["schema"] != 1
    ):
        raise ValueError("unsupported retirement plan")
    for key in ("owner", "authorization", "result", "limitations"):
        string(raw[key])
    root = Path(string(raw["root"]))
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
        raise ValueError("retirement root must be an absolute physical directory")
    if root == Path(root.anchor) or ".git" in root.parts or not root.is_dir():
        raise ValueError("invalid retirement root")
    protected = [Path(string(v)).resolve() for v in sequence(raw["protected"])]
    kept = [artifact(v, plan.parent, resolve=False) for v in sequence(raw["retained"])]
    removed = [artifact(v, plan.parent, resolve=False) for v in sequence(raw["remove"])]
    if not kept or not removed or len({a.path for a in removed}) != len(removed):
        raise ValueError("retained evidence and unique exact removal paths required")
    for item in removed:
        path = item.path
        if not path.is_absolute() or not path.is_relative_to(root) or path == root:
            raise ValueError("removal escaped retirement root")
        if any(part == ".git" for part in path.relative_to(root).parts):
            raise ValueError("Git storage requires separate preservation")
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("symlink retirement paths refused")
        if any(path == p or path.is_relative_to(p) for p in protected):
            raise ValueError("protected artifact cannot be retired")
        if any(k.path == path or k.path.is_relative_to(path) for k in kept):
            raise ValueError("retained evidence overlaps removal")
    for item in kept:
        if item.path.resolve().is_relative_to(root):
            raise ValueError("retained evidence must be independent of retired root")
        item.verify()
    return raw, kept, removed


def active_clients(root: Path) -> list[int]:
    found: list[int] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            paths = [process / "cwd", process / "exe", *list((process / "fd").iterdir())]
            if any(p.resolve().is_relative_to(root) for p in paths):
                found.append(int(process.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return found


def execute(plan: Path, receipt: Path, apply: str | None = None) -> Record:
    plan_pin = Artifact.capture(plan)
    raw, kept, removed = inspect(plan)
    root = Path(string(raw["root"]))
    if receipt.is_relative_to(root) or plan.is_relative_to(root):
        raise ValueError("plan and receipt must survive retired root")
    if apply is not None and apply != plan_pin.sha256:
        raise ValueError("apply requires the inspected plan SHA-256")
    if any(a.path.exists() and not os.access(a.path.parent, os.W_OK) for a in removed):
        raise ValueError("retirement parent is read-only; unlock only owned retired directories")
    clients = active_clients(root)
    if clients:
        raise ValueError("retirement root has active process dependencies: " + str(clients))
    result: Record = {
        "schema": 1,
        "plan": reference(plan_pin),
        "files": len(removed),
        "bytes": sum(a.size for a in removed),
        "retained": [reference(k) for k in kept],
        "phase": "verified" if apply is None else "complete",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix(".lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous: dict[str, Record] = {}
        if receipt.exists():
            for line in receipt.read_text().splitlines():
                row = record(cast(object, json.loads(line)))
                if row.get("plan_sha256") != plan_pin.sha256:
                    raise ValueError("receipt belongs to another retirement plan")
                if row.get("phase") == "retirement-intent":
                    previous[string(row["path"])] = row
        identities: dict[tuple[int, int], list[Artifact]] = {}
        for item in removed:
            if not item.path.exists():
                prior = previous.get(str(item.path))
                if prior is None or prior.get("sha256") != item.sha256:
                    raise ValueError("missing artifact without prior retirement intent")
                continue
            status = item.path.lstat()
            if not item.path.is_file() or status.st_uid != os.getuid():
                raise ValueError("retirement refuses foreign or nonregular file")
            identities.setdefault((status.st_dev, status.st_ino), []).append(item)
            item.verify()
        for items in identities.values():
            if items[0].path.stat().st_nlink != len(items):
                raise ValueError("retirement refuses hardlinks outside its exact plan")
        if apply is not None:
            plan_pin.verify()
            for item in kept:
                item.verify()
            with receipt.open("a") as output:
                for item in removed:
                    if not item.path.exists():
                        continue
                    item.verify()
                    entry: Record = {
                        **reference(item),
                        "plan_sha256": plan_pin.sha256,
                        "phase": "retirement-intent",
                        "source_result": raw["result"],
                    }
                    output.write(json.dumps(entry, sort_keys=True) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    item.path.unlink()
                terminal: Record = {
                    "path": "",
                    "plan_sha256": plan_pin.sha256,
                    "phase": "complete",
                    "files": integer(result["files"]),
                }
                output.write(json.dumps(terminal) + "\n")
                output.flush()
                os.fsync(output.fileno())
    return result


class Arguments(argparse.Namespace):
    plan: Path
    receipt: Path
    apply: str | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify or execute an authorized artifact retirement"
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--apply", metavar="PLAN_SHA256")
    args = parser.parse_args(namespace=Arguments())
    try:
        print(
            json.dumps(
                execute(args.plan.resolve(), args.receipt.resolve(), args.apply), sort_keys=True
            )
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("RETIREMENT_REFUSED: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

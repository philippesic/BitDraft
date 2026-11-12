from pathlib import Path
from uuid import uuid4


def project_root() -> Path:
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / ".git").exists() and (parent / "AGENTS.md").is_file():
            return parent
    return source.parents[1]


def relocated(path: Path, root: Path | None = None) -> Path:
    checkout = project_root() if root is None else root
    prefixes = (
        (
            "/home/philip/binary-eagle-decoding/runs/he4broad-20261110-01",
            "artifacts/datasets/retained-he4broad-original",
        ),
        (
            "/mnt/c/binary-eagle-artifacts/train-bank-v8-fp16-01/captures",
            "artifacts/datasets/train-bank-v8-fp16-01/windows-captures",
        ),
        (
            "/mnt/c/binary-eagle-artifacts/train-bank-v8-fp16-01",
            "artifacts/archive/windows-capture-records/train-bank-v8-fp16-01",
        ),
        (
            "/mnt/c/binary-eagle-component-20261105-01",
            "artifacts/archive/windows-legacy/binary-eagle-component-20261105-01",
        ),
        (
            "/mnt/c/binary-eagle-dflash2-smoke-20261106-01",
            "artifacts/archive/windows-legacy/binary-eagle-dflash2-smoke-20261106-01",
        ),
        (
            "/home/philip/.local/share/BitDraft/legacy-checkout-pending-delete",
            "artifacts/archive/legacy-checkout-pending-delete",
        ),
        ("/home/philip/binary-eagle-artifacts", "artifacts"),
        ("/home/philip/binary-eagle-evidence", "artifacts/archive"),
        ("/home/philip/binary-eagle-decoding-next", "."),
        ("/home/philip/BitDraft", "."),
        ("/home/philip/binary-eagle-decoding", "artifacts/archive/legacy-checkout-pending-delete"),
    )
    if not path.is_relative_to(checkout):
        for legacy, destination in prefixes:
            prefix = Path(legacy)
            if path.is_relative_to(prefix):
                path = checkout / destination / path.relative_to(prefix)
                break
    for previous, destination in (
        ("artifacts/precision-screen", "artifacts/qualification/precision-screen"),
        ("artifacts/storage-audit-01", "artifacts/receipts/storage-inventory-01"),
    ):
        prefix = checkout / previous
        if path.is_relative_to(prefix):
            return checkout / destination / path.relative_to(prefix)
    return path


def output_path(path: Path | None, category: str, name: str, root: Path | None = None) -> Path:
    checkout = project_root() if root is None else root.resolve()
    base = checkout / "artifacts"
    if not base.resolve().is_relative_to(checkout.resolve()):
        raise ValueError("project outputs must stay inside " + str(checkout))
    if Path(category).name != category or not category:
        raise ValueError("one artifact category required")
    if path is None:
        selected = base / category / (Path(name).stem + "-" + uuid4().hex[:12])
    elif path.is_absolute():
        selected = relocated(path, checkout)
    elif path.parts and path.parts[0] == "artifacts":
        selected = checkout / path
    else:
        selected = base / category / path
    selected = selected.resolve()
    if not selected.is_relative_to(base.resolve()) or selected == base.resolve():
        raise ValueError("project outputs must stay inside " + str(base))
    return selected

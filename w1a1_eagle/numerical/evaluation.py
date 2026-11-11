from pathlib import Path
from typing import cast

import torch

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Supervision
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.captures import Capture, CaptureSpec, load_capture
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import artifact, canonical, read


def evaluate(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw)
        != {
            "schema",
            "source",
            "vocabulary",
            "corpus",
            "states",
            "windows_per_task",
            "memory_bytes",
            "training_corpora",
            "exclusions",
        }
        or raw["schema"] != "development-loss-v1"
    ):
        raise ValueError("unsupported development evaluation specification")
    root = specification.resolve().parent
    corpus_ref = artifact(raw["corpus"], root)
    corpus_ref.verify()
    corpus = read(corpus_ref.path, 64 * 1024 * 1024)
    rows = [record(value) for value in sequence(corpus["records"])]
    if not rows or any(row["role"] != "dev" for row in rows):
        raise ValueError("loss evaluation requires a separate DEV-only corpus")
    exclusions = artifact(raw["exclusions"], root)
    exclusions.verify()
    protected = {"heldout24", *(string(v) for v in sequence(read(exclusions.path)["identities"]))}
    training_names: set[str] = set()
    for value in sequence(raw["training_corpora"]):
        ref = artifact(value, root)
        ref.verify()
        training = read(ref.path, 64 * 1024 * 1024)
        if training["target_sha256"] != corpus["target_sha256"]:
            raise ValueError("training/development target mismatch")
        for item in sequence(training["records"]):
            training_names.update(names(record(item)))
    seen: set[str] = set()
    for row in rows:
        current = names(row)
        if current & (training_names | protected):
            raise ValueError("protected or training identity in development evaluation")
        identity = string(row["id"])
        if identity in seen:
            raise ValueError("duplicate development task")
        seen.add(identity)
    count = integer(raw["windows_per_task"])
    memory = integer(raw["memory_bytes"])
    if not 1 <= count <= 64 or memory < 1 or output.exists():
        raise ValueError("bounded windows, memory and a new output directory required")
    source, vocabulary = artifact(raw["source"], root), artifact(raw["vocabulary"], root)
    source.verify()
    vocabulary.verify()
    states = {name: artifact(value, root) for name, value in record(raw["states"]).items()}
    if not states:
        raise ValueError("at least one candidate state required")
    for ref in states.values():
        ref.verify()
    with device_lease("cuda"):
        store = Store(output)
        code_root = Path(__file__).resolve().parents[2]
        code = source_identity(code_root)
        freeze_source(code_root, store.root / "sources", code, 0)
        store.atomic(store.root / "specification.json", canonical(raw))
        torch.set_num_threads(2)
        model = load_he4(source, vocabulary, memory).cuda()
        measurements: list[Record] = []
        for name, ref in states.items():
            parent = Store(ref.path.parent).describe(ref)
            worker = TensorWorker(model, NoSamples(), parent.config)
            worker.restore(parent, memory, "continue")
            model.eval()
            with torch.no_grad():
                for row in rows:
                    capture = load_capture(
                        CaptureSpec(
                            string(row["id"]),
                            artifact(row["metadata"], corpus_ref.path.parent),
                            tuple(
                                artifact(v, corpus_ref.path.parent) for v in sequence(row["files"])
                            ),
                        ),
                        memory,
                    )
                    windows = capture.windows
                    if not windows:
                        raise ValueError("development task has no supervision")
                    indices = sorted(
                        {min(len(windows) - 1, i * len(windows) // count) for i in range(count)}
                    )
                    values: list[float] = []
                    for index in indices:
                        window = windows[index]
                        logits = model.forward(capture.features, capture.tokens, window.anchor)[
                            1 : 1 + len(window.labels)
                        ]
                        labels = torch.tensor(
                            window.labels, device=logits.device, dtype=torch.int64
                        )
                        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                        weights = torch.exp(
                            -torch.arange(len(window.labels), device=logits.device) / 7
                        )
                        value = (loss * weights).sum() / weights.sum()
                        if not bool(torch.isfinite(value)):
                            raise ValueError("nonfinite development loss")
                        values.append(float(value))
                    measurements.append(
                        {
                            "arm": name,
                            "id": row["id"],
                            "group": row["group"],
                            "category": row["topic"],
                            "anchors": [windows[i].anchor for i in indices],
                            "losses": values,
                            "mean_loss": sum(values) / len(values),
                        }
                    )
                    store.atomic(store.root / "progress.json", canonical({"rows": measurements}))
            del worker
        if source_identity(code_root) != code:
            raise ValueError("evaluation source changed during execution")
        result: Record = {
            "status": "complete",
            "rows": measurements,
            "code_sha256": code,
            "target_sha256": corpus["target_sha256"],
            "summary": {
                name: summarize([row for row in measurements if row["arm"] == name])
                for name in states
            },
            "scope": "task-macro weighted hard-label development loss; native acceptance is evaluated separately",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result


def names(row: Record) -> set[str]:
    return {string(row["id"]), string(row["group"]), *(string(v) for v in sequence(row["aliases"]))}


def summarize(rows: list[Record]) -> Record:
    categories = sorted({string(row["category"]) for row in rows})
    return {
        "tasks": len(rows),
        "mean_loss": sum(cast(float, row["mean_loss"]) for row in rows) / len(rows),
        "by_category": {
            category: sum(
                cast(float, row["mean_loss"]) for row in rows if row["category"] == category
            )
            / sum(row["category"] == category for row in rows)
            for category in categories
        },
    }


class NoSamples:
    def sample(self, cursor: int) -> tuple[Capture, Supervision]:
        raise ValueError("evaluation cannot request a training sample")

import torch

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact, CaptureSpec
from w1a1_eagle.model import HE4
from w1a1_eagle.numerical.captures import CaptureCache
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.records import artifact, read
from w1a1_eagle.training import MAXIMUM_SLOTS


class Diagnostics:
    def __init__(self, specification: Artifact, training: Corpus, byte_budget: int) -> None:
        specification.verify()
        raw = read(specification.path)
        if (
            set(raw) != {"schema", "corpus", "exclusions", "windows"}
            or raw["schema"] != "training-diagnostics-v1"
        ):
            raise ValueError("unsupported training diagnostics")
        root = specification.path.parent
        source = artifact(raw["corpus"], root)
        source.verify()
        corpus = read(source.path, 64 * 1024 * 1024)
        if corpus["target_sha256"] != training.target:
            raise ValueError("diagnostic target differs from training")
        exclusions = artifact(raw["exclusions"], root)
        exclusions.verify()
        protected = {
            "heldout24",
            *(string(v) for v in sequence(read(exclusions.path)["identities"])),
        }
        names = {
            name
            for row in training.rows
            for name in (row.prompt.identity, row.prompt.group, *row.prompt.aliases)
        }
        specs: dict[str, CaptureSpec] = {}
        self.categories: dict[str, str] = {}
        for value in sequence(corpus["records"]):
            row = record(value)
            if row["role"] != "dev":
                raise ValueError("diagnostics require a DEV-only corpus")
            current = {
                string(row["id"]),
                string(row["group"]),
                *(string(v) for v in sequence(row["aliases"])),
            }
            if current & (names | protected):
                raise ValueError("training or protected identity in diagnostic DEV")
            name = string(row["id"])
            if name in specs:
                raise ValueError("duplicate diagnostic task")
            specs[name] = CaptureSpec(
                name,
                artifact(row["metadata"], source.path.parent),
                tuple(artifact(v, source.path.parent) for v in sequence(row["files"])),
            )
            specs[name].metadata.verify()
            if (
                read(specs[name].metadata.path).get("target_sha256", training.target)
                != training.target
            ):
                raise ValueError("individual DEV capture target differs from training")
            self.categories[name] = string(row.get("topic") or "unclassified")
        windows = [
            (string(record(v)["id"]), integer(record(v)["anchor"]))
            for v in sequence(raw["windows"])
        ]
        if (
            len(windows) != 120
            or len(set(windows)) != 120
            or any(name not in specs or anchor < 0 for name, anchor in windows)
        ):
            raise ValueError("diagnostics require 120 distinct fixed DEV windows")
        self.windows = tuple(sorted(windows))
        self.specs = specs
        self.cache = CaptureCache(byte_budget)

    def measure(self, model: HE4, slots: int = MAXIMUM_SLOTS) -> Record:
        size = model.block(slots)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if model.vocabulary.device.type == "cuda" else []
        losses: list[float] = []
        matched = labels_count = 0
        positions = [[0, 0, 0] for _ in range(MAXIMUM_SLOTS - 1)]
        categories: dict[str, list[tuple[float, int, int]]] = {}
        with torch.no_grad():
            for name, anchor in self.windows:
                capture = self.cache.get(self.specs[name])
                selected = [window for window in capture.windows if window.anchor == anchor]
                if len(selected) != 1:
                    raise ValueError("diagnostic anchor is not eligible in authenticated capture")
                window = selected[0]
                length = min(len(window.labels), size - 1)
                logits = model.forward(capture.features, capture.tokens, anchor, size)[
                    1 : 1 + length
                ]
                labels = torch.tensor(
                    window.labels[:length], device=logits.device, dtype=torch.int64
                )
                loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                weights = torch.exp(-torch.arange(length, device=logits.device) / 7)
                value = (loss * weights).sum() / weights.sum()
                if not bool(torch.isfinite(value)):
                    raise ValueError("nonfinite diagnostic loss")
                losses.append(float(value))
                hits = logits.argmax(dim=-1) == labels
                correct = int(hits.sum())
                survived = 1
                for position in range(length):
                    hit = int(hits[position])
                    survived *= hit
                    positions[position][0] += 1
                    positions[position][1] += hit
                    positions[position][2] += survived
                matched += correct
                labels_count += length
                categories.setdefault(self.categories[name], []).append(
                    (float(value), correct, length)
                )
        if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
            not torch.equal(a, b)
            for a, b in zip(
                cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [], strict=True
            )
        ):
            raise ValueError("diagnostics changed training RNG")
        return {
            "windows": len(losses),
            "labels": labels_count,
            "mean_loss": sum(losses) / len(losses),
            "top1": matched / labels_count,
            "by_category": {
                name: {
                    "windows": len(values),
                    "mean_loss": sum(value[0] for value in values) / len(values),
                    "labels": sum(value[2] for value in values),
                    "top1": sum(value[1] for value in values) / sum(value[2] for value in values),
                }
                for name, values in categories.items()
            },
            "slots": size,
            "positions": {
                "windows": [value[0] for value in positions],
                "hits": [value[1] for value in positions],
                "survivors": [value[2] for value in positions],
                "top1": [value[1] / value[0] if value[0] else None for value in positions],
                "prefix_survival": [
                    value[2] / value[0] if value[0] else None for value in positions
                ],
                "scope": "draft positions 1-15 measured in one block of this many slots; serving builds depth+1 slots per round",
            },
            "scope": "teacher-forced hard-label diagnostic; not native acceptance",
        }

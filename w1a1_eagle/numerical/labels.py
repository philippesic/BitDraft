from dataclasses import dataclass, field

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact, capture_supervision
from w1a1_eagle.drafting import CONTEXT_BANDS, context_band, thinking_mode
from w1a1_eagle.records import artifact, read

LONG_DOCUMENT_TOKENS = 2048


@dataclass
class Cell:
    conversations: int = 0
    labels: int = 0
    sources: set[str] = field(default_factory=set)

    def record(self) -> Record:
        return {
            "conversations": self.conversations,
            "labels": self.labels,
            "sources": len(self.sources),
        }


def scenario(topic: str, session: bool, thinking: str, prompt: int) -> str:
    if session:
        return "agentic-sessions"
    if thinking == "on":
        return "thinking-on"
    if topic == "documents" and prompt >= LONG_DOCUMENT_TOKENS:
        return "long-documents"
    return "ordinary-" + topic


class LabelReport:
    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, str, str], Cell] = {}

    def add(
        self, group: str, topic: str, metadata: Record, anchors: tuple[int, ...]
    ) -> tuple[str, str]:
        tokens = tuple(integer(value) for value in sequence(metadata["input_tokens"]))
        spans = [sequence(value) for value in sequence(metadata.get("supervised_spans", []))]
        session = bool(spans)
        prompt = integer(spans[0][0]) + 1 if session else integer(metadata["prompt_tokens"])
        thinking = thinking_mode(tokens, prompt)
        name = scenario(topic, session, thinking, prompt)
        longest = context_band(max(anchors) + 1 if anchors else prompt)
        conversation = self.cells.setdefault((name, topic, thinking, longest), Cell())
        conversation.conversations += 1
        conversation.sources.add(group)
        for anchor in anchors:
            cell = self.cells.setdefault((name, topic, thinking, context_band(anchor)), Cell())
            cell.labels += 1
            cell.sources.add(group)
        return name, thinking

    def collapse(self, keys: tuple[int, ...]) -> dict[str, Record]:
        merged: dict[str, Cell] = {}
        for key, cell in self.cells.items():
            name = "|".join(key[index] for index in keys)
            target = merged.setdefault(name, Cell())
            target.conversations += cell.conversations
            target.labels += cell.labels
            target.sources |= cell.sources
        return {name: merged[name].record() for name in sorted(merged)}

    def record(self) -> Record:
        total = Cell()
        for cell in self.cells.values():
            total.conversations += cell.conversations
            total.labels += cell.labels
            total.sources |= cell.sources
        return {
            "total": total.record(),
            "by_scenario": self.collapse((0,)),
            "by_category": self.collapse((1,)),
            "by_thinking": self.collapse((2,)),
            "by_context_band": self.collapse((3,)),
            "by_scenario_and_band": self.collapse((0, 3)),
            "by_scenario_category_thinking_band": self.collapse((0, 1, 2, 3)),
            "bands": list(CONTEXT_BANDS),
            "definitions": {
                "conversations": "capture records, counted once in the band of their longest supervised context",
                "labels": "supervision windows (one per anchor), counted in the band of their anchor's context length",
                "sources": "distinct source groups contributing to the cell",
                "thinking": "on when the last <think> before the first generated position is not closed; off when closed; absent without a thinking block",
                "scenario": "agentic-sessions for multi-span session records, thinking-on, long-documents for documents with a prompt of at least 2,048 tokens, otherwise ordinary-<category>",
            },
        }


def label_report(mixture: Artifact) -> Record:
    mixture.verify()
    raw = read(mixture.path)
    if raw.get("schema") != "corpus-mixture-v1":
        raise ValueError("label report requires a corpus-mixture-v1 manifest")
    root = mixture.path.parent
    report = LabelReport()
    components: list[Record] = []
    for value in sequence(raw["components"]):
        component = record(value)
        if component["kind"] != "corpus-v1":
            raise ValueError("label report reads direct corpus-v1 components only")
        source = artifact(component["source"], root)
        source.verify()
        corpus = read(source.path, 64 * 1024 * 1024)
        if corpus["target_sha256"] != raw["target_sha256"]:
            raise ValueError("mixture component target differs")
        windows = 0
        rows = [record(item) for item in sequence(corpus["records"])]
        for row in rows:
            if row["role"] != "train":
                raise ValueError("label report reads TRAIN records only")
            reference = artifact(row["metadata"], source.path.parent)
            reference.verify()
            metadata = read(reference.path, 64 * 1024 * 1024)
            anchors = tuple(
                window.anchor for window in capture_supervision(string(row["id"]), metadata)
            )
            windows += len(anchors)
            report.add(
                string(row["group"]), string(row.get("topic") or "unclassified"), metadata, anchors
            )
        components.append({"source": source.sha256, "conversations": len(rows), "labels": windows})
    return {
        "mixture": mixture.sha256,
        "components": components,
        **report.record(),
        "scope": "metadata only; feature arrays are not read",
    }


@dataclass(frozen=True)
class LabelFrequencies:
    mixture: str
    ranked: tuple[int, ...]
    counts: tuple[int, ...]
    labels: int

    def coverage(self, size: int) -> float | None:
        return sum(self.counts[:size]) / self.labels if self.labels else None


MIXTURE_COMPONENTS = {"corpus-mixture-v1": "corpus-v1", "fused-mixture-v1": "fused-corpus-v1"}


def label_frequencies(mixture: Artifact) -> LabelFrequencies:
    mixture.verify()
    raw = read(mixture.path)
    schema = raw.get("schema")
    if schema not in MIXTURE_COMPONENTS:
        raise ValueError(
            "label frequencies require a corpus-mixture-v1 or fused-mixture-v1 manifest"
        )
    kind = MIXTURE_COMPONENTS[string(schema)]
    root = mixture.path.parent
    counts: dict[int, int] = {}
    for value in sequence(raw["components"]):
        component = record(value)
        if component["kind"] != kind:
            raise ValueError("label frequencies read direct " + kind + " components only")
        source = artifact(component["source"], root)
        source.verify()
        corpus = read(source.path, 64 * 1024 * 1024)
        if corpus["target_sha256"] != raw["target_sha256"]:
            raise ValueError("mixture component target differs")
        if kind == "fused-corpus-v1" and corpus.get("schema") != kind:
            raise ValueError("fused mixture components must be fused-corpus-v1 manifests")
        for item in sequence(corpus["records"]):
            row = record(item)
            if row["role"] != "train":
                raise ValueError("label frequencies read TRAIN records only")
            reference = artifact(row["metadata"], source.path.parent)
            reference.verify()
            metadata = read(reference.path, 64 * 1024 * 1024)
            positions: dict[int, int] = {}
            for window in capture_supervision(string(row["id"]), metadata):
                for offset, token in enumerate(window.labels):
                    positions[window.anchor + 1 + offset] = token
            for token in positions.values():
                counts[token] = counts.get(token, 0) + 1
    ranked = [token for _, token in sorted((-count, token) for token, count in counts.items())]
    return LabelFrequencies(
        mixture.sha256,
        tuple(ranked),
        tuple(counts[token] for token in ranked),
        sum(counts.values()),
    )

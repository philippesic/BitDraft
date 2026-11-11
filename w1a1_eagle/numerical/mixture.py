from w1a1_eagle.contracts import record, sequence, string
from w1a1_eagle.data import Artifact, CaptureRow, Dataset
from w1a1_eagle.index import IndexReader
from w1a1_eagle.intent import Intent
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.records import artifact, identity, read


def mixture(manifest: Artifact, seed: int, byte_budget: int, hidden: int) -> Dataset:
    manifest.verify()
    raw = read(manifest.path)
    if (
        set(raw) != {"schema", "target_sha256", "components"}
        or raw["schema"] != "corpus-mixture-v1"
    ):
        raise ValueError("unsupported corpus mixture")
    components = sequence(raw["components"])
    if not 2 <= len(components) <= 8:
        raise ValueError("mixture requires two to eight explicit components")
    target = string(raw["target_sha256"])
    rows: list[CaptureRow] = []
    protected: set[str] = {"heldout24"}
    prior_names: set[str] = set()
    prefixes: set[str] = set()
    metadata_bytes = manifest.size * 16
    for value in components:
        component = record(value)
        if set(component) != {"kind", "source"}:
            raise ValueError("unsupported corpus mixture component")
        ref = artifact(component["source"], manifest.path.parent)
        ref.verify()
        if component["kind"] == "index-intent":
            intent = Intent.load(ref.path)
            if intent.index_spec is None or "corpus_format" in intent.raw:
                raise ValueError("index component requires a direct retained-index intent")
            dataset = IndexReader(intent.corpus, intent.index_spec, byte_budget).load()
            selected = dataset.rows
            protected.update(dataset.protected)
            metadata_bytes += dataset.metadata_bytes
            observed_target = dataset.target
        elif component["kind"] == "corpus-v1":
            corpus_raw = read(ref.path, 64 * 1024 * 1024)
            if any(record(row)["role"] != "train" for row in sequence(corpus_raw["records"])):
                raise ValueError("mixture components must contain TRAIN records only")
            corpus = Corpus(ref, seed, byte_budget, hidden)
            selected = corpus.rows
            protected.update(string(v) for v in sequence(corpus_raw["protected"]))
            metadata_bytes += ref.size * 16 + corpus.count * 40
            observed_target = corpus.target
        else:
            raise ValueError("unknown mixture component kind")
        if observed_target != target:
            raise ValueError("mixture target identities differ")
        current_names: set[str] = set()
        current_prefixes: set[str] = set()
        for row in selected:
            names = {row.prompt.identity, row.prompt.group, *row.prompt.aliases}
            metadata = read(row.capture.metadata.path)
            prompt_count = metadata["prompt_tokens"]
            if type(prompt_count) is not int:
                raise ValueError("integer prompt length required")
            prefix = identity({"tokens": sequence(metadata["input_tokens"])[:prompt_count]})
            if names & prior_names or prefix in prefixes:
                raise ValueError(
                    "duplicate source identity or tokenized prompt across mixture components"
                )
            current_names.update(names)
            current_prefixes.add(prefix)
        prior_names.update(current_names)
        prefixes.update(current_prefixes)
        rows.extend(selected)
    return Dataset(manifest, target, tuple(rows), frozenset(protected), metadata_bytes)

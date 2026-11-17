from pathlib import Path

import pytest

from benchmarks.comparison import Comparison
from benchmarks.native import file_reference
from benchmarks.runtime_edits import SOURCE_EDITS
from benchmarks.runtime_policy import arguments
from benchmarks.serving import server_arguments
from tests.test_arm_depths import LEGACY_PROTOCOL
from tests.test_references import evidence
from w1a1_eagle.contracts import sequence
from w1a1_eagle.records import canonical, identity, read

EDIT = SOURCE_EDITS["vocabulary_shortlist_v1"]


def declare(path: Path, shortlists: object, **extra: object) -> Comparison:
    raw = read(path)
    raw.pop("shortlists", None)
    if shortlists is not None:
        raw["shortlists"] = shortlists
    raw.update(extra)
    path.write_bytes(canonical(raw))
    return Comparison.load(path)


def flag(argv: list[str]) -> str | None:
    return (
        argv[argv.index("--spec-draft-shortlist") + 1] if "--spec-draft-shortlist" in argv else None
    )


def kept_in_order(old: str, new: str) -> bool:
    remaining = iter(new.splitlines())
    return all(any(line == candidate for candidate in remaining) for line in old.splitlines())


def test_shortlist_edit_only_inserts_code_except_the_guarded_scoring_branch() -> None:
    changed = [(path, old, new) for path, old, new in EDIT if not kept_in_order(old, new)]
    assert [path for path, _, _ in changed] == ["src/models/dflash.cpp"]
    _, old, new = changed[0]
    crop, scoring = old.splitlines()
    assert new.startswith(crop + "\n")
    guarded, default = new.split("    } else {\n")
    assert default == "    " + scoring + "\n    }\n"
    assert "if (shortlist && shortlist->source == output && !output_s" in guarded
    assert "const llama_vocab_shortlist * shortlist = cparams.vocab_shortlist;" in guarded


def test_shortlist_is_attached_only_from_drafter_metadata_or_the_sidecar_flag() -> None:
    by_path: dict[str, str] = {}
    for path, _, new in EDIT:
        by_path[path] = by_path.get(path, "") + new
    assert (
        "const struct llama_vocab_shortlist * vocab_shortlist = nullptr;"
        in by_path["src/llama-cparams.h"]
    )
    context = by_path["src/llama-context.cpp"]
    assert context.count("cparams.vocab_shortlist = vocab_shortlist.get();") == 1
    speculative = by_path["common/speculative.cpp"]
    assert speculative.count("llama_vocab_shortlist_init(") == 1
    assert "if (n_shortlist > 0) {" in speculative
    assert speculative.count("if (shortlist_active) {") == 2
    assert "bool shortlist_active = false;" in speculative
    assert (
        'gguf_find_key(ml.metadata, "dflash.vocab_shortlist")' in by_path["src/models/dflash.cpp"]
    )
    assert '{"--spec-draft-shortlist"}' in by_path["common/arg.cpp"]
    assert "std::string shortlist;" in by_path["common/common.h"]


def test_shortlist_edit_matches_its_anchors_once_on_a_synthetic_source(tmp_path: Path) -> None:
    import benchmarks.native as native

    texts: dict[str, str] = {}
    for path, old, _ in EDIT:
        texts[path] = texts.get(path, "head\n") + old + "middle\n"
    for path, text in texts.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    native.apply_source_edits(tmp_path, "vocabulary_shortlist_v1")
    for path, _, new in EDIT:
        assert new in (tmp_path / path).read_text()


def test_omitted_shortlist_keeps_the_legacy_protocol_and_server_arguments(tmp_path: Path) -> None:
    spec = Comparison.load(evidence(tmp_path))
    protocol = spec.reference_protocol()
    pinned = {**protocol, "serving": ["source", *sequence(protocol["serving"])[1:]]}
    assert spec.shortlists == () and identity(pinned) == LEGACY_PROTOCOL
    assert "arm_shortlists" not in protocol and "shortlist" not in spec.arm_protocol("baseline")
    assert flag(server_arguments(spec, spec.arms[1], 1)) is None
    assert arguments(Path("s"), Path("t"), Path("d"), 16, 3, 1) == arguments(
        Path("s"), Path("t"), Path("d"), 16, 3, 1, shortlist=None
    )
    with pytest.raises(ValueError, match="needs a drafter"):
        arguments(Path("s"), Path("t"), None, 16, 3, 1, shortlist=Path("x"))


def test_declared_shortlist_binds_identity_and_serves_only_its_arm(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    legacy = Comparison.load(path)
    (tmp_path / "shortlist.i32").write_bytes(bytes(8))
    sidecar = file_reference(tmp_path / "shortlist.i32")
    spec = declare(path, {"baseline": sidecar})
    digest = str(sidecar["sha256"])
    assert spec.reference_protocol() == {
        **legacy.reference_protocol(),
        "arm_shortlists": {"baseline": digest},
    }
    assert spec.arm_protocol("target") == legacy.arm_protocol("target")
    drafted = spec.arm_protocol("baseline")
    assert drafted["shortlist"] == digest
    assert flag([str(v) for v in sequence(drafted["draft_arguments"])]) == "shortlist"
    assert flag(server_arguments(spec, spec.arms[1], 1)) == str(tmp_path / "shortlist.i32")
    assert flag(server_arguments(spec, spec.arms[0], 1)) is None
    spec.verify()
    for invalid in ({"target": sidecar}, {"missing": sidecar}, {}):
        with pytest.raises(ValueError, match="shortlist"):
            declare(path, invalid)
    with pytest.raises(ValueError, match="draft-dflash"):
        declare(path, {"baseline": sidecar}, speculation={"baseline": "draft-dspark"})

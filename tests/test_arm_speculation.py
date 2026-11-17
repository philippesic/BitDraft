from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.comparison import Arm, Comparison
from benchmarks.metrics import verify_generation
from benchmarks.native import file_reference
from benchmarks.runtime_policy import arguments, payload
from benchmarks.serving import server_arguments
from tests.test_arm_depths import LEGACY_PROTOCOL, registered
from tests.test_references import evidence
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.records import artifact, canonical, identity, read


def declare(path: Path, speculation: object) -> Comparison:
    raw = read(path)
    if speculation is None:
        raw.pop("speculation", None)
    else:
        raw["speculation"] = speculation
    path.write_bytes(canonical(raw))
    return Comparison.load(path)


def spec_type(argv: list[str]) -> str:
    return argv[argv.index("--spec-type") + 1]


def answered(mode: str) -> Record:
    return {
        "generation_settings": {
            "seed": 8101,
            "top_k": 0,
            "n_predict": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "min_p": 0.0,
            "ignore_eos": False,
            "grammar": "",
            "repeat_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "dynatemp_range": 0.0,
            "speculative.types": mode,
        }
    }


def test_omitted_speculation_keeps_the_legacy_protocol_and_server_arguments(
    tmp_path: Path,
) -> None:
    spec = Comparison.load(evidence(tmp_path))
    protocol = spec.reference_protocol()
    pinned = {**protocol, "serving": ["source", *sequence(protocol["serving"])[1:]]}
    assert spec.speculation == () and identity(pinned) == LEGACY_PROTOCOL
    assert "arm_speculation" not in protocol
    assert spec.speculation_of("baseline") == "draft-dflash"
    assert spec_type(server_arguments(spec, spec.arms[1], 1)) == "draft-dflash"
    assert spec_type(server_arguments(spec, spec.arms[0], 1)) == "none"
    assert arguments(Path("s"), Path("t"), Path("d"), 16, 3, 1) == arguments(
        Path("s"), Path("t"), Path("d"), 16, 3, 1, speculation="draft-dflash"
    )


def test_declared_speculation_parses_binds_identity_and_serves_each_arm(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    legacy = Comparison.load(path).reference_protocol()
    spec = declare(path, {"baseline": "draft-dspark"})
    assert spec.speculation == (("baseline", "draft-dspark"),)
    assert spec.reference_protocol() == {**legacy, "arm_speculation": {"baseline": "draft-dspark"}}
    assert spec.arm_protocol("target") == legacy
    drafted = [str(v) for v in sequence(spec.arm_protocol("baseline")["draft_arguments"])]
    assert spec_type(drafted) == "draft-dspark"
    assert spec_type(server_arguments(spec, spec.arms[1], 1)) == "draft-dspark"
    assert spec_type(server_arguments(spec, spec.arms[0], 1)) == "none"
    raw = read(path)
    raw["depths"] = {"baseline": 4}
    path.write_bytes(canonical(raw))
    assert Comparison.load(path).reference_protocol() == {
        **legacy,
        "arm_depths": {"baseline": 4},
        "arm_speculation": {"baseline": "draft-dspark"},
    }
    for invalid in (
        {"target": "draft-dspark"},
        {"missing": "draft-dspark"},
        {"baseline": "draft-eagle3"},
        {"baseline": 3},
        {},
    ):
        with pytest.raises(ValueError, match="speculation|string"):
            declare(path, invalid)


def test_generation_check_requires_the_declared_native_mode() -> None:
    query = payload([9], 2, 0.0, 1.0, 0, 0.0, 8101)
    verify_generation(answered("draft-dflash"), query, True)
    verify_generation(answered("none"), query, False)
    verify_generation(answered("draft-dspark"), query, True, "draft-dspark")
    for response, mode in (
        (answered("draft-dspark"), "draft-dflash"),
        (answered("draft-dflash"), "draft-dspark"),
    ):
        with pytest.raises(ValueError, match="speculative mode"):
            verify_generation(response, query, True, mode)


def test_references_bind_the_speculation_their_arm_ran_with(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    target = registered(tmp_path, path, "target")["reference"]
    baseline = registered(tmp_path, path, "baseline")["reference"]
    assert "arm_speculation" not in record(read(artifact(baseline, tmp_path).path)["protocol"])
    declare(path, {"baseline": "draft-dspark"})
    with pytest.raises(ValueError, match="executed server policy"):
        registered(tmp_path, path, "baseline", "other-registry")
    declare(path, None)
    model = artifact(file_reference(tmp_path / "drafter"), tmp_path)
    candidate = replace(
        Comparison.load(path),
        arms=(Arm("dspark", model),),
        references=(
            ("target", artifact(target, tmp_path)),
            ("baseline", artifact(baseline, tmp_path)),
        ),
    )
    for unchanged in ((), (("dspark", "draft-dspark"),)):
        assert set(replace(candidate, speculation=unchanged).cached()) == {"target", "baseline"}
    with pytest.raises(ValueError, match="reference protocol"):
        replace(candidate, speculation=(("baseline", "draft-dspark"),)).cached()

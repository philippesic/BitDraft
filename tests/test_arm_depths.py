from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.comparison import Arm, Comparison
from benchmarks.native import file_reference
from benchmarks.references import register
from benchmarks.runtime_policy import arguments
from benchmarks.serving import server_arguments
from tests.test_references import evidence
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.records import artifact, canonical, identity, read

LEGACY_PROTOCOL = "4139158e13794043edfff59ab5fefa489185b020d25072f436b3b4a6667e405d"


def declare(path: Path, depths: object) -> Comparison:
    raw = read(path)
    if depths is None:
        raw.pop("depths", None)
    else:
        raw["depths"] = depths
    path.write_bytes(canonical(raw))
    return Comparison.load(path)


def executed_at(root: Path, arm: str, depth: int) -> None:
    command = root / ("0-0-" + arm) / "command.json"
    command.write_bytes(
        canonical(
            {
                "argv": arguments(
                    root / "llama-server", root / "target", root / "drafter", 16, depth, 12345
                )
            }
        )
    )


def registered(root: Path, desired: Path, arm: str, registry: str = "registry") -> Record:
    registration = root / (arm + "-registration.json")
    registration.write_bytes(
        canonical(
            {
                "schema": 1,
                "comparison": file_reference(desired),
                "result": file_reference(root / "result.json"),
                "arm": arm,
                "rep": 0,
            }
        )
    )
    return register(registration, root / registry)


def test_omitted_depths_keep_the_legacy_protocol_and_server_arguments(tmp_path: Path) -> None:
    spec = Comparison.load(evidence(tmp_path))
    protocol = spec.reference_protocol()
    pinned = {**protocol, "serving": ["source", *sequence(protocol["serving"])[1:]]}
    assert spec.depths == () and identity(pinned) == LEGACY_PROTOCOL
    assert spec.arm_protocol("baseline") == protocol == spec.arm_protocol("target")
    assert spec.arm_depths() == {"target": None, "baseline": 2}
    argv = server_arguments(spec, spec.arms[1], 1)
    assert argv[argv.index("--spec-draft-n-max") + 1] == "2"


def test_declared_depths_parse_bind_identity_and_serve_each_arm(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    legacy = Comparison.load(path).reference_protocol()
    spec = declare(path, {"baseline": 4})
    assert spec.depths == (("baseline", 4),)
    assert spec.arm_depths() == {"target": None, "baseline": 4}
    assert spec.reference_protocol() == {**legacy, "arm_depths": {"baseline": 4}}
    assert spec.arm_protocol("target") == legacy
    assert spec.arm_protocol("baseline") == replace(spec, depth=4, depths=()).reference_protocol()
    assert sequence(spec.arm_protocol("baseline")["serving"])[-2] == "4"
    argv = server_arguments(spec, spec.arms[1], 1)
    assert argv[argv.index("--spec-draft-n-max") + 1] == "4"
    assert "--spec-draft-n-max" not in server_arguments(spec, spec.arms[0], 1)
    for invalid in ({"target": 3}, {"missing": 3}, {"baseline": 0}, {"baseline": 16}, {}):
        with pytest.raises(ValueError, match="depth"):
            declare(path, invalid)


def test_reference_binds_the_depth_its_arm_ran_at(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    executed_at(tmp_path, "baseline", 4)
    with pytest.raises(ValueError, match="executed server policy"):
        registered(tmp_path, path, "baseline")
    declare(path, {"baseline": 4})
    target = registered(tmp_path, path, "target")["reference"]
    baseline = registered(tmp_path, path, "baseline")["reference"]
    saved = read(artifact(baseline, tmp_path).path)
    assert sequence(record(saved["protocol"])["serving"])[-2] == "4"
    assert "arm_depths" not in record(saved["protocol"])
    legacy = tmp_path / "legacy.json"
    legacy.write_bytes(canonical({k: v for k, v in read(path).items() if k != "depths"}))
    with pytest.raises(ValueError, match="only complete valid matching evidence"):
        registered(tmp_path, legacy, "baseline", "other-registry")
    model = artifact(file_reference(tmp_path / "drafter"), tmp_path)
    candidate = replace(
        Comparison.load(path),
        arms=(Arm("candidate", model),),
        references=(
            ("target", artifact(target, tmp_path)),
            ("baseline", artifact(baseline, tmp_path)),
        ),
    )
    for declared in ((("baseline", 4),), (("baseline", 4), ("candidate", 2))):
        assert set(replace(candidate, depths=declared).cached()) == {"target", "baseline"}
    for wrong in ((), (("baseline", 3),)):
        with pytest.raises(ValueError, match="reference protocol"):
            replace(candidate, depths=wrong).cached()


def test_uniform_references_authenticate_unchanged_and_refuse_other_depths(tmp_path: Path) -> None:
    path = evidence(tmp_path)
    refs = tuple(
        (arm, artifact(registered(tmp_path, path, arm)["reference"], tmp_path))
        for arm in ("target", "baseline")
    )
    model = artifact(file_reference(tmp_path / "drafter"), tmp_path)
    candidate = replace(Comparison.load(path), arms=(Arm("candidate", model),), references=refs)
    assert set(candidate.cached()) == {"target", "baseline"}
    assert set(replace(candidate, depths=(("candidate", 3),)).cached()) == {"target", "baseline"}
    with pytest.raises(ValueError, match="reference protocol"):
        replace(candidate, depths=(("baseline", 3),)).cached()


def test_scenario_report_accepts_mixed_depth_panels_and_refuses_other_runtime(
    tmp_path: Path,
) -> None:
    from benchmarks.scenario_report import report

    uniform = tmp_path / "uniform"
    mixed = tmp_path / "mixed"
    uniform.mkdir()
    mixed.mkdir()
    evidence(uniform)
    for name in ("specification.json", "result.json"):
        (mixed / name).write_bytes((uniform / name).read_bytes())
    declare(mixed / "specification.json", {"baseline": 4})
    specification = tmp_path / "report.json"

    def write(panels: Record) -> None:
        specification.write_bytes(
            canonical(
                {
                    "schema": "scenario-report-v1",
                    "level": 3,
                    "candidate": "baseline",
                    "panels": panels,
                    "confirmation": None,
                }
            )
        )

    write(
        {
            "requests": file_reference(uniform / "result.json"),
            "screen": file_reference(mixed / "result.json"),
        }
    )
    output = report(specification, tmp_path / "output")
    panels = record(output["panels"])
    assert record(panels["requests"])["arm_depths"] == {"target": None, "baseline": 2}
    assert record(panels["screen"])["arm_depths"] == {"target": None, "baseline": 4}
    other = read(mixed / "specification.json") | {"context": 32}
    (mixed / "specification.json").write_bytes(canonical(other))
    with pytest.raises(ValueError, match="must share"):
        report(specification, tmp_path / "refused")

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from w1a1_eagle import storage
from w1a1_eagle.contracts import Record
from w1a1_eagle.data import Artifact
from w1a1_eagle.index import Mount
from w1a1_eagle.records import artifact, read


def test_local_output_defaults_and_escape_refusals(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    first = storage.output_path(None, "datasets", "capture.json", root)
    second = storage.output_path(None, "datasets", "capture.json", root)
    assert first.parent == root / "artifacts/datasets" and first != second
    assert not first.exists()
    assert (
        storage.output_path(Path("capture-1"), "datasets", "ignored", root)
        == root / "artifacts/datasets/capture-1"
    )
    assert (
        storage.output_path(Path("artifacts/models/basis"), "datasets", "ignored", root)
        == root / "artifacts/models/basis"
    )
    with pytest.raises(ValueError, match="inside"):
        storage.output_path(tmp_path / "outside", "datasets", "capture", root)
    with pytest.raises(ValueError, match="inside"):
        storage.output_path(Path("../../outside"), "datasets", "capture", root)
    (root / "artifacts").mkdir()
    (root / "artifacts/escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        storage.output_path(Path("artifacts/escape/new"), "datasets", "capture", root)
    escaped_root = tmp_path / "escaped-checkout"
    escaped_root.mkdir()
    (escaped_root / "artifacts").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        storage.output_path(None, "datasets", "capture", escaped_root)


def test_legacy_artifacts_read_without_old_directories_or_record_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "project_root", lambda: tmp_path)
    root = tmp_path / "artifacts/datasets/example"
    root.mkdir(parents=True)
    payload = root / "features"
    payload.write_bytes(b"verified existing feature bytes")
    old = "/home/philip/binary-eagle-artifacts/datasets/example/features"
    document = {
        "path": old,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "bytes": payload.stat().st_size,
    }
    raw = json.dumps(document).encode()
    metadata = root / "record.json"
    metadata.write_bytes(raw)
    loaded = artifact(
        read(Path("/home/philip/binary-eagle-artifacts/datasets/example/record.json")), tmp_path
    )
    loaded.verify()
    assert loaded.path == payload and metadata.read_bytes() == raw
    assert Artifact.capture(Path(old)).sha256 == loaded.sha256
    mount = Mount(Path("/home/philip/binary-eagle-artifacts/datasets/example"), root)
    assert mount.map(document).path == payload
    document["path"] = old + "/../../outside"
    with pytest.raises(ValueError, match="escaped"):
        mount.map(document)


def test_frozen_source_finds_owning_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("# Test checkout\n")
    frozen = tmp_path / "artifacts/training/executed-sources/frozen/w1a1_eagle/storage.py"
    monkeypatch.setattr(storage, "__file__", str(frozen))
    assert storage.project_root() == tmp_path


def test_public_training_and_capture_refuse_external_outputs_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks import __main__ as benchmark_cli
    from w1a1_eagle import __main__ as training_cli

    external = tmp_path / "outside"
    monkeypatch.setattr(sys, "argv", ["benchmarks", "capture", "unused.json", str(external)])
    assert benchmark_cli.main() == 2
    assert "inside" in capsys.readouterr().out and not external.exists()
    monkeypatch.setattr(
        sys, "argv", ["w1a1_eagle", "run", "unused.json", str(external), "--scope", "fixture-only"]
    )
    assert training_cli.main() == 2
    assert "inside" in capsys.readouterr().out and not external.exists()


def test_public_capture_and_training_use_local_default_and_explicit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from benchmarks import __main__ as benchmark_cli
    from w1a1_eagle import __main__ as training_cli

    monkeypatch.setattr(storage, "project_root", lambda: tmp_path)
    observed: list[Path] = []

    def capture_fixture(spec: Path, output: Path) -> dict[str, str]:
        assert spec.name == "new-capture.json"
        output.mkdir(parents=True)
        (output / "fixture.json").write_text('{"status":"captured"}')
        observed.append(output)
        return {"status": "fixture-only"}

    monkeypatch.setattr(benchmark_cli, "capture", capture_fixture)
    monkeypatch.setattr(sys, "argv", ["benchmarks", "--frozen", "capture", "new-capture.json"])
    assert benchmark_cli.main() == 0
    assert observed[-1].parent == tmp_path / "artifacts/datasets"
    assert (observed[-1] / "fixture.json").is_file()
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmarks", "--frozen", "capture", "new-capture.json", "followup"],
    )
    assert benchmark_cli.main() == 0
    assert observed[-1] == tmp_path / "artifacts/datasets/followup"
    assert "fixture-only" in capsys.readouterr().out

    def execute_fixture(
        intent: Path, output: Path, scope: str, emit: Callable[[Record], None]
    ) -> Record:
        assert intent.name == "training.json" and scope == "fixture-only"
        output.mkdir(parents=True)
        observed.append(output)
        emit({"scope": scope})
        return {"operational": "terminal", "latest": {"outcome": "complete"}}

    monkeypatch.setattr(training_cli, "execute", execute_fixture)
    monkeypatch.setattr(
        sys,
        "argv",
        ["w1a1_eagle", "run", "training.json", "--scope", "fixture-only", "--json"],
    )
    assert training_cli.main() == 0
    assert observed[-1].parent == tmp_path / "artifacts/training"
    assert "fixture-only" in capsys.readouterr().out


def test_cpu_cuda_and_native_caches_stay_in_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from w1a1_eagle import config

    monkeypatch.setattr(config, "project_root", lambda: tmp_path)
    for environment in (
        config.cpu_environment(),
        config.execution_environment("cuda"),
        config.native_environment(tmp_path / "libraries"),
    ):
        for key in (
            "TMPDIR",
            "PYTHONPYCACHEPREFIX",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "TORCH_HOME",
            "TORCH_EXTENSIONS_DIR",
            "PYTORCH_KERNEL_CACHE_PATH",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
        ):
            assert Path(environment[key]).is_relative_to(tmp_path / "artifacts/cache")
    assert config.cpu_environment()["CUDA_VISIBLE_DEVICES"] == ""

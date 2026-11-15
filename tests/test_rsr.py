from pathlib import Path

import pytest

from benchmarks.native import file_reference
from benchmarks.rsr import RSR_COMMIT, install_rsr, integer_identity, rsr_kernel


def test_rsr_device_body_is_copied_without_host_wrapper() -> None:
    body = "namespace {\ntemplate<int T> void rsr_gemv_v5_9_kernel() {}\n}  // namespace"
    source = "untrusted host prefix\n" + body + "\ntorch::Tensor wrapper() {}"
    assert rsr_kernel(source) == "namespace rsr_upstream {\n" + body + "\n}\n"
    with pytest.raises(ValueError, match="boundaries"):
        rsr_kernel(source + body)
    with pytest.raises(ValueError, match="missing"):
        rsr_kernel(body.replace("rsr_gemv_v5_9_kernel", "different"))


def test_integer_failure_is_not_hidden_by_float_success() -> None:
    assert integer_identity("sample,0,5,1\nintegers,0\n") is False
    assert integer_identity("sample,0,5,1\nintegers,1\n") is True
    with pytest.raises(ValueError):
        integer_identity("sample,0,5,1")
    with pytest.raises(ValueError):
        integer_identity("integers,1\nintegers,1")


def test_changed_rsr_source_is_refused_before_build(tmp_path: Path) -> None:
    source, licence = tmp_path / "source.cu", tmp_path / "LICENSE"
    source.write_text("unreviewed source")
    licence.write_text("MIT License\nCopyright (c) 2026 IndexLab")
    with pytest.raises(ValueError, match="reviewed commit"):
        install_rsr(
            {
                "source": file_reference(source),
                "licence": file_reference(licence),
                "commit": RSR_COMMIT,
            },
            tmp_path,
            tmp_path / "build",
        )
    assert not (tmp_path / "build").exists()

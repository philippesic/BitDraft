import math
import struct
from pathlib import Path

import pytest

from benchmarks.kernel_benchmark import Weights, metrics, samples, validate_activation_manifest
from w1a1_eagle.contracts import Record, number


def test_sample_failure_is_retained() -> None:
    times, exact = samples("sample,0,12.5,1\nsample,1,13.0,0\nsample,2,12.8,1\n")
    assert times == [12.5, 13.0, 12.8]
    assert exact is False


@pytest.mark.parametrize(
    "text",
    ["", "sample,1,12,1", "sample,0,0,1", "sample,0,nan,1", "sample,0,inf,1", "sample,0,12,2"],
)
def test_bad_samples(text: str) -> None:
    with pytest.raises(ValueError):
        samples(text)


def test_metrics_count_multiply_add_once() -> None:
    result = metrics([9.0, 10.0, 11.0], 128, 256, 4, 2, 8192, 800)
    assert math.isclose(number(result["multiply_adds_per_second"]), 26_214_400_000)
    assert math.isclose(number(result["weight_bytes_per_second"]), 819_200_000)
    assert math.isclose(number(result["memory_floor_microseconds"]), 0.01024)
    assert result["spread_percent"] == 20.0


def test_weights_f16_conversion_and_bounds(tmp_path: Path) -> None:
    name = b"scale"
    directory = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    directory += struct.pack("<Q", len(name)) + name + struct.pack("<IQIQ", 1, 2, 1, 0)
    directory += bytes((-len(directory)) % 32)
    source = tmp_path / "weights.gguf"
    source.write_bytes(directory + struct.pack("<ee", 0.5, -0.0))
    weights = Weights(source)
    assert weights.tensor("scale", {1}) == struct.pack("<ff", 0.5, -0.0)
    with pytest.raises(ValueError, match="unsupported"):
        weights.tensor("scale", {26})
    source.write_bytes(directory)
    with pytest.raises(ValueError, match="bounds"):
        weights.tensor("scale", {1})


@pytest.mark.parametrize(("kind", "size"), [(12, 144), (14, 210)])
def test_preserve_baseline_quantized_payload(tmp_path: Path, kind: int, size: int) -> None:
    name = b"weight"
    directory = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    directory += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 256, 1, kind, 0)
    directory += bytes((-len(directory)) % 32)
    payload = bytes(range(size))
    source = tmp_path / "baseline.gguf"
    source.write_bytes(directory + payload)
    weights = Weights(source)
    assert weights.tensors["weight"].kind == kind
    assert weights.tensors["weight"].size == size
    assert weights.tensor("weight", {12, 14}) == payload


@pytest.mark.parametrize(("role", "digest"), [("heldout", "expected"), ("dev", "different")])
def test_activation_manifest_rejects_wrong_role_or_drafter(role: str, digest: str) -> None:
    manifest: Record = {
        "schema": "kernel-activations-v1",
        "role": role,
        "specification": {"drafter": {"sha256": digest}},
    }
    with pytest.raises(ValueError):
        validate_activation_manifest(manifest, "expected")


def test_activation_manifest_accepts_captured_drafter() -> None:
    validate_activation_manifest(
        {
            "schema": "kernel-activations-v1",
            "role": "dev",
            "specification": {"drafter": {"sha256": "expected"}},
        },
        "expected",
    )


def test_experimental_runtime_uses_qualified_kernel() -> None:
    from benchmarks.runtime_edits import SOURCE_EDITS

    root = Path(__file__).resolve().parents[1]
    variants = (root / "native/kernel-variants.cuh").read_text()
    marker = "static __global__ void mma_warp_dot("
    kernel = marker + variants.split(marker, 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    edits = SOURCE_EDITS["kernel_gate_mma_warp_v1"]
    assert edits[0][2] == kernel + "\n" + edits[0][1]
    assert edits[1][2].endswith(edits[1][1])
    assert "n == 4 || n == 8 || n == 16" in edits[1][2]


def test_recovered_runtime_uses_qualified_kernel() -> None:
    from benchmarks.runtime_edits import SOURCE_EDITS

    root = Path(__file__).resolve().parents[1]
    source = (root / "native/kernel-imma.cuh").read_text()
    kernel = source.split("static __global__ void w1a8_sum_codes", 1)[0]
    edits = SOURCE_EDITS["kernel_recovered_imma_v1"]
    assert edits[0][2] == kernel + "\n" + edits[0][1]
    for _, original, replacement in edits[1:]:
        assert replacement.endswith(original)
        assert "n == 2 || n == 4 || n == 8" in replacement
        assert "recovered_imma<4,false>" in replacement


def test_slot_rounding_edit_keeps_the_shared_bound_path_and_applies_once(
    tmp_path: Path,
) -> None:
    from benchmarks.native import apply_source_edits
    from benchmarks.runtime_edits import SOURCE_EDITS

    edits = SOURCE_EDITS["slot_rounding_v1"]
    assert {relative for relative, _, _ in edits} == {
        "src/models/dflash.cpp",
        "ggml/src/ggml-cuda/w1a1.cu",
        "ggml/src/ggml-impl.h",
        "ggml/src/ggml-cpu/ggml-cpu.c",
    }
    loader, pack, match, kernels, launch, admission, reference = edits
    assert loader[2].startswith(loader[1]) and ".w1ax.clip_anchor" in loader[2]
    assert "GGUF_TYPE_FLOAT32" in loader[2] and "!mode" in loader[2]
    assert pack[2].startswith(pack[1]) and "ubatch.n_seqs_unq" in pack[2]
    assert "op_params[3]" in pack[2] and "n_tokens-blocks" in pack[2]
    assert match[1].replace(",12)", ",16)") == match[2]
    assert kernels[2].endswith(kernels[1])
    assert "token%period==0)?anchor:clip" in kernels[2]
    assert launch[2].endswith("} else {\n" + launch[1] + "        }\n")
    assert "if(anchor>0) {" in launch[2]
    assert "uint32_t i = 3;" in admission[1] and "uint32_t i = 5;" in admission[2]
    assert reference[2] == reference[1].replace(
        "(dst));\n", "(dst));\n    GGML_ASSERT(ggml_get_op_params_i32(dst, 3) == 0);\n"
    )
    source = tmp_path / "source"
    for relative in {relative for relative, _, _ in edits}:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "head\n" + "".join(old + "\n" for name, old, _ in edits if name == relative)
        )
    apply_source_edits(source, "slot_rounding_v1")
    edited = (source / "ggml/src/ggml-cuda/w1a1.cu").read_text()
    assert edited.count("exec_tiled_finish<<<") == 1 and edited.count("exec_slot_finish<<<") == 1
    with pytest.raises(ValueError, match="does not match"):
        apply_source_edits(source, "slot_rounding_v1")


SLOT_ADMISSION_HARNESS = """#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define GGML_MAX_SRC 10
#define GGML_MAX_OP_PARAMS 64
struct ggml_tensor {
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int64_t ne[4];
    struct ggml_tensor * src[GGML_MAX_SRC];
};
static int32_t ggml_get_op_params_i32(const struct ggml_tensor * tensor, uint32_t i) {
    return tensor->op_params[i];
}
static bool admits(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
ADMISSION
    return true;
}
int main(void) {
    long long rows;
    int mode, anchor, period, extra;
    while (scanf("%lld %d %d %d %d", &rows, &mode, &anchor, &period, &extra) == 5) {
        struct ggml_tensor a, op;
        memset(&a, 0, sizeof(a));
        memset(&op, 0, sizeof(op));
        a.ne[1] = rows;
        op.src[0] = &a;
        op.op_params[0] = 4;
        op.op_params[1] = mode;
        op.op_params[2] = 0x3f800000;
        op.op_params[3] = anchor;
        op.op_params[4] = period;
        op.op_params[5] = extra;
        printf("%d\\n", admits(&op) ? 1 : 0);
    }
    return 0;
}
"""


def float_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", value), "little", signed=True)


def test_slot_rounding_admission_accepts_only_the_anchor_layout(tmp_path: Path) -> None:
    import subprocess

    from benchmarks.runtime_edits import SOURCE_EDITS

    admission = SOURCE_EDITS["slot_rounding_v1"][5]
    assert admission[0] == "ggml/src/ggml-impl.h"
    program = tmp_path / "admission.c"
    program.write_text(SLOT_ADMISSION_HARNESS.replace("ADMISSION\n", admission[2]))
    binary = tmp_path / "admission"
    subprocess.run(
        ["cc", "-std=c11", "-Wall", "-Werror", str(program), "-o", str(binary)], check=True
    )
    anchor = float_bits(0.75)
    cases = [
        ((8, 0, 0, 0, 0), 1),
        ((8, 1, 0, 0, 0), 1),
        ((8, 1, anchor, 4, 0), 1),
        ((8, 2, anchor, 8, 0), 1),
        ((8, 1, anchor, 1, 0), 1),
        ((7, 1, anchor, 0, 0), 1),
        ((8, 0, anchor, 4, 0), 0),
        ((8, 0, anchor, 0, 0), 0),
        ((8, 1, 0, 4, 0), 0),
        ((8, 1, anchor, -4, 0), 0),
        ((8, 1, anchor, 3, 0), 0),
        ((8, 1, anchor, 16, 0), 0),
        ((8, 1, float_bits(-0.75), 4, 0), 0),
        ((8, 1, float_bits(-0.0), 0, 0), 0),
        ((8, 1, float_bits(math.inf), 4, 0), 0),
        ((8, 1, float_bits(math.nan), 4, 0), 0),
        ((8, 1, anchor, 4, 1), 0),
        ((8, 0, 0, 0, 1), 0),
    ]
    stdin = "".join(" ".join(str(value) for value in case) + "\n" for case, _ in cases)
    result = subprocess.run([str(binary)], input=stdin, capture_output=True, text=True, check=True)
    assert result.stdout.split() == [str(expected) for _, expected in cases]


def write_kernel_weights(
    path: Path,
    metadata: dict[str, int | float],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        encoded = key.encode()
        header += struct.pack("<Q", len(encoded)) + encoded
        header += (
            struct.pack("<Ii", 4, value) if type(value) is int else struct.pack("<If", 6, value)
        )
    payload = b""
    for name, shape, kind, data in tensors:
        encoded = name.encode()
        header += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", kind, len(payload))
        payload += data
        payload += bytes((-len(payload)) % 32)
    path.write_bytes(header + bytes((-len(header)) % 32) + payload)


@pytest.mark.parametrize("slots", [1, 4])
def test_feed_forward_adapter_preserves_geometry_and_payload(tmp_path: Path, slots: int) -> None:
    from benchmarks.kernel_benchmark import prepare_feed_forward
    from benchmarks.native import file_reference
    from w1a1_eagle.data import Artifact

    base = "blk.0.ffn_down"
    one, baseline, activation, manifest = (
        tmp_path / name for name in ("one", "four", "x", "manifest")
    )
    write_kernel_weights(
        one,
        {base + ".w1ax.bits": 8, base + ".w1ax.mode": 2, base + ".w1ax.clip": 1.0},
        [
            (base + ".w1a1_packed", (8, 16), 26, bytes(512)),
            (base + ".w1a1_scale", (16,), 0, struct.pack("<16f", *([1.0] * 16))),
            (base + ".w1ax_midpoint", (16,), 0, bytes(64)),
            (base + ".w1ax_gain", (256,), 0, struct.pack("<256f", *([1.0] * 256))),
        ],
    )
    payload = bytes(range(210)) * 16
    write_kernel_weights(baseline, {}, [(base + ".weight", (256, 16), 14, payload)])
    captured_slots = 2 if slots == 1 else slots
    activation.write_bytes(struct.pack("<f", 3.0) * (256 * captured_slots))
    manifest.write_text("{}")
    inputs = {
        key: Artifact.capture(path)
        for key, path in [("drafter", one), ("baseline", baseline), ("activations", manifest)]
    }
    specification: Record = {
        "layers": [0],
        "groups": ["down"],
        "slots": [slots],
        "variants": ["reference", "baseline"],
        "warmup": 1,
        "repeats": 1,
        "trials": 3,
    }
    output = tmp_path / "prepared"
    prepared = prepare_feed_forward(
        specification,
        inputs,
        tmp_path / "probe",
        {"tensors": {base + f".{captured_slots}": file_reference(activation)}},
        output,
    )
    assert len(prepared) == 2
    command, row = prepared[1]
    assert command[3:7] == ["16", "256", str(slots), "14"]
    assert row["weight_format"] == "Q6_K"
    assert row["weight_bytes"] == 3360
    assert (output / f"layer-0-down-{slots}.down.quantized").read_bytes() == payload
    assert (output / f"layer-0-down-{slots}.down.input").read_bytes() == activation.read_bytes()[
        : 256 * slots * 4
    ]
    write_kernel_weights(baseline, {}, [(base + ".weight", (512, 8), 14, payload)])
    with pytest.raises(ValueError, match="geometries differ"):
        prepare_feed_forward(
            specification,
            inputs,
            tmp_path / "probe",
            {"tensors": {base + f".{captured_slots}": file_reference(activation)}},
            tmp_path / "mismatched",
        )

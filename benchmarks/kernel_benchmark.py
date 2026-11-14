import hashlib
import math
import statistics
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from benchmarks.kernel_probe import runtime_kernels
from benchmarks.measurement import now
from benchmarks.native import file_reference
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.numerical.gguf_readback import Reader
from w1a1_eagle.records import artifact, canonical, read

VARIANTS = ("reference", "cross", "mma", "mma-broadcast", "mma-warp", "baseline")


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    kind: int
    offset: int
    size: int


class Weights:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tensors: dict[str, Tensor] = {}
        with path.open("rb") as stream:
            reader = Reader(stream, path.stat().st_size)
            fields, count, alignment = reader.header(True)
            self.metadata = {key: value for key, value, _, _ in fields}
            for _ in range(count):
                name, dimensions = reader.text(), reader.uint(4)
                if not 1 <= dimensions <= 4 or name in self.tensors:
                    raise ValueError("invalid tensor directory")
                shape = tuple(reader.uint(8) for _ in range(dimensions))
                kind, offset = reader.uint(4), reader.uint(8)
                if not all(0 < dimension <= 2**31 for dimension in shape) or offset % alignment:
                    raise ValueError("invalid tensor geometry or alignment")
                if kind in {0, 26}:
                    size = math.prod(shape) * 4
                elif kind == 1:
                    size = math.prod(shape) * 2
                elif kind in {12, 14}:
                    if math.prod(shape) % 256:
                        raise ValueError("K-quant tensor is not block aligned")
                    size = math.prod(shape) // 256 * (144 if kind == 12 else 210)
                else:
                    size = 0
                self.tensors[name] = Tensor(shape, kind, offset, size)
            self.beginning = (stream.tell() + alignment - 1) // alignment * alignment

    def tensor(self, name: str, kinds: set[int]) -> bytes:
        tensor = self.tensors[name]
        if tensor.kind not in kinds or not tensor.size:
            raise ValueError("unsupported tensor type: " + name)
        with self.path.open("rb") as stream:
            stream.seek(self.beginning + tensor.offset)
            data = Reader(stream, self.path.stat().st_size).read(tensor.size)
        if tensor.kind == 1:
            values = cast(tuple[float, ...], struct.unpack("<" + str(len(data) // 2) + "e", data))
            return struct.pack("<" + str(len(values)) + "f", *values)
        return data


def samples(text: str) -> tuple[list[float], bool]:
    values: list[float] = []
    exact = True
    for line in text.splitlines():
        if not line.startswith("sample,"):
            continue
        fields = line.split(",")
        if len(fields) != 4 or int(fields[1]) != len(values) or fields[3] not in {"0", "1"}:
            raise ValueError("invalid native sample record")
        value = float(fields[2])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("nonpositive or nonfinite timing")
        values.append(value)
        exact = exact and fields[3] == "1"
    if not values:
        raise ValueError("native probe produced no samples")
    return values, exact


def metrics(
    times: list[float],
    m: int,
    k: int,
    slots: int,
    matrices: int,
    weight_bytes: int,
    bandwidth: float,
) -> Record:
    median = statistics.median(times)
    seconds = median * 1e-6
    return {
        "microseconds": median,
        "minimum_microseconds": min(times),
        "maximum_microseconds": max(times),
        "spread_percent": (max(times) - min(times)) / median * 100,
        "multiply_adds_per_second": m * k * slots * matrices / seconds,
        "weight_bytes_per_second": weight_bytes / seconds,
        "memory_floor_microseconds": weight_bytes / (bandwidth * 1e9) * 1e6,
        "weight_bytes": weight_bytes,
    }


def build_kernel_benchmark(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) != {"schema", "runtime", "compiler", "architecture"}
        or raw["schema"] != "kernel-build-v1"
    ):
        raise ValueError("unsupported kernel build specification")
    root = specification.parent
    runtime = artifact(raw["runtime"], root)
    compiler = artifact(raw["compiler"], root)
    runtime.verify()
    compiler.verify()
    built = read(runtime.path)
    if built.get("status") != "built":
        raise ValueError("runtime must have a successful build receipt")
    architecture = string(raw["architecture"])
    if not architecture.isalnum():
        raise ValueError("invalid CUDA architecture")
    if output.exists():
        raise ValueError("kernel build output must be new")
    owner = Store(output)
    runtime_root = runtime.path.parent
    source = runtime_root / "source"
    cuda_source = source / "ggml/src/ggml-cuda"
    kernel_source = cuda_source / "w1a1.cu"
    owner.atomic(output / "runtime-kernels.cuh", runtime_kernels(kernel_source).encode())
    project = Path(__file__).resolve().parents[1]
    authored = [
        project / "native" / name
        for name in ("kernel-benchmark.cu", "kernel-variants.cuh", "kernel-capture.cpp")
    ]
    for path in authored:
        owner.atomic(output / path.name, path.read_bytes())
    includes = [
        output,
        runtime_root / "compat-include",
        source / "ggml/include",
        source / "ggml/src",
        cuda_source,
    ]
    libraries = runtime_root / "build/bin"
    runtime_files = [artifact(value, runtime_root) for value in sequence(built["artifacts"])]
    for library_name in ("libggml-cuda.so", "libggml-base.so", "libggml.so"):
        resolved = (libraries / library_name).resolve()
        matching = [value for value in runtime_files if value.path.resolve() == resolved]
        if len(matching) != 1:
            raise ValueError("runtime receipt does not authenticate linked library")
        matching[0].verify()
    command = [
        "nice",
        "-n",
        "19",
        str(compiler.path),
        "-O3",
        "-std=c++17",
        "-arch=sm_" + architecture,
        "-DNDEBUG",
        "-use_fast_math",
        "--ftz=false",
        "-DGGML_CUDA_USE_GRAPHS",
    ]
    command.extend("-I" + str(path) for path in includes)
    command.extend(
        [
            str(output / "kernel-benchmark.cu"),
            "-L" + str(libraries),
            "-lggml-cuda",
            "-lggml-base",
            "-lggml",
            "-lcublas",
            "-Xlinker",
            "-rpath=" + str(libraries),
            "-o",
            str(output / "kernel-benchmark"),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    owner.atomic(output / "build.log", (result.stdout + result.stderr).encode())
    if result.returncode:
        raise ValueError("kernel compile failed; see " + str(output / "build.log"))
    capture_command = [
        "nice",
        "-n",
        "19",
        str(compiler.path),
        "-O2",
        "-std=c++17",
        "-arch=sm_" + architecture,
        "-DGGML_CUDA_USE_GRAPHS",
        "-x",
        "cu",
        "-shared",
        "-Xcompiler",
        "-fPIC",
    ]
    capture_command.extend("-I" + str(path) for path in includes)
    capture_command.extend(
        [
            str(output / "kernel-capture.cpp"),
            "-L" + str(libraries),
            "-lggml-cuda",
            "-lggml-base",
            "-ldl",
            "-o",
            str(output / "kernel-capture.so"),
        ]
    )
    capture_result = subprocess.run(capture_command, capture_output=True, text=True, check=False)
    owner.atomic(
        output / "capture-build.log", (capture_result.stdout + capture_result.stderr).encode()
    )
    if capture_result.returncode:
        raise ValueError("capture helper compile failed; inspect capture-build.log")
    record_out: Record = {
        "schema": "kernel-build-result-v1",
        "command": command,
        "capture_command": capture_command,
        "capture_library": file_reference(output / "kernel-capture.so"),
        "runtime": raw["runtime"],
        "runtime_kernel": file_reference(kernel_source),
        "kernel_slice_sha256": hashlib.sha256(runtime_kernels(kernel_source).encode()).hexdigest(),
        "compiler": raw["compiler"],
        "sources": [file_reference(path) for path in authored],
        "executable": file_reference(output / "kernel-benchmark"),
        "libraries": [
            file_reference(libraries / name)
            for name in ("libggml-cuda.so", "libggml-base.so", "libggml.so")
        ],
    }
    owner.atomic(output / "result.json", canonical(record_out))
    return record_out


def validate_activation_manifest(manifest: Record, drafter_digest: str) -> None:
    if manifest.get("schema") != "kernel-activations-v1" or manifest.get("role") != "dev":
        raise ValueError("authenticated DEV activation manifest required")
    specification = record(manifest["specification"])
    drafter = record(specification["drafter"])
    if drafter.get("sha256") != drafter_digest:
        raise ValueError("activation capture drafter differs from benchmark drafter")


def benchmark_kernels(specification: Path, output: Path) -> Record:
    raw = read(specification)
    required = {
        "schema",
        "build",
        "drafter",
        "baseline",
        "activations",
        "layers",
        "groups",
        "slots",
        "variants",
        "warmup",
        "repeats",
        "trials",
        "bandwidth_gb_s",
    }
    if set(raw) != required or raw["schema"] != "kernel-benchmark-v1":
        raise ValueError("unsupported kernel benchmark specification")
    root = specification.parent
    inputs = {
        name: artifact(raw[name], root) for name in ("build", "drafter", "baseline", "activations")
    }
    for value in inputs.values():
        value.verify()
    build = read(inputs["build"].path)
    executable = artifact(build["executable"], inputs["build"].path.parent)
    executable.verify()
    libraries = [
        artifact(value, inputs["build"].path.parent) for value in sequence(build["libraries"])
    ]
    for library in libraries:
        library.verify()
    activation_manifest = read(inputs["activations"].path)
    validate_activation_manifest(activation_manifest, inputs["drafter"].sha256)
    layers = [integer(value) for value in sequence(raw["layers"])]
    groups = [string(value) for value in sequence(raw["groups"])]
    slots = [integer(value) for value in sequence(raw["slots"])]
    variants = [string(value) for value in sequence(raw["variants"])]
    warmup, repeats, trials = (integer(raw[name]) for name in ("warmup", "repeats", "trials"))
    bandwidth = number(raw["bandwidth_gb_s"])
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(not 0 <= layer < 64 for layer in layers)
    ):
        raise ValueError("distinct layer indices required")
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(value not in {"gate", "down"} for value in groups)
    ):
        raise ValueError("distinct gate/down matrix groups required")
    if (
        not slots
        or len(set(slots)) != len(slots)
        or any(value not in {2, 4, 8, 16} for value in slots)
    ):
        raise ValueError("distinct supported slot counts required")
    if (
        not variants
        or len(set(variants)) != len(variants)
        or any(value not in VARIANTS for value in variants)
    ):
        raise ValueError("distinct named variants required")
    if (
        not 1 <= warmup <= 10000
        or not 1 <= repeats <= 100000
        or not 3 <= trials <= 101
        or not math.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise ValueError("invalid timing bounds")
    if output.exists():
        raise ValueError("kernel measurement output must be new")
    owner = Store(output)
    one, four = Weights(inputs["drafter"].path), Weights(inputs["baseline"].path)
    prepared: list[tuple[list[str], Record]] = []
    activation_rows = record(activation_manifest["tensors"])
    for layer in layers:
        for group in groups:
            base = f"blk.{layer}.ffn_{group}"
            geometry = one.tensors[base + ".w1a1_packed"].shape
            if len(geometry) != 2:
                raise ValueError("matrix tensor required")
            k, m = geometry[0] * 32, geometry[1]
            bits = integer(one.metadata[base + ".w1ax.bits"])
            mode = integer(one.metadata[base + ".w1ax.mode"])
            clip_value = one.metadata[base + ".w1ax.clip"]
            clip = (
                cast(tuple[float], struct.unpack("<e", struct.pack("<H", integer(clip_value))))[0]
                if type(clip_value) is int
                else number(clip_value)
            )
            for count in slots:
                prefix = output / f"layer-{layer}-{group}-{count}"
                activation = artifact(
                    activation_rows[f"{base}.{count}"], inputs["activations"].path.parent
                )
                activation.verify()
                if activation.size != k * count * 4:
                    raise ValueError("activation geometry differs")
                owner.atomic(Path(str(prefix) + ".input"), activation.path.read_bytes())
                for suffix, tensor_suffix, kinds in (
                    ("weights", "w1a1_packed", {26}),
                    ("scales", "w1a1_scale", {0, 1}),
                    ("midpoint", "w1ax_midpoint", {0, 1}),
                    ("gain", "w1ax_gain", {0, 1}),
                ):
                    owner.atomic(
                        Path(str(prefix) + "." + suffix),
                        one.tensor(base + "." + tensor_suffix, kinds),
                    )
                if group == "gate":
                    for suffix, tensor_suffix, kinds in (
                        ("weights", "w1a1_packed", {26}),
                        ("scales", "w1a1_scale", {0, 1}),
                        ("midpoint", "w1ax_midpoint", {0, 1}),
                    ):
                        owner.atomic(
                            Path(str(prefix) + ".up." + suffix),
                            one.tensor(f"blk.{layer}.ffn_up." + tensor_suffix, kinds),
                        )
                for variant in variants:
                    names = (
                        ("gate", "up") if variant == "baseline" and group == "gate" else (group,)
                    )
                    for name in names:
                        selected = prefix
                        if variant == "baseline":
                            if four.tensors[f"blk.{layer}.ffn_{name}.weight"].shape != (k, m):
                                raise ValueError("Baseline and one-bit matrix geometries differ")
                            selected = Path(str(prefix) + "." + name)
                            owner.atomic(
                                Path(str(selected) + ".input"), activation.path.read_bytes()
                            )
                            owner.atomic(
                                Path(str(selected) + ".quantized"),
                                four.tensor(f"blk.{layer}.ffn_{name}.weight", {12, 14}),
                            )
                        matrices = 2 if group == "gate" and variant != "baseline" else 1
                        weight_bytes = (
                            four.tensors[f"blk.{layer}.ffn_{name}.weight"].size
                            if variant == "baseline"
                            else matrices * (m * k // 8 + m * 8)
                        )
                        command = [
                            str(executable.path),
                            str(selected),
                            variant,
                            str(m),
                            str(k),
                            str(count),
                            str(
                                four.tensors[f"blk.{layer}.ffn_{name}.weight"].kind
                                if variant == "baseline"
                                else bits
                            ),
                            str(mode),
                            str(clip),
                            str(warmup),
                            str(repeats),
                            str(trials),
                            str(int(matrices == 2)),
                            "v1",
                        ]
                        row: Record = {
                            "layer": layer,
                            "group": group,
                            "matrix": name,
                            "variant": variant,
                            "slots": count,
                            "m": m,
                            "k": k,
                            "matrices": matrices,
                            "weight_bytes": weight_bytes,
                            "weight_format": {12: "Q4_K", 14: "Q6_K"}[
                                four.tensors[f"blk.{layer}.ffn_{name}.weight"].kind
                            ]
                            if variant == "baseline"
                            else "W1Ax",
                            "activation": file_reference(activation.path),
                        }
                        prepared.append((command, row))
    owner.atomic(output / "specification.json", canonical(raw))
    results: list[Record] = []
    with device_lease("cuda"):
        identity = cuda_identity()
        began = now()
        for index, (command, row) in enumerate(prepared):
            measured = subprocess.run(
                command,
                env=native_environment(libraries[0].path.parent),
                capture_output=True,
                text=True,
                check=False,
            )
            owner.atomic(output / f"{index:04d}.log", (measured.stdout + measured.stderr).encode())
            if measured.returncode:
                raise ValueError("native benchmark failed; see " + str(output / f"{index:04d}.log"))
            times, exact = samples(measured.stdout)
            if len(times) != trials:
                raise ValueError("native trial count differs")
            row.update(
                metrics(
                    times,
                    integer(row["m"]),
                    integer(row["k"]),
                    integer(row["slots"]),
                    integer(row["matrices"]),
                    integer(row["weight_bytes"]),
                    bandwidth,
                )
            )
            row.update(
                {
                    "samples_microseconds": times,
                    "byte_identical": exact,
                    "reference": "ordinary runtime baseline graph"
                    if row["variant"] == "baseline"
                    else "runtime-v7 dispatched one-bit output",
                }
            )
            results.append(row)
            owner.atomic(output / "partial.json", canonical({"rows": results}))
        seconds = now() - began
    result: Record = {
        "schema": "kernel-measurement-v1",
        "specification": raw,
        "gpu": identity,
        "gpu_seconds": seconds,
        "rows": results,
        "all_byte_identical": all(row["byte_identical"] is True for row in results),
        "artifact_disposition": "Retain source/input/build identities, specifications, raw samples and logs; extracted tensors are reproducible copies of retained read-only inputs.",
    }
    owner.atomic(output / "result.json", canonical(result))
    return result

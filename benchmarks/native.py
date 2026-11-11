import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, sequence, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import artifact, canonical, read, reference


def file_reference(path: Path) -> Record:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def unpack_bundle(bundle: Path, destination: Path, maximum: int) -> None:
    seen: set[str] = set()
    total = 0
    with tarfile.open(bundle, "r|gz") as archive:
        for member in archive:
            relative = Path(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] not in {"source", "compat-include"}
                or member.name in seen
                or not member.isfile()
            ):
                raise ValueError("native bundle must contain unique regular source/header files")
            seen.add(member.name)
            total += member.size
            if total > maximum:
                raise ValueError("native source bundle exceeds unpacked budget")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("native source member unavailable")
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
    if not (destination / "source/CMakeLists.txt").is_file():
        raise ValueError("native bundle lacks the complete source root")


def build(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) != {"schema", "bundle", "cuda_compiler", "architecture", "jobs", "unpacked_bytes"}
        or raw["schema"] != 1
    ):
        raise ValueError("unsupported native build specification")
    bundle = artifact(raw["bundle"], specification.resolve().parent)
    compiler = artifact(raw["cuda_compiler"], specification.resolve().parent)
    bundle.verify()
    compiler.verify()
    architecture = string(raw["architecture"])
    jobs, bound = integer(raw["jobs"]), integer(raw["unpacked_bytes"])
    if not architecture.isalnum() or not 1 <= jobs <= 8 or not 1 <= bound <= 1024**3:
        raise ValueError("native build bounds invalid")
    if output.exists():
        raise ValueError("native build output must be new; no reused objects")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    snapshot = freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", specification.read_bytes())
    unpack_bundle(bundle.path, store.root, bound)
    source, compiled = store.root / "source", store.root / "build"
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(compiled),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_COMPILER=" + str(compiler.path),
        "-DCMAKE_CUDA_ARCHITECTURES=" + architecture,
        "-DCMAKE_CUDA_FLAGS=-I" + str(store.root / "compat-include"),
        "-DCMAKE_UNITY_BUILD=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
        "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16",
    ]
    commands = [
        configure,
        [
            "cmake",
            "--build",
            str(compiled),
            "--target",
            "llama-server",
            "llama-cli",
            "-j" + str(jobs),
        ],
    ]
    binaries = compiled / "bin"
    commands.extend(helper_commands(snapshot, source, binaries, binaries))
    store.atomic(store.root / "commands.json", canonical({"commands": commands}))
    try:
        for index, command in enumerate(commands):
            with (store.root / (str(index) + "-build.log")).open("wb") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=native_environment(binaries),
                    check=True,
                    timeout=3600,
                )
        files = [
            file_reference(path)
            for path in sorted(binaries.iterdir())
            if path.is_file() and not path.is_symlink()
        ]
        result: Record = {
            "status": "built",
            "source_bundle": reference(bundle),
            "code_sha256": code,
            "artifacts": files,
            "objects_reused": False,
            "qualification": "build only; serving/capture must be exercised on this hardware",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        store.atomic(store.root / "failure.json", canonical({"error": str(error)}))
        raise


def helper_commands(
    snapshot: Path, source: Path, binaries: Path, libraries: Path
) -> list[list[str]]:
    commands: list[list[str]] = []
    for name, extra in (
        ("tokenize", ["-lllama", "-lllama-common"]),
        ("capture", ["-lllama", "-lggml", "-lggml-base", "-ldl"]),
    ):
        commands.append(
            [
                "c++",
                "-O2",
                "-std=c++17",
                str(snapshot / "native" / (name + ".cpp")),
                "-I",
                str(source / "include"),
                "-I",
                str(source / "ggml/include"),
                "-I",
                str(source / "vendor"),
                "-I",
                str(source / "common"),
                "-L",
                str(libraries),
                *extra,
                "-Wl,-rpath," + str(libraries),
                "-o",
                str(binaries / ("binary-" + name)),
            ]
        )
    return commands


def helpers(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "runtime"} or raw["schema"] != 1:
        raise ValueError("unsupported helper build specification")
    runtime = artifact(raw["runtime"], specification.resolve().parent)
    runtime.verify()
    previous = read(runtime.path)
    bundle = artifact(previous["source_bundle"], runtime.path.parent)
    bundle.verify()
    files = [artifact(value, runtime.path.parent) for value in sequence(previous["artifacts"])]
    for value in files:
        value.verify()
    servers = [value for value in files if value.path.name == "llama-server"]
    if len(servers) != 1 or output.exists():
        raise ValueError("one authenticated server and a new helper output required")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[1]
    code = source_identity(code_root)
    snapshot = freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    unpack_bundle(bundle.path, store.root, 1024**3)
    binaries = Store(store.root / "bin").root
    libraries = servers[0].path.parent
    commands = helper_commands(snapshot, store.root / "source", binaries, libraries)
    store.atomic(store.root / "commands.json", canonical({"commands": commands}))
    for index, command in enumerate(commands):
        with (store.root / (str(index) + "-build.log")).open("wb") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=native_environment(libraries),
                timeout=120,
                check=True,
            )
    outputs = [
        reference(value)
        for value in files
        if value.path.name not in {"binary-tokenize", "binary-capture"}
    ]
    outputs.extend(file_reference(path) for path in sorted(binaries.iterdir()) if path.is_file())
    result: Record = {
        "status": "built",
        "runtime": reference(runtime),
        "source_bundle": reference(bundle),
        "code_sha256": code,
        "artifacts": outputs,
        "qualification": "helper build only; template/capture matching requires execution",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result

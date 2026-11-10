import ast
import sys
from pathlib import PurePosixPath

from tools.config import BOUNDARIES, ENTRY_MODULES
from tools.contracts import Diagnostic, sequence, string

PURE_IMPORTS = {
    "math",
    "collections",
    "dataclasses",
    "typing",
    "enum",
    "itertools",
    "functools",
    "statistics",
}
FORBIDDEN = tuple(
    (string(value).split(" -> ")[0], string(value).split(" -> ")[1])
    for value in sequence(BOUNDARIES["forbidden"])
)
WRITERS = {string(value) for value in sequence(BOUNDARIES["writers"])}
IO_FREE = {string(value) for value in sequence(BOUNDARIES["io_free"])}


def module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def inspect_tree(files: dict[str, str]) -> list[Diagnostic]:
    diagnostics: set[Diagnostic] = set()
    graph: dict[str, set[str]] = {}
    paths: dict[str, str] = {}
    for path, text in sorted(files.items()):
        if not path.endswith(".py"):
            continue
        module = module_name(path)
        paths[module] = path
        graph[module] = set()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            diagnostics.add(Diagnostic("ARCH_SYNTAX", path))
            continue
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    graph[module].add(alias.name)
                    aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                package = (
                    module.split(".")
                    if PurePosixPath(path).name == "__init__.py"
                    else module.split(".")[:-1]
                )
                base = (
                    ".".join(
                        package[: len(package) - node.level + 1]
                        + ([node.module] if node.module else [])
                    )
                    if node.level
                    else node.module or ""
                )
                graph[module].add(base)
                for alias in node.names:
                    aliases[alias.asname or alias.name] = base + "." + alias.name
                    if base + "." + alias.name in {module_name(p) for p in files}:
                        graph[module].add(base + "." + alias.name)

        def name(node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return aliases.get(node.id, node.id)
            if isinstance(node, ast.Subscript):
                return name(node.value)
            if isinstance(node, ast.Attribute):
                return name(node.value) + "." + node.attr
            return ""

        for dep in graph[module]:
            if dep.split(".")[0] in {"runs", "research", "history", "checkouts", "outputs"}:
                diagnostics.add(Diagnostic("ARCH_HISTORY", path))
            if (
                module in {"tools.contracts", "w1a1_eagle.contracts"}
                and dep != "w1a1_eagle.contracts"
                and (
                    dep.split(".")[0] not in PURE_IMPORTS
                    or dep.split(".")[0] not in sys.stdlib_module_names
                )
            ):
                diagnostics.add(Diagnostic("ARCH_PURE", path))
            for source, target in FORBIDDEN:
                if matches(module, source) and matches(dep, target):
                    diagnostics.add(Diagnostic("ARCH_DIRECT", path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call = name(node.func)
                if call in {
                    "exec",
                    "eval",
                    "compile",
                    "__import__",
                    "importlib.import_module",
                    "importlib.util.spec_from_file_location",
                    "runpy.run_path",
                } or call.endswith(".exec_module"):
                    diagnostics.add(Diagnostic("ARCH_DYNAMIC", path))
                if (
                    call.startswith("sys.path.")
                    or call == "setattr"
                    and any(
                        isinstance(a, ast.Constant) and a.value == "provider" for a in node.args
                    )
                ):
                    diagnostics.add(Diagnostic("ARCH_SYSPATH", path))
                if (
                    call in {"os.getenv", "os.environ.get", "os.environ.copy"}
                    and path != "tools/config.py"
                    and module != "w1a1_eagle.config"
                ):
                    diagnostics.add(Diagnostic("ENV_OWNER", path))
                if call in {
                    "time.time",
                    "time.perf_counter",
                    "time.monotonic",
                    "time.monotonic_ns",
                    "time.perf_counter_ns",
                    "time.time_ns",
                    "datetime.datetime.now",
                    "datetime.datetime.utcnow",
                    "datetime.date.today",
                    "torch.cuda.synchronize",
                    "torch.cuda.Event",
                } and module not in {
                    "benchmarks.measurement",
                    "w1a1_eagle.clock",
                    "w1a1_eagle.device",
                }:
                    diagnostics.add(Diagnostic("TIMING_OWNER", path))
                if call.startswith("random.") and call != "random.Random":
                    diagnostics.add(Diagnostic("SEEDED_RANDOM", path))
                if call == "random.Random" and (
                    not node.args
                    or isinstance(node.args[0], ast.Constant)
                    and node.args[0].value is None
                ):
                    diagnostics.add(Diagnostic("SEEDED_RANDOM", path))
                if (call.startswith("numpy.random.") and call != "numpy.random.default_rng") or (
                    call in {"torch.rand", "torch.randn", "torch.randint", "torch.randperm"}
                    and not any(keyword.arg == "generator" for keyword in node.keywords)
                ):
                    diagnostics.add(Diagnostic("SEEDED_RANDOM", path))
                if call == "numpy.random.default_rng" and not node.args and not node.keywords:
                    diagnostics.add(Diagnostic("SEEDED_RANDOM", path))
                if module in IO_FREE and (
                    call
                    in {"open", "torch.save", "os.replace", "os.rename", "os.remove", "os.unlink"}
                    or call.endswith((".write_text", ".write_bytes", ".open", ".mkdir", ".unlink"))
                ):
                    diagnostics.add(
                        Diagnostic(
                            "ARCH_PURE" if module.endswith("contracts") else "OWNER_PATH", path
                        )
                    )
                if (
                    call in {"sink.publish", "store.publish", "ledger.assign_role"}
                    and module not in WRITERS
                ):
                    diagnostics.add(Diagnostic("OWNER_CALL", path))
                if any(
                    isinstance(a, ast.Constant)
                    and isinstance(a.value, str)
                    and (
                        "binary-eagle-decoding" in a.value
                        or a.value.startswith(("runs/", "research/"))
                    )
                    for a in node.args
                ) and call.endswith(("Path", "run", "Popen")):
                    diagnostics.add(Diagnostic("ARCH_HISTORY", path))
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(
                    name(t).startswith("sys.path") or name(t).endswith(".provider") for t in targets
                ):
                    diagnostics.add(Diagnostic("ARCH_SYSPATH", path))
            if (
                isinstance(node, ast.Subscript)
                and name(node.value) == "os.environ"
                and path != "tools/config.py"
                and module != "w1a1_eagle.config"
            ):
                diagnostics.add(Diagnostic("ENV_OWNER", path))
            if isinstance(node, (ast.If, ast.Match, ast.Dict)) and not module.endswith(
                ".discovery"
            ):
                if any(
                    isinstance(n, ast.Name) and n.id in {"adapter_id", "dataset_id", "workload_id"}
                    for n in ast.walk(node)
                ):
                    diagnostics.add(Diagnostic("DISCOVERY_BRANCH", path))
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and any(isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(node.test))
                and module not in ENTRY_MODULES
            ):
                diagnostics.add(Diagnostic("PUBLIC_ENTRY", path))
        pending: list[ast.AST] = list(tree.body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pending.extend(
                    node.decorator_list
                    + node.args.defaults
                    + [d for d in node.args.kw_defaults if d is not None]
                )
                continue
            if isinstance(node, ast.If) and any(
                isinstance(n, ast.Name) and n.id == "__name__" for n in ast.walk(node.test)
            ):
                continue
            if isinstance(node, ast.Call):
                call = name(node.func)
                if (
                    call.endswith(
                        (
                            ".mkdir",
                            ".write_text",
                            ".write_bytes",
                            ".unlink",
                            ".open",
                            ".run",
                            ".Popen",
                            ".synchronize",
                            ".rmtree",
                            ".makedirs",
                        )
                    )
                    or call == "open"
                    or call
                    in {
                        "torch.tensor",
                        "torch.zeros",
                        "torch.ones",
                        "torch.randn",
                        "torch.rand",
                        "torch.empty",
                        "torch.cuda.init",
                        "torch.cuda.set_device",
                    }
                    or call.endswith(".cuda")
                ):
                    diagnostics.add(Diagnostic("ARCH_IMPORT_EFFECT", path))
            pending.extend(ast.iter_child_nodes(node))
    for module in graph:
        visited: set[str] = set()
        todo = list(graph[module])
        while todo:
            dep = todo.pop()
            if dep == module:
                diagnostics.add(Diagnostic("ARCH_CYCLE", paths[module]))
            if dep in visited:
                continue
            visited.add(dep)
            todo.extend(graph.get(dep, set()))
        for source, target in FORBIDDEN:
            if (
                matches(module, source)
                and any(matches(dep, target) for dep in visited)
                and not any(matches(dep, target) for dep in graph[module])
            ):
                diagnostics.add(Diagnostic("ARCH_TRANSITIVE", paths[module]))
    return sorted(diagnostics)

from pathlib import Path

from tools.boundaries import inspect_tree
from tools.contracts import Diagnostic
from tools.repo_files import inventory


def test_actual_tool_graph() -> None:
    files, _ = inventory(Path.cwd())
    assert inspect_tree(files) == []


def test_transitive_and_cycle() -> None:
    assert Diagnostic("ARCH_TRANSITIVE", "tools/a.py") in inspect_tree(
        {"tools/a.py": "import bridge\n", "bridge.py": "import tests\n"}
    )
    assert {d.code for d in inspect_tree({"a.py": "import b\n", "b.py": "import a\n"})} == {
        "ARCH_CYCLE"
    }


def test_import_default_and_class_effects() -> None:
    for text in [
        "from pathlib import Path\ndef f(x=Path('x').mkdir()):\n    pass\n",
        "from pathlib import Path\nclass A:\n    x = Path('x').mkdir()\n",
    ]:
        assert inspect_tree({"a.py": text}) == [Diagnostic("ARCH_IMPORT_EFFECT", "a.py")]


def test_alias_and_relative_imports() -> None:
    assert Diagnostic("ARCH_SYSPATH", "a.py") in inspect_tree(
        {"a.py": "import sys\nsys.path[:0] = ['external']\n"}
    )
    assert Diagnostic("ARCH_DYNAMIC", "a.py") in inspect_tree(
        {"a.py": "import importlib.util\nimportlib.util.spec_from_file_location('x', 'x.py')\n"}
    )
    assert (
        inspect_tree(
            {"w1a1_eagle/clock.py": "import time\ndef now():\n    return time.monotonic_ns()\n"}
        )
        == []
    )
    assert Diagnostic("TIMING_OWNER", "a.py") in inspect_tree(
        {"a.py": "import time\ndef now():\n    return time.monotonic_ns()\n"}
    )
    assert inspect_tree(
        {"a.py": "import os as system\ndef f():\n    return system.getenv('X')\n"}
    ) == [Diagnostic("ENV_OWNER", "a.py")]
    assert Diagnostic("ARCH_CYCLE", "p/a.py") in inspect_tree(
        {"p/a.py": "from . import b\n", "p/b.py": "from . import a\n"}
    )


def test_reserved_writer_and_discovery() -> None:
    assert (
        inspect_tree({"w1a1_eagle/checkpoints/store.py": "def f():\n    open('best.json', 'w')\n"})
        == []
    )
    assert inspect_tree({"model.py": "def f():\n    open('best.json', 'w')\n"}) == [
        Diagnostic("OWNER_PATH", "model.py")
    ]
    assert inspect_tree({"a.py": "def f(adapter):\n    return adapter.run()\n"}) == []


def test_owner_rule_uses_module_not_filename_words() -> None:
    assert inspect_tree({"helper.py": "def f():\n    open('best_effort.log', 'w')\n"}) == []
    for text in [
        "import torch\ndef f(value, destination):\n    torch.save(value, destination)\n",
        "import os\ndef f(a,b):\n    os.replace(a,b)\n",
    ]:
        assert Diagnostic("OWNER_PATH", "w1a1_eagle/model.py") in inspect_tree(
            {"w1a1_eagle/model.py": text}
        )
    assert Diagnostic("ARCH_IMPORT_EFFECT", "a.py") in inspect_tree(
        {"a.py": "import shutil\nshutil.rmtree('scratch')\n"}
    )

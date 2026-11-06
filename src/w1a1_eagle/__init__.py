"""W1A1 EAGLE components; metadata tools do not initialize PyTorch."""

from importlib import import_module

__all__ = [
    "DrafterStructureError",
    "DrafterW1A1Adapter",
    "FakeUniformLinear",
    "GROUP_PATHS",
    "UniformQuantConfig",
    "W1A1Config",
    "W1A1Linear",
    "fake_binary_linear",
    "fake_uniform_linear",
    "install_fake_uniform",
    "install_w1a1",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"W1A1Config", "W1A1Linear", "fake_binary_linear"}:
        module = "fake_binary"
    elif name in {"FakeUniformLinear", "UniformQuantConfig", "fake_uniform_linear"}:
        module = "fake_uniform"
    else:
        module = "adapter"
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

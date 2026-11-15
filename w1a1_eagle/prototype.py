import math
from dataclasses import dataclass
from typing import Literal

Levels = Literal["sign", "ternary", "int2", "int4", "full"]


@dataclass(frozen=True)
class Weights:
    levels: Levels = "sign"
    block: int | None = None
    residual: bool = False

    @property
    def default(self) -> bool:
        return self.levels == "sign" and self.block is None and not self.residual


@dataclass(frozen=True)
class Prototype:
    levels: Levels = "sign"
    block: int | None = None
    residual: bool = False
    down_int4_layers: tuple[int, ...] = ()
    teacher: Literal["logits", "hidden"] | None = None
    teacher_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.levels not in ("sign", "ternary", "int2", "int4", "full"):
            raise ValueError("unsupported prototype weight levels")
        if self.block is not None and (type(self.block) is not int or self.block < 1):
            raise ValueError("prototype block scale needs a positive input count")
        if (self.block is not None or self.residual) and self.levels != "sign":
            raise ValueError("block scales and residual signs apply to one-bit weights only")
        if self.block is not None and self.residual:
            raise ValueError("block scales and residual signs are separate prototypes")
        if any(type(layer) is not int or layer < 0 for layer in self.down_int4_layers):
            raise ValueError("4-bit down layers must be nonnegative layer indices")
        if (self.teacher is None) != (self.teacher_weight == 0.0) or not (
            math.isfinite(self.teacher_weight) and self.teacher_weight >= 0
        ):
            raise ValueError("a teacher needs a positive finite mixing weight")
        if self.teacher not in (None, "logits", "hidden"):
            raise ValueError("teacher target must be logits or hidden")

    def weights(self, layer: int, down: bool) -> Weights:
        if down and layer in self.down_int4_layers:
            return Weights("int4")
        return Weights(self.levels, self.block, self.residual)

    @property
    def servable(self) -> bool:
        return Weights(self.levels, self.block, self.residual).default and not self.down_int4_layers

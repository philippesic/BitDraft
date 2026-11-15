import math
from dataclasses import dataclass
from typing import Callable, cast

import torch
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx
from torch.nn import functional as functional

from w1a1_eagle.prototype import Levels, Prototype, Weights


@dataclass(frozen=True)
class Geometry:
    hidden: int = 2560
    intermediate: int = 9216
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    layers: int = 6
    slots: int = 16
    taps: int = 8
    window: int = 4096
    epsilon: float = 1e-6
    theta: float = 1e7

    def __post_init__(self) -> None:
        sizes = (
            self.hidden,
            self.intermediate,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.layers,
            self.slots,
            self.taps,
            self.window,
        )
        if any(type(v) is not int or v < 1 for v in sizes):
            raise ValueError("invalid geometry")
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("invalid grouped rotary geometry")
        if (
            not math.isfinite(self.epsilon)
            or self.epsilon <= 0
            or not math.isfinite(self.theta)
            or self.theta <= 0
        ):
            raise ValueError("invalid numerical constants")


def rms(value: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon) * weight


def rope(value: Tensor, positions: Tensor, theta: float) -> Tensor:
    width = value.shape[-1]
    frequency = torch.tensor(theta, dtype=torch.float32, device=value.device).pow(
        -torch.arange(0, width, 2, dtype=torch.float32, device=value.device) / width
    )
    angle = positions.float()[:, None] * frequency
    angle = torch.cat((angle, angle), -1)[:, None, :]
    half = width // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), -1)
    return value * angle.cos() + rotated * angle.sin()


def sign_forward(context: FunctionCtx, value: Tensor) -> Tensor:
    context.save_for_backward(value)
    return torch.where(value >= 0, 1.0, -1.0)


def sign_backward(context: FunctionCtx, gradient: Tensor) -> Tensor:
    (value,) = cast(tuple[Tensor], getattr(context, "saved_tensors"))
    return gradient * (value.abs() <= 1).to(gradient.dtype)


def autograd_function(
    name: str,
    forward: Callable[[FunctionCtx, Tensor], Tensor],
    backward: Callable[[FunctionCtx, Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    namespace: dict[str, object] = {
        "forward": staticmethod(forward),
        "backward": staticmethod(backward),
    }
    bases: tuple[type, ...] = (cast(type, getattr(torch.autograd, "Function")),)
    return cast(Callable[[Tensor], Tensor], getattr(type(name, bases, namespace), "apply"))


SIGN = autograd_function("Sign", sign_forward, sign_backward)


def hard_sign(value: Tensor) -> Tensor:
    if value.dtype != torch.float32:
        raise ValueError("latent signs must be F32")
    return SIGN(value)


TERNARY_THRESHOLD = 0.7 / 3
RESIDUAL_THRESHOLD = 1 / 3


def uniform_levels(value: Tensor, bits: int) -> Tensor:
    half = 1 << (bits - 1)
    top = (half - 0.5) / half
    return ((value * half).floor() + 0.5).div(half).clamp(-top, top)


def levels(value: Tensor, kind: Levels) -> Tensor:
    if kind == "sign":
        return torch.where(value >= 0, 1.0, -1.0)
    if kind == "ternary":
        return torch.where(value.abs() > TERNARY_THRESHOLD, torch.where(value >= 0, 1.0, -1.0), 0.0)
    if kind == "int2":
        return uniform_levels(value, 2)
    if kind == "int4":
        return uniform_levels(value, 4)
    return value


def residual_signs(value: Tensor) -> Tensor:
    return torch.where(value - RESIDUAL_THRESHOLD * levels(value, "sign") >= 0, 1.0, -1.0)


def effective_forward(
    context: FunctionCtx, latent: Tensor, scale: Tensor, residual: Tensor, block: Tensor
) -> Tensor:
    kind = cast(Levels, getattr(context, "levels"))
    context.save_for_backward(latent, scale, residual, block)
    quantized = levels(latent, kind)
    if block.numel():
        quantized = (quantized.view(block.shape[0], block.shape[1], -1) * block[:, :, None]).view(
            latent.shape
        )
    weight = quantized * scale[:, None]
    if residual.numel():
        weight = weight + residual_signs(latent) * residual[:, None]
    return weight


def effective_backward(
    context: FunctionCtx, gradient: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    kind = cast(Levels, getattr(context, "levels"))
    latent, scale, residual, block = cast(
        tuple[Tensor, Tensor, Tensor, Tensor], getattr(context, "saved_tensors")
    )
    inside = (latent.abs() <= 1).to(gradient.dtype)
    quantized = levels(latent, kind)
    if block.numel():
        rows, groups = block.shape
        expanded = block[:, :, None].expand(rows, groups, latent.shape[1] // groups)
        expanded = expanded.reshape(latent.shape)
        block_gradient = (gradient * quantized * scale[:, None]).view(rows, groups, -1).sum(-1)
        quantized = quantized * expanded
        latent_gradient = gradient * scale[:, None] * expanded * inside
    else:
        block_gradient = torch.zeros_like(block)
        latent_gradient = gradient * scale[:, None] * inside
    scale_gradient = (gradient * quantized).sum(-1)
    if residual.numel():
        offset = latent - RESIDUAL_THRESHOLD * levels(latent, "sign")
        latent_gradient = latent_gradient + gradient * residual[:, None] * (offset.abs() <= 1).to(
            gradient.dtype
        )
        residual_gradient = (gradient * residual_signs(latent)).sum(-1)
    else:
        residual_gradient = torch.zeros_like(residual)
    return latent_gradient, scale_gradient, residual_gradient, block_gradient


def effective_function(kind: Levels) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    def forward(
        context: FunctionCtx, latent: Tensor, scale: Tensor, residual: Tensor, block: Tensor
    ) -> Tensor:
        setattr(context, "levels", kind)
        return effective_forward(context, latent, scale, residual, block)

    def backward(context: FunctionCtx, gradient: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        setattr(context, "levels", kind)
        return effective_backward(context, gradient)

    namespace: dict[str, object] = {
        "forward": staticmethod(forward),
        "backward": staticmethod(backward),
    }
    bases: tuple[type, ...] = (cast(type, getattr(torch.autograd, "Function")),)
    function = type("Effective" + kind.capitalize(), bases, namespace)
    return cast(Callable[[Tensor, Tensor, Tensor, Tensor], Tensor], getattr(function, "apply"))


LEVELS: tuple[Levels, ...] = ("sign", "ternary", "int2", "int4", "full")
EFFECTIVE: dict[str, Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]] = {
    kind: effective_function(kind) for kind in LEVELS
}


def cache_storage(value: Tensor) -> Tensor:
    rounded = value.half().float()
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("F16 cache storage overflow")
    return value + (rounded - value).detach()


class Table(nn.Module):
    operand: Tensor

    def __init__(self, weight: Tensor, trained: bool = False) -> None:
        super().__init__()
        value = weight.detach().float().clone()
        if trained:
            self.operand = nn.Parameter(value)
        else:
            self.register_buffer("operand", value)

    def forward(self, value: Tensor) -> Tensor:
        return functional.linear(value, self.operand)


class Projection(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        bits: int,
        gain: nn.Parameter,
        clip: nn.Parameter,
        weights: Weights = Weights(),
    ) -> None:
        super().__init__()
        if bits not in (4, 8) or weight.ndim != 2 or not bool(torch.isfinite(weight).all()):
            raise ValueError("invalid HE4 projection")
        self.bits = bits
        self.weights = weights
        self.latent_sign = nn.Parameter(weight.detach().float().clone())
        self.scale = nn.Parameter(weight.detach().float().abs().mean(-1))
        self.midpoint = nn.Parameter(torch.zeros(weight.shape[0]))
        self.log_gain = gain
        self.log_clip = clip
        if not weights.default:
            self.initialize_prototype(weight.detach().float())

    def initialize_prototype(self, weight: Tensor) -> None:
        rows, columns = weight.shape
        magnitude = weight.abs()
        mean = magnitude.mean(-1, keepdim=True)
        bounded = (weight / torch.where(mean > 0, mean * 3, 1.0)).clamp(-1, 1)
        quantized = levels(bounded, self.weights.levels).abs()
        if self.weights.block is not None:
            if columns % self.weights.block:
                raise ValueError("block scale must divide the input width")
            groups = columns // self.weights.block
            block_mean = magnitude.view(rows, groups, -1).mean(-1)
            self.block_scale = nn.Parameter(
                torch.where(mean > 0, block_mean / torch.where(mean > 0, mean, 1.0), 1.0)
            )
            return
        numerator = (magnitude * quantized).sum(-1)
        denominator = quantized.square().sum(-1)
        with torch.no_grad():
            self.scale.copy_(
                torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30), 0.0)
            )
        if self.weights.residual:
            remainder = (magnitude - mean).abs().mean(-1)
            self.residual_scale = nn.Parameter(remainder)

    def effective_weight(self) -> Tensor:
        empty = self.latent_sign.new_zeros(0)
        residual = cast(Tensor, getattr(self, "residual_scale", empty))
        block = cast(Tensor, getattr(self, "block_scale", empty.view(0, 0)))
        return EFFECTIVE[self.weights.levels](
            self.latent_sign, self.scale.clamp_min(0), residual, block
        )

    def forward(self, value: Tensor) -> Tensor:
        gained = value.float() * self.log_gain.exp()
        if not bool(torch.isfinite(gained).all()):
            raise ValueError("nonfinite quantizer input")
        bound = self.log_clip.exp()
        if not bool(torch.isfinite(bound).all()) or not bool((bound > 0).all()):
            raise ValueError("invalid learned clipping bound")
        maximum = (1 << (self.bits - 1)) - 1
        clipped = torch.maximum(torch.minimum(gained, bound), -bound)
        normalized = clipped * (maximum / bound)
        rounded = normalized.round().clamp(-maximum, maximum)
        codes = normalized + (rounded - normalized).detach()
        step = bound / maximum
        if not self.weights.default:
            values = codes * step
            return functional.linear(values, self.effective_weight()) + (
                values.sum(-1, keepdim=True) * self.midpoint
            )
        signs = hard_sign(self.latent_sign)
        scale = self.scale.clamp_min(0)
        native = (functional.linear(codes, signs) * scale) * step + (
            codes.sum(-1, keepdim=True) * step
        ) * self.midpoint
        values = codes * step
        surrogate = (
            functional.linear(values, signs) * scale + values.sum(-1, keepdim=True) * self.midpoint
        )
        return native.detach() + (surrogate - surrogate.detach())


def bound_latents(model: nn.Module, multiple: float) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, Projection):
                latent = module.latent_sign
                mean = latent.abs().mean(-1, keepdim=True)
                bounded = (latent / torch.where(mean > 0, mean * multiple, 1.0)).clamp(-1, 1)
                if not torch.equal(bounded >= 0, latent >= 0):
                    raise ValueError("bounded latent scale changed a sign")
                latent.copy_(bounded)


def clamp_latents(model: nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, Projection):
                module.latent_sign.clamp_(-1, 1)


class TeacherLayer(nn.Module):
    attn_norm: Tensor
    q_norm: Tensor
    k_norm: Tensor
    ffn_norm: Tensor
    gate: Tensor
    up: Tensor
    down: Tensor

    def __init__(self, tensors: dict[str, Tensor], prefix: str) -> None:
        super().__init__()
        for name, key in (
            ("attn_norm", "attn_norm"),
            ("q_norm", "attn_q_norm"),
            ("k_norm", "attn_k_norm"),
            ("ffn_norm", "ffn_norm"),
        ):
            self.register_buffer(name, tensors[prefix + key + ".weight"].detach().float().clone())
        for name in ("gate", "up", "down"):
            weight = tensors[prefix + "ffn_" + name + ".weight"].detach()
            stored = weight.to(torch.bfloat16)
            if not torch.equal(stored.float(), weight.float()):
                raise ValueError("teacher feed-forward weights must be exact in BF16")
            self.register_buffer(name, stored)

    def feed(self, norm: Tensor) -> Tensor:
        gate = functional.linear(norm, self.gate.float())
        up = functional.linear(norm, self.up.float())
        return functional.linear(functional.silu(gate) * up, self.down.float())


class Teacher(nn.Module):
    encoder_norm: Tensor
    output_norm: Tensor

    def __init__(self, tensors: dict[str, Tensor], geometry: Geometry) -> None:
        super().__init__()
        self.register_buffer("encoder_norm", tensors["enc.output_norm.weight"].float().clone())
        self.register_buffer("output_norm", tensors["output_norm.weight"].float().clone())
        self.layers = nn.ModuleList(
            [TeacherLayer(tensors, f"blk.{i}.") for i in range(geometry.layers)]
        )


class Layer(nn.Module):
    def __init__(
        self,
        tensors: dict[str, Tensor],
        prefix: str,
        geometry: Geometry,
        trained: bool = False,
        gate_up: Weights = Weights(),
        down: Weights = Weights(),
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.attn_norm = nn.Parameter(tensors[prefix + "attn_norm.weight"].clone())
        self.q_norm = nn.Parameter(tensors[prefix + "attn_q_norm.weight"].clone())
        self.k_norm = nn.Parameter(tensors[prefix + "attn_k_norm.weight"].clone())
        self.ffn_norm = nn.Parameter(tensors[prefix + "ffn_norm.weight"].clone())
        self.query = Table(tensors[prefix + "attn_q.weight"], trained)
        self.key = Table(tensors[prefix + "attn_k.weight"], trained)
        self.value = Table(tensors[prefix + "attn_v.weight"], trained)
        self.output = Table(tensors[prefix + "attn_output.weight"], trained)
        gain = nn.Parameter(torch.zeros(geometry.hidden))
        clip = nn.Parameter(torch.tensor(0.0))
        self.gate = Projection(tensors[prefix + "ffn_gate.weight"], 4, gain, clip, gate_up)
        self.up = Projection(tensors[prefix + "ffn_up.weight"], 4, gain, clip, gate_up)
        self.down = Projection(
            tensors[prefix + "ffn_down.weight"],
            8,
            nn.Parameter(torch.zeros(geometry.intermediate)),
            nn.Parameter(torch.tensor(0.0)),
            down,
        )

    def context(
        self, encoded: Tensor, positions: Tensor, teacher: TeacherLayer | None = None
    ) -> tuple[Tensor, Tensor]:
        cfg = self.geometry
        key = self.key.forward(encoded).view(-1, cfg.kv_heads, cfg.head_dim)
        value = self.value.forward(encoded).view(-1, cfg.kv_heads, cfg.head_dim)
        k_norm = self.k_norm if teacher is None else teacher.k_norm
        key = rope(rms(key, k_norm, cfg.epsilon), positions, cfg.theta)
        return cache_storage(key), cache_storage(value)

    def forward(
        self, state: Tensor, context: tuple[Tensor, Tensor], positions: Tensor, allowed: Tensor
    ) -> Tensor:
        cfg = self.geometry
        norm = rms(state, self.attn_norm, cfg.epsilon)
        query = self.query.forward(norm).view(-1, cfg.heads, cfg.head_dim)
        key = self.key.forward(norm).view(-1, cfg.kv_heads, cfg.head_dim)
        value = self.value.forward(norm).view(-1, cfg.kv_heads, cfg.head_dim)
        query = rope(rms(query, self.q_norm, cfg.epsilon), positions, cfg.theta)
        key = rope(rms(key, self.k_norm, cfg.epsilon), positions, cfg.theta)
        key = torch.cat((context[0], cache_storage(key)), 0)
        value = torch.cat((context[1], cache_storage(value)), 0)
        slots = positions.shape[0]
        query = query.view(slots, cfg.kv_heads, cfg.heads // cfg.kv_heads, cfg.head_dim)
        scores = torch.einsum("qhgd,khd->hgqk", query, key) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(~allowed[None, None], -torch.inf)
        attended = torch.einsum("hgqk,khd->qhgd", functional.softmax(scores, -1), value)
        residual = state + self.output.forward(attended.reshape(slots, -1))
        norm = rms(residual, self.ffn_norm, cfg.epsilon)
        return residual + self.down.forward(
            functional.silu(self.gate.forward(norm)) * self.up.forward(norm)
        )

    def group(
        self,
        state: Tensor,
        context: tuple[Tensor, Tensor],
        positions: Tensor,
        allowed: Tensor,
        slot_allowed: Tensor | None,
        teacher: TeacherLayer | None = None,
    ) -> Tensor:
        cfg = self.geometry
        count, slots, length = positions.shape[0], positions.shape[1], context[0].shape[0]
        groups, heads = cfg.heads // cfg.kv_heads, cfg.kv_heads
        flat = positions.reshape(-1)
        norms = (
            (self.attn_norm, self.q_norm, self.k_norm, self.ffn_norm)
            if teacher is None
            else (teacher.attn_norm, teacher.q_norm, teacher.k_norm, teacher.ffn_norm)
        )
        norm = rms(state, norms[0], cfg.epsilon)
        query = self.query.forward(norm).view(-1, cfg.heads, cfg.head_dim)
        key = self.key.forward(norm).view(-1, heads, cfg.head_dim)
        value = self.value.forward(norm).view(-1, heads, cfg.head_dim)
        query = rope(rms(query, norms[1], cfg.epsilon), flat, cfg.theta)
        key = cache_storage(rope(rms(key, norms[2], cfg.epsilon), flat, cfg.theta))
        value = cache_storage(value)
        query = (
            query.view(count, slots, heads, groups, cfg.head_dim)
            .permute(2, 0, 3, 1, 4)
            .reshape(heads, count, groups * slots, cfg.head_dim)
        )
        scale = math.sqrt(cfg.head_dim)
        context_scores = torch.matmul(
            query.view(heads, count * groups * slots, cfg.head_dim), context[0].permute(1, 2, 0)
        ).view(heads, count, groups, slots, length)
        context_scores.div_(scale).masked_fill_(~allowed[None, :, None], -torch.inf)
        slot_scores = torch.matmul(
            query, key.view(count, slots, heads, cfg.head_dim).permute(2, 0, 3, 1)
        ).view(heads, count, groups, slots, slots)
        slot_scores.div_(scale)
        if slot_allowed is not None:
            slot_scores.masked_fill_(~slot_allowed, -torch.inf)
        weights = functional.softmax(torch.cat((context_scores, slot_scores), -1), -1)
        context_weights, slot_weights = weights.split_with_sizes([length, slots], -1)
        attended = torch.matmul(
            context_weights.view(heads, count * groups * slots, length),
            context[1].permute(1, 0, 2),
        ).view(heads, count, groups, slots, cfg.head_dim) + torch.matmul(
            slot_weights.view(heads, count, groups * slots, slots),
            value.view(count, slots, heads, cfg.head_dim).permute(2, 0, 1, 3),
        ).view(heads, count, groups, slots, cfg.head_dim)
        attended = attended.permute(1, 3, 0, 2, 4).reshape(count * slots, -1)
        residual = state + self.output.forward(attended)
        norm = rms(residual, norms[3], cfg.epsilon)
        if teacher is not None:
            return residual + teacher.feed(norm)
        return residual + self.down.forward(
            functional.silu(self.gate.forward(norm)) * self.up.forward(norm)
        )


class HE4(nn.Module):
    vocabulary: Tensor

    def __init__(
        self,
        tensors: dict[str, Tensor],
        vocabulary: Tensor,
        mask: int,
        geometry: Geometry = Geometry(),
        trained_tables: bool = False,
        prototype: Prototype = Prototype(),
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.trained_tables = trained_tables
        self.prototype = prototype
        self.mask = mask
        self.origin: tuple[str, str] | None = None
        self.frozen_format = "Q4_0"
        self.frozen_baseline: str | None = None
        self.parent_frozen_sha256: str | None = None
        self.native_types: tuple[tuple[str, str], ...] = ()
        self.native_codes: tuple[tuple[str, tuple[int, ...], bytes], ...] = ()
        if (
            vocabulary.ndim != 2
            or vocabulary.shape[1] != geometry.hidden
            or type(mask) is not int
            or not 0 <= mask < vocabulary.shape[0]
            or not bool(torch.isfinite(vocabulary).all())
        ):
            raise ValueError("invalid frozen vocabulary geometry or mask")
        self.register_buffer("vocabulary", vocabulary.detach().float().clone())
        self.fusion = Table(tensors["fc.weight"], trained_tables)
        self.encoder_norm = nn.Parameter(tensors["enc.output_norm.weight"].clone())
        self.output_norm = nn.Parameter(tensors["output_norm.weight"].clone())
        self.blocks = nn.ModuleList(
            [
                Layer(
                    tensors,
                    f"blk.{i}.",
                    geometry,
                    trained_tables,
                    prototype.weights(i, False),
                    prototype.weights(i, True),
                )
                for i in range(geometry.layers)
            ]
        )

    def check(self, features: Tensor, tokens: tuple[int, ...], anchors: tuple[int, ...]) -> None:
        cfg = self.geometry
        if (
            not anchors
            or any(type(anchor) is not int or not 0 <= anchor < len(tokens) for anchor in anchors)
            or features.ndim != 2
            or features.dtype not in (torch.float16, torch.float32)
            or features.shape != (len(tokens), cfg.taps * cfg.hidden)
            or any(
                type(token) is not int or not 0 <= token < self.vocabulary.shape[0]
                for token in tokens
            )
        ):
            raise ValueError("invalid model inputs")

    def block(self, slots: int | None) -> int:
        size = self.geometry.slots if slots is None else slots
        if type(size) is not int or not 1 <= size <= self.geometry.slots:
            raise ValueError("block size must be between one slot and the model's slots")
        return size

    def group_hidden(
        self,
        features: Tensor,
        tokens: tuple[int, ...],
        anchors: tuple[int, ...],
        slots: int | None = None,
        teacher: Teacher | None = None,
    ) -> Tensor:
        self.check(features, tokens, anchors)
        cfg = self.geometry
        size = self.block(slots)
        device = self.vocabulary.device
        starts = torch.tensor(anchors, device=device)
        offsets = torch.arange(size, device=device)
        positions = starts[:, None] + offsets[None, :]
        last = max(anchors)
        context_positions = torch.arange(last, device=device)
        identifiers = torch.full((len(anchors), size), self.mask, device=device)
        first: list[int] = [tokens[anchor] for anchor in anchors]
        identifiers[:, 0] = torch.tensor(first, device=device)
        state = functional.embedding(identifiers.reshape(-1), self.vocabulary)
        encoded = rms(
            self.fusion.forward(features[:last].to(device).float()),
            self.encoder_norm if teacher is None else teacher.encoder_norm,
            cfg.epsilon,
        )
        visible = context_positions[None, None, :] < starts[:, None, None]
        local = visible & (
            (positions[:, :, None] - context_positions[None, None, :]).abs() < cfg.window
        )
        slot_local = (offsets[:, None] - offsets[None, :]).abs() < cfg.window
        for index, module in enumerate(self.blocks):
            block = cast(Layer, module)
            final = index == cfg.layers - 1
            layer = None if teacher is None else cast(TeacherLayer, teacher.layers[index])
            state = block.group(
                state,
                block.context(encoded, context_positions, layer),
                positions,
                visible if final else local,
                None if final else slot_local,
                layer,
            )
        output_norm = self.output_norm if teacher is None else teacher.output_norm
        return rms(state, output_norm, cfg.epsilon).view(len(anchors), size, cfg.hidden)

    def forward(
        self, features: Tensor, tokens: tuple[int, ...], anchor: int, slots: int | None = None
    ) -> Tensor:
        cfg = self.geometry
        if type(anchor) is not int:
            raise ValueError("invalid model inputs")
        self.check(features, tokens, (anchor,))
        size = self.block(slots)
        device = self.vocabulary.device
        positions = torch.arange(anchor, anchor + size, device=device)
        context_positions = torch.arange(anchor, device=device)
        identifiers = torch.tensor((tokens[anchor],) + (self.mask,) * (size - 1), device=device)
        state = functional.embedding(identifiers, self.vocabulary)
        encoded = rms(
            self.fusion.forward(features[:anchor].to(device).float()),
            self.encoder_norm,
            cfg.epsilon,
        )
        keys = torch.cat((context_positions, positions))
        for index, module in enumerate(self.blocks):
            block = cast(Layer, module)
            allowed = torch.ones((size, anchor + size), dtype=torch.bool, device=device)
            if index < cfg.layers - 1:
                allowed &= (positions[:, None] - keys[None, :]).abs() < cfg.window
            state = block.forward(
                state, block.context(encoded, context_positions), positions, allowed
            )
        return functional.linear(rms(state, self.output_norm, cfg.epsilon), self.vocabulary)


class TeacherView:
    def __init__(self, student: HE4, teacher: Teacher) -> None:
        self.student = student
        self.teacher = teacher

    @property
    def vocabulary(self) -> Tensor:
        return self.student.vocabulary

    def block(self, slots: int | None) -> int:
        return self.student.block(slots)

    def forward(
        self, features: Tensor, tokens: tuple[int, ...], anchor: int, slots: int | None = None
    ) -> Tensor:
        hidden = self.student.group_hidden(features, tokens, (anchor,), slots, self.teacher)
        return functional.linear(hidden[0], self.student.vocabulary)

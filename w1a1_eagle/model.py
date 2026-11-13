import math
from dataclasses import dataclass
from typing import Callable, cast

import torch
from torch import Tensor, nn
from torch.autograd.function import FunctionCtx
from torch.nn import functional as functional


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


def cache_storage(value: Tensor) -> Tensor:
    rounded = value.half().float()
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError("F16 cache storage overflow")
    return value + (rounded - value).detach()


class FrozenLinear(nn.Module):
    operand: Tensor

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.register_buffer("operand", weight.detach().float().clone())

    def forward(self, value: Tensor) -> Tensor:
        return functional.linear(value, self.operand)


class Projection(nn.Module):
    def __init__(self, weight: Tensor, bits: int, gain: nn.Parameter, clip: nn.Parameter) -> None:
        super().__init__()
        if bits not in (4, 8) or weight.ndim != 2 or not bool(torch.isfinite(weight).all()):
            raise ValueError("invalid HE4 projection")
        self.bits = bits
        self.latent_sign = nn.Parameter(weight.detach().float().clone())
        self.scale = nn.Parameter(weight.detach().float().abs().mean(-1))
        self.midpoint = nn.Parameter(torch.zeros(weight.shape[0]))
        self.log_gain = gain
        self.log_clip = clip

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


class Layer(nn.Module):
    def __init__(self, tensors: dict[str, Tensor], prefix: str, geometry: Geometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.attn_norm = nn.Parameter(tensors[prefix + "attn_norm.weight"].clone())
        self.q_norm = nn.Parameter(tensors[prefix + "attn_q_norm.weight"].clone())
        self.k_norm = nn.Parameter(tensors[prefix + "attn_k_norm.weight"].clone())
        self.ffn_norm = nn.Parameter(tensors[prefix + "ffn_norm.weight"].clone())
        self.query = FrozenLinear(tensors[prefix + "attn_q.weight"])
        self.key = FrozenLinear(tensors[prefix + "attn_k.weight"])
        self.value = FrozenLinear(tensors[prefix + "attn_v.weight"])
        self.output = FrozenLinear(tensors[prefix + "attn_output.weight"])
        gain = nn.Parameter(torch.zeros(geometry.hidden))
        clip = nn.Parameter(torch.tensor(0.0))
        self.gate = Projection(tensors[prefix + "ffn_gate.weight"], 4, gain, clip)
        self.up = Projection(tensors[prefix + "ffn_up.weight"], 4, gain, clip)
        self.down = Projection(
            tensors[prefix + "ffn_down.weight"],
            8,
            nn.Parameter(torch.zeros(geometry.intermediate)),
            nn.Parameter(torch.tensor(0.0)),
        )

    def context(self, encoded: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.geometry
        key = self.key.forward(encoded).view(-1, cfg.kv_heads, cfg.head_dim)
        value = self.value.forward(encoded).view(-1, cfg.kv_heads, cfg.head_dim)
        key = rope(rms(key, self.k_norm, cfg.epsilon), positions, cfg.theta)
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
        query = query.view(cfg.slots, cfg.kv_heads, cfg.heads // cfg.kv_heads, cfg.head_dim)
        scores = torch.einsum("qhgd,khd->hgqk", query, key) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(~allowed[None, None], -torch.inf)
        attended = torch.einsum("hgqk,khd->qhgd", functional.softmax(scores, -1), value)
        residual = state + self.output.forward(attended.reshape(cfg.slots, -1))
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
    ) -> Tensor:
        cfg = self.geometry
        count, slots, length = positions.shape[0], cfg.slots, context[0].shape[0]
        groups, heads = cfg.heads // cfg.kv_heads, cfg.kv_heads
        flat = positions.reshape(-1)
        norm = rms(state, self.attn_norm, cfg.epsilon)
        query = self.query.forward(norm).view(-1, cfg.heads, cfg.head_dim)
        key = self.key.forward(norm).view(-1, heads, cfg.head_dim)
        value = self.value.forward(norm).view(-1, heads, cfg.head_dim)
        query = rope(rms(query, self.q_norm, cfg.epsilon), flat, cfg.theta)
        key = cache_storage(rope(rms(key, self.k_norm, cfg.epsilon), flat, cfg.theta))
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
        norm = rms(residual, self.ffn_norm, cfg.epsilon)
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
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.mask = mask
        self.origin: tuple[str, str] | None = None
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
        self.fusion = FrozenLinear(tensors["fc.weight"])
        self.encoder_norm = nn.Parameter(tensors["enc.output_norm.weight"].clone())
        self.output_norm = nn.Parameter(tensors["output_norm.weight"].clone())
        self.blocks = nn.ModuleList(
            [Layer(tensors, f"blk.{i}.", geometry) for i in range(geometry.layers)]
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

    def group_hidden(
        self, features: Tensor, tokens: tuple[int, ...], anchors: tuple[int, ...]
    ) -> Tensor:
        self.check(features, tokens, anchors)
        cfg = self.geometry
        device = self.vocabulary.device
        starts = torch.tensor(anchors, device=device)
        offsets = torch.arange(cfg.slots, device=device)
        positions = starts[:, None] + offsets[None, :]
        last = max(anchors)
        context_positions = torch.arange(last, device=device)
        identifiers = torch.full((len(anchors), cfg.slots), self.mask, device=device)
        first: list[int] = [tokens[anchor] for anchor in anchors]
        identifiers[:, 0] = torch.tensor(first, device=device)
        state = functional.embedding(identifiers.reshape(-1), self.vocabulary)
        encoded = rms(
            self.fusion.forward(features[:last].to(device).float()), self.encoder_norm, cfg.epsilon
        )
        visible = context_positions[None, None, :] < starts[:, None, None]
        local = visible & (
            (positions[:, :, None] - context_positions[None, None, :]).abs() < cfg.window
        )
        slot_local = (offsets[:, None] - offsets[None, :]).abs() < cfg.window
        for index, module in enumerate(self.blocks):
            block = cast(Layer, module)
            final = index == cfg.layers - 1
            state = block.group(
                state,
                block.context(encoded, context_positions),
                positions,
                visible if final else local,
                None if final else slot_local,
            )
        return rms(state, self.output_norm, cfg.epsilon).view(len(anchors), cfg.slots, cfg.hidden)

    def forward(self, features: Tensor, tokens: tuple[int, ...], anchor: int) -> Tensor:
        cfg = self.geometry
        if type(anchor) is not int:
            raise ValueError("invalid model inputs")
        self.check(features, tokens, (anchor,))
        device = self.vocabulary.device
        positions = torch.arange(anchor, anchor + cfg.slots, device=device)
        context_positions = torch.arange(anchor, device=device)
        identifiers = torch.tensor(
            (tokens[anchor],) + (self.mask,) * (cfg.slots - 1), device=device
        )
        state = functional.embedding(identifiers, self.vocabulary)
        encoded = rms(
            self.fusion.forward(features[:anchor].to(device).float()),
            self.encoder_norm,
            cfg.epsilon,
        )
        keys = torch.cat((context_positions, positions))
        for index, module in enumerate(self.blocks):
            block = cast(Layer, module)
            allowed = torch.ones((cfg.slots, anchor + cfg.slots), dtype=torch.bool, device=device)
            if index < cfg.layers - 1:
                allowed &= (positions[:, None] - keys[None, :]).abs() < cfg.window
            state = block.forward(
                state, block.context(encoded, context_positions), positions, allowed
            )
        return functional.linear(rms(state, self.output_norm, cfg.epsilon), self.vocabulary)

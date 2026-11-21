import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, cast

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
    frequency = torch.full((), theta, dtype=torch.float32, device=value.device).pow(
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
    return gradient * (value.abs() <= 1)


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


def soft_ratio(latent: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    positive = (scale > 0)[:, None]
    return latent / torch.where(positive, scale[:, None], 1.0), positive


def soft_forward(context: FunctionCtx, latent: Tensor, scale: Tensor, temperature: float) -> Tensor:
    context.save_for_backward(latent, scale)
    setattr(context, "temperature", temperature)
    ratio, positive = soft_ratio(latent, scale)
    soft = scale[:, None] * torch.tanh(temperature * ratio) / math.tanh(temperature)
    return torch.where(positive, soft, 0.0)


def soft_backward(context: FunctionCtx, gradient: Tensor) -> tuple[Tensor, Tensor, None]:
    latent, scale = cast(tuple[Tensor, Tensor], getattr(context, "saved_tensors"))
    temperature = cast(float, getattr(context, "temperature"))
    ratio, positive = soft_ratio(latent, scale)
    argument = temperature * ratio
    hyperbolic = torch.tanh(argument)
    slope = 1 - hyperbolic.square()
    normalizer = math.tanh(temperature)
    latent_gradient = torch.where(positive, gradient * (temperature * slope / normalizer), 0.0)
    scale_gradient = torch.where(
        positive, gradient * ((hyperbolic - argument * slope) / normalizer), 0.0
    ).sum(-1)
    return latent_gradient, scale_gradient, None


SOFT_NAMESPACE: dict[str, object] = {
    "forward": staticmethod(soft_forward),
    "backward": staticmethod(soft_backward),
}
SOFT = cast(
    Callable[[Tensor, Tensor, float], Tensor],
    getattr(
        type("SoftSign", (cast(type, getattr(torch.autograd, "Function")),), SOFT_NAMESPACE),
        "apply",
    ),
)


def soft_weight(latent: Tensor, scale: Tensor, temperature: float) -> Tensor:
    if (
        latent.ndim != 2
        or scale.shape != latent.shape[:1]
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("soft weights need a matrix, one scale per row and a positive temperature")
    return SOFT(latent, scale, temperature)


def hard_sign(value: Tensor) -> Tensor:
    if value.dtype != torch.float32:
        raise ValueError("latent signs must be F32")
    return SIGN(value)


class DeferredChecks:
    def __init__(self) -> None:
        self.failed: list[Tensor] = []
        self.messages: list[str] = []

    def require(self, valid: Tensor, message: str) -> None:
        self.failed.append(valid.detach().logical_not().reshape(1))
        self.messages.append(message)

    def read(self, *values: Tensor) -> tuple[float, ...]:
        parts = [torch.cat(self.failed).double()] if self.failed else []
        parts.extend(value.detach().double().reshape(1) for value in values)
        measured = cast(list[float], torch.cat(parts).tolist())
        for failed, message in zip(measured, self.messages):
            if failed:
                raise ValueError(message)
        return tuple(measured[len(self.failed) :])


def require(valid: Tensor, message: str, checks: DeferredChecks | None) -> None:
    if checks is not None:
        checks.require(valid, message)
    elif not bool(valid):
        raise ValueError(message)


@contextmanager
def tensor_cores(enabled: bool) -> Iterator[None]:
    matmul = cast(object, torch.backends.cuda.matmul)
    previous = cast(bool, getattr(matmul, "allow_tf32"))
    setattr(matmul, "allow_tf32", enabled or previous)
    try:
        yield
    finally:
        setattr(matmul, "allow_tf32", previous)


def cache_storage(value: Tensor, checks: DeferredChecks | None = None) -> Tensor:
    rounded = value.half().float()
    require(torch.isfinite(rounded).all(), "F16 cache storage overflow", checks)
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


BYPASS_CHANNELS = (0, 850)


class Projection(nn.Module):
    bypass_columns: Tensor | None
    bypass_weights: Tensor | None

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
        self.temperature: float | None = None
        self.statistics: Tensor | None = None
        self.observed = 0
        self.recorded: list[Tensor] | None = None
        self.checks: DeferredChecks | None = None
        self.single_product = False
        self.tensor_cores = False
        self.register_buffer("bypass_columns", None, persistent=False)
        self.register_buffer("bypass_weights", None, persistent=False)

    def enable_bypass(self) -> None:
        if self.bits != 4 or self.latent_sign.shape[-1] <= max(BYPASS_CHANNELS):
            raise ValueError("two-channel bypass requires the 4-bit gate/up input geometry")
        self.bypass_columns = torch.tensor(BYPASS_CHANNELS, device=self.latent_sign.device)
        self.bypass_weights = self.latent_sign.detach().index_select(-1, self.bypass_columns)

    def quantizer_input(self, value: Tensor) -> Tensor:
        columns = self.bypass_columns
        return value if columns is None else value.index_fill(-1, columns, 0)

    @property
    def maximum(self) -> int:
        return (1 << (self.bits - 1)) - 1

    def quantize(self, value: Tensor) -> tuple[Tensor, Tensor]:
        gained = value.float() * self.log_gain.exp()
        require(torch.isfinite(gained).all(), "nonfinite quantizer input", self.checks)
        bound = self.log_clip.exp()
        require(
            torch.isfinite(bound).all() & (bound > 0).all(),
            "invalid learned clipping bound",
            self.checks,
        )
        maximum = self.maximum
        clipped = torch.maximum(torch.minimum(gained, bound), -bound)
        normalized = clipped * (maximum / bound)
        rounded = normalized.round().clamp(-maximum, maximum)
        if self.statistics is not None:
            with torch.no_grad():
                counts = torch.stack(((gained.abs() > bound).sum(), (rounded == 0).sum()))
                self.statistics.add_(counts.to(self.statistics.dtype))
                self.observed += gained.numel()
        codes = normalized + (rounded - normalized).detach()
        return codes, bound / maximum

    def forward(self, value: Tensor) -> Tensor:
        if self.recorded is not None:
            self.recorded.append(value.detach())
        result = self.project(self.quantizer_input(value))
        columns = self.bypass_columns
        weights = self.bypass_weights
        if columns is not None and weights is not None:
            result = result + functional.linear(value.index_select(-1, columns).float(), weights)
        return result

    def project(self, value: Tensor) -> Tensor:
        codes, step = self.quantize(value)
        scale = self.scale.clamp_min(0)
        if self.temperature is not None:
            values = codes * step
            weight = soft_weight(self.latent_sign, scale, self.temperature)
            return functional.linear(values, weight) + values.sum(-1, keepdim=True) * self.midpoint
        signs = hard_sign(self.latent_sign)
        if self.single_product:
            with tensor_cores(self.tensor_cores):
                product = functional.linear(codes, signs)
            return (product * scale) * step + (codes.sum(-1, keepdim=True) * step) * self.midpoint
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


class Layer(nn.Module):
    def __init__(
        self, tensors: dict[str, Tensor], prefix: str, geometry: Geometry, trained: bool = False
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.checks: DeferredChecks | None = None
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
        return cache_storage(key, self.checks), cache_storage(value, self.checks)

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
        key = torch.cat((context[0], cache_storage(key, self.checks)), 0)
        value = torch.cat((context[1], cache_storage(value, self.checks)), 0)
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
    ) -> Tensor:
        cfg = self.geometry
        count, slots, length = positions.shape[0], positions.shape[1], context[0].shape[0]
        groups, heads = cfg.heads // cfg.kv_heads, cfg.kv_heads
        flat = positions.reshape(-1)
        norm = rms(state, self.attn_norm, cfg.epsilon)
        query = self.query.forward(norm).view(-1, cfg.heads, cfg.head_dim)
        key = self.key.forward(norm).view(-1, heads, cfg.head_dim)
        value = self.value.forward(norm).view(-1, heads, cfg.head_dim)
        query = rope(rms(query, self.q_norm, cfg.epsilon), flat, cfg.theta)
        key = cache_storage(rope(rms(key, self.k_norm, cfg.epsilon), flat, cfg.theta), self.checks)
        value = cache_storage(value, self.checks)
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
        trained_tables: bool = False,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.trained_tables = trained_tables
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
            [Layer(tensors, f"blk.{i}.", geometry, trained_tables) for i in range(geometry.layers)]
        )

    def check(
        self,
        features: Tensor,
        tokens: tuple[int, ...],
        anchors: tuple[int, ...],
        fused: bool = False,
    ) -> None:
        cfg = self.geometry
        if fused and self.trained_tables:
            raise ValueError("fused inputs require a frozen fusion table")
        if (
            not anchors
            or any(type(anchor) is not int or not 0 <= anchor < len(tokens) for anchor in anchors)
            or features.ndim != 2
            or features.dtype not in (torch.float16, torch.float32)
            or features.shape != (len(tokens), cfg.hidden if fused else cfg.taps * cfg.hidden)
            or any(
                type(token) is not int or not 0 <= token < self.vocabulary.shape[0]
                for token in tokens
            )
        ):
            raise ValueError("invalid model inputs")

    def encode(self, features: Tensor, count: int, fused: bool) -> Tensor:
        values = features[:count].to(self.vocabulary.device).float()
        return rms(
            values if fused else self.fusion.forward(values),
            self.encoder_norm,
            self.geometry.epsilon,
        )

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
        fused: bool = False,
        contexts: tuple[tuple[Tensor, Tensor], ...] | None = None,
    ) -> Tensor:
        self.check(features, tokens, anchors, fused)
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
        if contexts is not None and (
            len(contexts) != cfg.layers or any(part[0].shape[0] < last for part in contexts)
        ):
            raise ValueError("precomputed contexts do not cover the anchors")
        given = () if contexts is None else contexts
        encoded = self.encode(features, last, fused) if contexts is None else None
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
                block.context(encoded, context_positions)
                if encoded is not None
                else (given[index][0][:last], given[index][1][:last]),
                positions,
                visible if final else local,
                None if final else slot_local,
            )
        return rms(state, self.output_norm, cfg.epsilon).view(len(anchors), size, cfg.hidden)

    def group_contexts(
        self, features: Tensor, tokens: tuple[int, ...], count: int, fused: bool = False
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        self.check(features, tokens, (count - 1,), fused)
        encoded = self.encode(features, count, fused)
        positions = torch.arange(count, device=self.vocabulary.device)
        return tuple(cast(Layer, module).context(encoded, positions) for module in self.blocks)

    def forward(
        self,
        features: Tensor,
        tokens: tuple[int, ...],
        anchor: int,
        slots: int | None = None,
        fused: bool = False,
    ) -> Tensor:
        cfg = self.geometry
        if type(anchor) is not int:
            raise ValueError("invalid model inputs")
        self.check(features, tokens, (anchor,), fused)
        size = self.block(slots)
        device = self.vocabulary.device
        positions = torch.arange(anchor, anchor + size, device=device)
        context_positions = torch.arange(anchor, device=device)
        identifiers = torch.tensor((tokens[anchor],) + (self.mask,) * (size - 1), device=device)
        state = functional.embedding(identifiers, self.vocabulary)
        encoded = self.encode(features, anchor, fused)
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


def enable_gate_up_bypass(model: HE4) -> None:
    for _, projection in named_projections(model):
        if projection.bits == 4:
            projection.enable_bypass()


def named_projections(model: HE4) -> list[tuple[str, Projection]]:
    result: list[tuple[str, Projection]] = []
    for index, module in enumerate(model.blocks):
        block = cast(Layer, module)
        for part, projection in (("gate", block.gate), ("up", block.up), ("down", block.down)):
            result.append((f"blocks.{index}.{part}", projection))
    return result

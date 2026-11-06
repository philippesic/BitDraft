"""Bounded binary research adapters; capacity forks are not the native W1 codec."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from w1a1_eagle.block_qat import BlockBinaryLinear
from w1a1_eagle.recurrent_binary import hard_sign_ste
from w1a1_eagle.recurrent_qat import hard_activation, fixed_activation_codes


def signs(module):
    return (hard_sign_ste(module.latent_sign) if module._round_hard_signs is None
            else module._round_hard_signs)


def a8_dot(x, binary):
    """Native whole-row A8 integer-dot order with the existing activation STE."""
    values, _, _ = hard_activation(x, 8)
    surrogate = F.linear(values, binary)
    with torch.no_grad():
        codes, beta, _ = fixed_activation_codes(x.float(), 8)
        native = F.linear(codes, binary.detach()) * beta
    return native.detach() + (surrogate - surrogate.detach())


class SelectiveGradient:
    """Backward-only ParetoQ-inspired scale-window + quantization-error trust.

    This is a transplant, not full ParetoQ or QuEST. Keep the existing latent
    parameterization/forward. Hooks act before the trainer's global norm clip.
    The row-relative window uses fitted effective scales; the trust term is
    1/(1+|latent-sign(latent)*scale|/scale), with an explicit nonzero floor.
    """
    def __init__(self, linears, window=1.0, trust_floor=0.1, mode="combined"):
        if not math.isfinite(window) or window <= 0 or not 0 <= trust_floor <= 1:
            raise ValueError("positive window and trust floor in [0,1] required")
        if mode not in {"combined", "elastic", "trust"}:
            raise ValueError("unknown selective gradient mode")
        self.handles = []
        self.stats = {}
        for name, module in linears.items():
            def hook(grad, name=name, module=module):
                with torch.no_grad():
                    alpha = module.effective_scales().detach()[:, None].clamp_min(1e-12)
                    latent = module.latent_sign.detach()
                    binary = torch.where(latent < 0, -1.0, 1.0)
                    error = (latent - binary * alpha).abs() / alpha
                    trust = (1 / (1 + error)).clamp_min(trust_floor)
                    multiplier = (latent.abs() < window * alpha).to(grad.dtype)
                    if mode == "trust":
                        multiplier = torch.ones_like(multiplier)
                    if mode != "elastic":
                        multiplier = multiplier * trust
                    transformed = grad * multiplier
                    self.stats[name] = {
                        "window_active_fraction": float((multiplier > 0).float().mean()),
                        "mean_multiplier": float(multiplier.mean()),
                        "input_norm": float(grad.norm()),
                        "output_norm": float(transformed.norm()),
                        "cosine": float(F.cosine_similarity(
                            grad.reshape(1, -1), transformed.reshape(1, -1), eps=1e-30)),
                    }
                    return transformed
            self.handles.append(module.latent_sign.register_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class MixtureScales(BlockBinaryLinear):
    """Two scale experts sharing one hard binary matrix, BinaryMoS Eq.3-5.

    Router sees unquantized token; input scale is applied BEFORE A8 packing.
    Thus gate/up inputs generally require separate new packs. The exact
    integer-dot scale order is preserved within this new operator.
    """
    def __init__(self, parent, *, static=False, seed=8101, perturbation=0.005):
        super().__init__(parent.latent_sign, parent.initial_scale, parent.contract)
        self.load_state_dict(parent.state_dict())
        self.static = static
        generator = torch.Generator(device=parent.latent_sign.device).manual_seed(seed)
        self.router = nn.Parameter(torch.randn(
            self.in_features, 2, generator=generator,
            device=parent.latent_sign.device) / math.sqrt(self.in_features) * 0.01)
        # Opposite expert perturbations avoid permanently identical experts.
        din = torch.randn(self.in_features, generator=generator, device=self.router.device)
        dout = torch.randn(self.out_features, generator=generator, device=self.router.device)
        self.input_experts = nn.Parameter(torch.stack([1 + perturbation * din,
                                                     1 - perturbation * din]))
        self.output_expert_offsets = nn.Parameter(torch.stack([perturbation * dout,
                                                              -perturbation * dout]))

    def forward(self, x):
        if self.static:
            logits = self.router.mean(0).expand(*x.shape[:-1], 2)
        else:
            logits = x.float() @ self.router
        gate = logits.softmax(-1)
        inscale = gate @ self.input_experts
        outscale = self.effective_scales() * (1 + gate @ self.output_expert_offsets)
        scaled = x.float() * inscale
        # Multiplication order matches native A8: integer dot * weight scale * beta.
        binary = signs(self)
        values, _, _ = hard_activation(scaled, 8)
        surrogate = F.linear(values, binary) * outscale
        with torch.no_grad():
            codes, beta, _ = fixed_activation_codes(scaled, 8)
            native = (F.linear(codes, binary.detach()) * outscale.detach()) * beta
        return native.detach() + (surrogate - surrogate.detach())

    def cost(self):
        m, n = self.out_features, self.in_features
        floats = 2 * n + 2 * n + 2 * m + m
        return {"binary_bits": m*n, "float32_parameters": floats,
                "effective_bits_per_original_weight": (m*n + 32*floats)/(m*n),
                "router_macs_per_token": 0 if self.static else 2*n,
                "mixture_macs_per_token": 2*(n+m),
                "extra_scale_multiplies_per_token": n+m,
                "a8_packs_per_call": 1, "shared_gate_up_pack": False,
                "representation": "shared binary + F32 router/input/output experts"}


class DoubleBinary(BlockBinaryLinear):
    """NanoQuant Eq.1 double-binary factors, bounded alternating-sign initializer.

    Uses source-parent reconstruction, not NanoQuant Hessian-aware ADMM or AF1
    allocation. Input is native-style A8; intermediate and both signed matmuls
    use F32 arithmetic. No claim of current native-codec compatibility.
    """
    def __init__(self, parent, *, rank=32, sweeps=2, ridge=1e-3):
        if type(rank) is not int or not 1 <= rank <= min(parent.in_features, parent.out_features):
            raise ValueError("factor rank outside matrix dimensions")
        if type(sweeps) is not int or not 0 <= sweeps <= 4:
            raise ValueError("bounded initializer requires 0..4 sweeps")
        super().__init__(parent.latent_sign, parent.effective_scales().detach(), parent.contract)
        self.rank = rank
        with torch.no_grad():
            target = torch.where(parent.latent_sign < 0, -1.0, 1.0)
            # Normalize away parent row scales, fit binary correlation structure.
            columns = torch.linspace(0, parent.in_features - 1, rank,
                                     device=target.device).long()
            u = target[:, columns].clone()
            eye = torch.eye(rank, device=target.device) * ridge
            v = torch.linalg.solve(u.T @ u + eye, u.T @ target)
            v = torch.where(v < 0, -1.0, 1.0)
            for _ in range(sweeps):
                u = torch.linalg.solve(v @ v.T + eye, v @ target.T).T
                u = torch.where(u < 0, -1.0, 1.0)
                v = torch.linalg.solve(u.T @ u + eye, u.T @ target)
                v = torch.where(v < 0, -1.0, 1.0)
            # Per-output least squares scale; no giant dense UV materialization.
            numerator = ((target @ v.T) * u).sum(-1)
            denominator = ((u @ (v @ v.T)) * u).sum(-1).clamp_min(1e-12)
            fitted = numerator / denominator
            # Put negative row signs into U to keep the legacy nonnegative scale API.
            u *= torch.where(fitted[:, None] < 0, -1.0, 1.0)
            self.initial_scale.copy_(parent.effective_scales().detach() * fitted.abs())
        # Smaller latent magnitudes retain the existing clipped-STE gradient.
        self.latent_sign = nn.Parameter(v * 0.5)
        self.factor_out = nn.Parameter(u * 0.5)
        self.input_scale = nn.Parameter(torch.ones(parent.in_features, device=v.device))

    def forward(self, x):
        intermediate = a8_dot(x.float() * self.input_scale, signs(self))
        result = F.linear(intermediate, hard_sign_ste(self.factor_out))
        return result * self.effective_scales()

    @torch.no_grad()
    def project_scales_(self):
        super().project_scales_()
        self.factor_out.clamp_(-1, 1)

    def cost(self):
        m, n, r = self.out_features, self.in_features, self.rank
        bits, floats = r*(m+n), m+n
        return {"binary_bits": bits, "float32_parameters": floats,
                "rank": r, "total_storage_bits": bits+32*floats, "original_weight_count": m*n,
                "effective_bits_per_original_weight": (bits+32*floats)/(m*n),
                "factor_macs_per_token": r*(m+n),
                "intermediate_f32_elements_per_token": r,
                "a8_packs_per_call": 1, "intermediate_a8_packs": 0,
                "representation": "two binary factors; F32 scales/intermediate/matmuls"}


class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, latent, threshold):
        ctx.save_for_backward(latent)
        return torch.where(latent.abs() < threshold, 0., torch.where(latent < 0, -1., 1.))

    @staticmethod
    def backward(ctx, gradient):
        latent, = ctx.saved_tensors
        return gradient * (latent.abs() <= 1), None


class TernaryA8(BlockBinaryLinear):
    """Separate W2A8 frontier: fixed row deadzone, ternary integer-dot forward.

    A bounded representative, not a reproduction of pretrained BitNet. Threshold
    is 0.7*mean(abs(parent latent)) per row, fixed after initialization. Two bits
    store each {-1,0,+1} code. Existing native W1 export cannot encode this.
    """
    def __init__(self, parent):
        super().__init__(parent.latent_sign, parent.initial_scale, parent.contract)
        self.load_state_dict(parent.state_dict())
        self.register_buffer("threshold", .7*self.latent_sign.detach().abs().mean(-1, keepdim=True))

    def codes(self):
        return _TernarySTE.apply(self.latent_sign, self.threshold)

    def forward(self, x):
        ternary = self.codes()
        values, _, _ = hard_activation(x, 8)
        surrogate = F.linear(values, ternary) * self.effective_scales()
        with torch.no_grad():
            codes, beta, _ = fixed_activation_codes(x.float(), 8)
            native = (F.linear(codes, ternary.detach()) * self.effective_scales()) * beta
        return native.detach() + (surrogate-surrogate.detach())

    def cost(self):
        m, n = self.out_features, self.in_features
        return {"weight_bits": 2*m*n, "float32_parameters": m,
                "effective_bits_per_original_weight": 2+32/n,
                "training_threshold_f32_values": m,
                "a8_packs_per_call": 1,
                "representation": "2-bit ternary codes + F32 row scales; F32 arithmetic reference"}


def install(model, technique, *, names=None, rank=32, sweeps=2):
    inventory = model.binary_linears()
    selected = sorted(inventory if names is None else names)
    if not selected or not set(selected) <= set(inventory):
        raise ValueError("selected names must be a nonempty subset of original binary projections")
    if technique in {"control", "selective-gradient", "elastic-gradient", "trust-gradient",
                     "calibration-a8", "calibration-weight-mse"}:
        return {}
    if technique not in {"mos", "mos-static", "double-binary", "ternary-a8"}:
        raise ValueError("unknown isolated technique")
    costs = {}
    for name in selected:
        parent = inventory[name]
        if technique == "double-binary":
            module = DoubleBinary(parent, rank=rank, sweeps=sweeps)
        elif technique == "ternary-a8":
            module = TernaryA8(parent)
        else:
            module = MixtureScales(parent, static=technique == "mos-static")
        if name == "fc":
            model.fc = module
        else:
            _, index, projection = name.split(".")
            setattr(model.layers[int(index)], projection, module)
        costs[name] = module.cost()
    return costs


def optimizer_for(model):
    """Keep original family LRs; put every added parameter under explicit ownership."""
    families = {"sign": [], "scale": []}
    for module in model.binary_linears().values():
        for name, parameter in module.named_parameters():
            family = "sign" if name in {"latent_sign", "factor_out"} else "scale"
            families[family].append(parameter)
    return torch.optim.AdamW([
        {"params": families["sign"], "lr": model.config.sign_lr, "family": "sign"},
        {"params": families["scale"], "lr": model.config.scale_lr, "family": "scale"},
    ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False)

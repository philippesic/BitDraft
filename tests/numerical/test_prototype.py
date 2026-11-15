import unittest

import torch
from torch import nn

from w1a1_eagle.model import (
    EFFECTIVE,
    HE4,
    RESIDUAL_THRESHOLD,
    Geometry,
    Projection,
    Teacher,
    TeacherView,
    levels,
    residual_signs,
)
from w1a1_eagle.prototype import Levels, Prototype, Weights

CASES: tuple[tuple[Levels, str], ...] = (
    ("sign", "plain"),
    ("sign", "block"),
    ("sign", "residual"),
    ("ternary", "plain"),
    ("int2", "plain"),
    ("int4", "plain"),
    ("full", "plain"),
)


def tensors(generator: torch.Generator) -> dict[str, torch.Tensor]:
    result = {
        "fc.weight": torch.randn((4, 32), generator=generator) * 0.1,
        "enc.output_norm.weight": torch.ones(4),
        "output_norm.weight": torch.ones(4),
    }
    for index in range(6):
        for name, shape in {
            "attn_norm": (4,),
            "attn_q_norm": (2,),
            "attn_k_norm": (2,),
            "ffn_norm": (4,),
            "attn_q": (4, 4),
            "attn_k": (2, 4),
            "attn_v": (2, 4),
            "attn_output": (4, 4),
            "ffn_gate": (8, 4),
            "ffn_up": (8, 4),
            "ffn_down": (4, 8),
        }.items():
            result[f"blk.{index}.{name}.weight"] = (
                torch.ones(shape)
                if len(shape) == 1
                else (torch.randn(shape, generator=generator) * 0.1).bfloat16().float()
            )
    return result


class PrototypeTest(unittest.TestCase):
    def test_effective_weight_gradient_matches_straight_through_reference(self) -> None:
        generator = torch.Generator().manual_seed(5)
        rows, columns = 6, 256
        for kind, mode in CASES:
            latent = (torch.randn((rows, columns), generator=generator) * 0.4).clamp(-1.2, 1.2)
            latent.requires_grad_()
            scale = (torch.rand(rows, generator=generator) + 0.5).requires_grad_()
            residual = (
                torch.rand(rows, generator=generator) if mode == "residual" else torch.zeros(0)
            ).requires_grad_()
            block = (
                torch.rand((rows, 2), generator=generator) + 0.5
                if mode == "block"
                else torch.zeros((0, 0))
            ).requires_grad_()
            upstream = torch.randn((rows, columns), generator=generator)
            output = EFFECTIVE[kind](latent, scale, residual, block)
            (output * upstream).sum().backward()
            measured = [
                value.grad.clone() if value.grad is not None else torch.zeros_like(value)
                for value in (latent, scale, residual, block)
            ]
            for value in (latent, scale, residual, block):
                value.grad = None
            inside = (latent.abs() <= 1).float()
            through = latent * inside + (levels(latent, kind) - latent * inside).detach()
            if mode == "block":
                through = (through.view(rows, 2, -1) * block[:, :, None]).view(rows, columns)
            reference = through * scale[:, None]
            if mode == "residual":
                offset = latent - RESIDUAL_THRESHOLD * levels(latent, "sign").detach()
                near = (offset.abs() <= 1).float()
                second = offset * near + (residual_signs(latent) - offset * near).detach()
                reference = reference + second * residual[:, None]
            self.assertTrue(torch.allclose(output, reference.detach(), atol=1e-6), (kind, mode))
            (reference * upstream).sum().backward()
            for left, value in zip(measured, (latent, scale, residual, block), strict=True):
                right = value.grad if value.grad is not None else torch.zeros_like(value)
                self.assertTrue(torch.allclose(left, right, atol=1e-5), (kind, mode))

    def test_prototype_initialization_fits_weights_and_default_stays_one_bit(self) -> None:
        generator = torch.Generator().manual_seed(3)
        weight = torch.randn((64, 256), generator=generator)
        errors: dict[str, float] = {}
        for name, weights in (
            ("sign", Weights()),
            ("ternary", Weights("ternary")),
            ("int2", Weights("int2")),
            ("int4", Weights("int4")),
            ("full", Weights("full")),
            ("block", Weights("sign", 128)),
            ("residual", Weights("sign", None, True)),
        ):
            projection = Projection(
                weight, 4, nn.Parameter(torch.zeros(256)), nn.Parameter(torch.tensor(0.0)), weights
            )
            mean = weight.abs().mean(-1, keepdim=True)
            with torch.no_grad():
                projection.latent_sign.copy_((weight / (3 * mean)).clamp(-1, 1))
                effective = (
                    EFFECTIVE["sign"](
                        projection.latent_sign,
                        projection.scale,
                        torch.zeros(0),
                        torch.zeros((0, 0)),
                    )
                    if weights.default
                    else projection.effective_weight()
                )
            errors[name] = float((effective - weight).square().sum() / weight.square().sum())
        self.assertLess(errors["full"], errors["int4"])
        self.assertLess(errors["int4"], errors["int2"])
        self.assertLess(errors["int2"], errors["sign"])
        self.assertLess(errors["ternary"], errors["sign"])
        self.assertLess(errors["residual"], errors["sign"])
        self.assertLess(errors["block"], errors["sign"])
        self.assertTrue(Prototype().servable)
        self.assertFalse(Prototype(down_int4_layers=(2, 5)).servable)
        self.assertTrue(Prototype(teacher="logits", teacher_weight=1.0).servable)
        with self.assertRaises(ValueError):
            Prototype(teacher="hidden")
        with self.assertRaises(ValueError):
            Prototype("int2", block=128)

    def test_teacher_runs_full_precision_feed_forward_with_shared_tables(self) -> None:
        generator = torch.Generator().manual_seed(11)
        cfg = Geometry(hidden=4, intermediate=8, heads=2, kv_heads=1, head_dim=2)
        weights = tensors(generator)
        vocabulary = torch.randn((20, 4), generator=generator)
        student = HE4(weights, vocabulary, 19, cfg, prototype=Prototype(down_int4_layers=(2, 5)))
        teacher = Teacher(weights, cfg)
        features = torch.randn((20, 32), generator=generator)
        tokens = tuple(range(20))
        hidden = student.group_hidden(features, tokens, (3, 9), 4, teacher)
        self.assertEqual(tuple(hidden.shape), (2, 4, 4))
        self.assertTrue(bool(torch.isfinite(hidden).all()))
        self.assertFalse(torch.equal(hidden, student.group_hidden(features, tokens, (3, 9), 4)))
        logits = TeacherView(student, teacher).forward(features, tokens, 9, 4)
        expected = torch.nn.functional.linear(hidden[1], vocabulary)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-5))
        self.assertEqual(teacher.layers[0].gate.dtype, torch.bfloat16)

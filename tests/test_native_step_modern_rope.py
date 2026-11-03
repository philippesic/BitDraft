"""CPU checks for the modern EAGLE partial-RoPE training graph prerequisite."""

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.native_step import NativeStepAdapter, _rope_settings  # noqa: E402
from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402

# Relevant fields copied verbatim from the inspected local source config:
# runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json.
# Keep the parser check independent of untracked local weights/run artifacts.
MODERN_CONFIG = """{
    "head_dim": 160,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "intermediate_size": 7680,
    "max_position_embeddings": 262144,
    "num_attention_heads": 16,
    "num_hidden_layers": 1,
    "num_key_value_heads": 4,
    "partial_rotary_factor": 0.25,
    "pretraining_tp": 1,
    "rope_parameters": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default"
    }
}"""


class _Norm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = 1e-6


def _small_drafter(*, legacy=False):
    """Keep actual Q/KV/head geometry; shrink nonattention widths and vocab."""
    config = json.loads(MODERN_CONFIG)
    config.update(hidden_size=4, intermediate_size=5)
    if legacy:
        del config["rope_parameters"]
        del config["partial_rotary_factor"]
    drafter = nn.Module()
    drafter.config = SimpleNamespace(**config)
    generator = torch.Generator().manual_seed(815)

    def binary(out_features, in_features):
        return GroupedBinaryLinear(
            torch.randn(out_features, in_features, generator=generator) * 0.3,
            torch.full((out_features, math.ceil(in_features / 128)), 0.06),
            group_size=128,
        )

    drafter.embed_tokens = nn.Embedding(11, 4, dtype=torch.float16)
    with torch.no_grad():
        drafter.embed_tokens.weight.copy_(torch.randn(11, 4, generator=generator).half())
    drafter.fc = binary(4, 12)
    drafter.midlayer = nn.Module()
    layer = drafter.midlayer
    layer.input_layernorm = _Norm(4)
    layer.hidden_norm = _Norm(4)
    layer.post_attention_layernorm = _Norm(4)
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = binary(16 * 160, 8)
    layer.self_attn.k_proj = binary(4 * 160, 8)
    layer.self_attn.v_proj = binary(4 * 160, 8)
    layer.self_attn.o_proj = binary(4, 16 * 160)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = binary(5, 4)
    layer.mlp.up_proj = binary(5, 4)
    layer.mlp.down_proj = binary(4, 5)
    drafter.norm = _Norm(4)
    drafter.lm_head = binary(7, 4)
    return drafter


def _independent_rotation(x, positions, theta, dimensions):
    """Scalar F64 frequency/pair math, separate from the adapter's F32 loop."""
    expected = x.detach().double().clone()
    half = dimensions // 2
    for row, position in enumerate(positions):
        for channel in range(half):
            angle = position / theta ** (2 * channel / dimensions)
            cosine, sine = math.cos(angle), math.sin(angle)
            first = x[row, :, channel].detach().double()
            second = x[row, :, channel + half].detach().double()
            expected[row, :, channel] = first * cosine - second * sine
            expected[row, :, channel + half] = second * cosine + first * sine
    return expected.float()


class ModernRopeTests(unittest.TestCase):
    def test_actual_modern_config_fields_and_legacy_fallbacks(self):
        config = SimpleNamespace(**json.loads(MODERN_CONFIG))
        self.assertEqual(_rope_settings(config, config.head_dim), (10000000.0, 40))
        self.assertEqual((config.hidden_size, config.intermediate_size), (2560, 7680))
        self.assertEqual((config.num_attention_heads, config.num_key_value_heads), (16, 4))
        # HF can populate these legacy attributes with defaults: nested wins.
        config.rope_theta = 10000.0
        config.partial_rotary_factor = 1.0
        self.assertEqual(_rope_settings(config, 160), (10000000.0, 40))
        self.assertEqual(_rope_settings(SimpleNamespace(), 160), (10000.0, 160))
        self.assertEqual(
            _rope_settings(SimpleNamespace(rope_theta=20000, partial_rotary_factor=0.25), 160),
            (20000.0, 40),
        )

    def test_rejects_scaled_and_invalid_rotary_settings(self):
        invalid = [
            {"rope_scaling": {"type": "linear", "factor": 2}},
            {"rope_parameters": []},
            {"rope_parameters": {"rope_type": "linear"}},
            {"rope_parameters": {"factor": 2}},
            *({"rope_parameters": {"rope_theta": value}} for value in [0, -1, math.inf, True]),
            *(
                {"partial_rotary_factor": value}
                for value in [0, -0.25, 1.25, math.nan, True, 0.00625, 0.01875, 0.015625]
            ),
        ]
        for attributes in invalid:
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                _rope_settings(SimpleNamespace(**attributes), 160)

    def test_vectorized_rotation_matches_independent_math_and_preserves_tail_gradient(self):
        adapter = NativeStepAdapter(_small_drafter())
        self.assertEqual((adapter.rope_theta, adapter.rotary_dim), (10000000.0, 40))
        generator = torch.Generator().manual_seed(914)
        x = torch.randn(3, 16, 160, generator=generator, requires_grad=True)
        positions = [0, 1, 63]
        rotated = adapter._rope(x, torch.tensor(positions))
        expected = _independent_rotation(x, positions, 10000000.0, 40)
        torch.testing.assert_close(rotated, expected, rtol=2e-5, atol=2e-5)
        self.assertTrue(torch.equal(rotated[..., 40:], x[..., 40:]))
        rotated.sum().backward()
        self.assertTrue(torch.equal(x.grad[..., 40:], torch.ones_like(x.grad[..., 40:])))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[..., :40].abs().sum()), 0)

    def test_serial_qk_rotation_and_vectorized_context_cache_agree(self):
        adapter = NativeStepAdapter(_small_drafter())
        generator = torch.Generator().manual_seed(761)
        raw = torch.randn(4, 12, generator=generator)
        tokens = [1, 3, 5, 7]
        cache = adapter.new_cache()
        raw_keys, rotated_keys = [], []
        for position, token in enumerate(tokens):
            taps = {}
            step = adapter.decode_step(
                token, adapter.encode_feature(raw[position]), position, cache,
                trace_callback=lambda name, value: taps.__setitem__(name, value),
            )
            for name, heads in (("Qcur", 16), ("Kcur", 4)):
                before = taps[f"{name}-0"].reshape(1, heads, 160)
                after = taps[f"{name}_rope-0"]
                expected = _independent_rotation(before, [position], 10000000.0, 40)[0]
                torch.testing.assert_close(after, expected, rtol=2e-5, atol=2e-6)
                self.assertTrue(torch.equal(after[:, 40:], before[0, :, 40:]))
            raw_keys.append(taps["Kcur-0"].reshape(4, 160))
            rotated_keys.append(taps["Kcur_rope-0"])
            cache = step.cache
        vectorized = adapter._rope(torch.stack(raw_keys), torch.arange(4))
        self.assertTrue(torch.equal(vectorized, torch.stack(rotated_keys)))
        torch.testing.assert_close(
            cache.key, vectorized.half().float().transpose(0, 1), rtol=0, atol=0
        )
        context = adapter.build_context_cache(tokens, raw, chunk_size=2)
        torch.testing.assert_close(context.key, cache.key, rtol=0, atol=0)
        torch.testing.assert_close(context.value, cache.value, rtol=0, atol=0)
        step.logits.square().sum().backward()
        for name in ("q_proj", "k_proj", "v_proj"):
            projection = getattr(adapter.drafter.midlayer.self_attn, name)
            self.assertGreater(float(projection.scale_offset.grad.abs().sum()), 0)

    def test_legacy_config_still_rotates_full_head_with_default_theta(self):
        adapter = NativeStepAdapter(_small_drafter(legacy=True))
        self.assertEqual((adapter.rope_theta, adapter.rotary_dim), (10000.0, 160))
        x = torch.linspace(-1, 1, 160).reshape(1, 1, 160)
        expected = _independent_rotation(x, [7], 10000.0, 160)
        torch.testing.assert_close(
            adapter._rope(x, torch.tensor([7])), expected, rtol=2e-5, atol=2e-6
        )


if __name__ == "__main__":
    unittest.main()

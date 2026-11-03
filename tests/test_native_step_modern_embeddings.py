"""Small CPU contract checks; synthetic rows do not establish target ancestry."""

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_native_step import _drafter  # noqa: E402

from w1a1_eagle.native_step import (  # noqa: E402
    ExternalNativeF32Embedding,
    NativeStepAdapter,
)


class _Rows:
    """Tiny stand-in for an admitted native lookup, never real target weights."""

    vocab_size = 11
    hidden_size = 4

    def __init__(self):
        # All four values change when rounded through F16.
        self.rows = torch.tensor(
            [1.0001, -0.33333334, 0.000123456, 3.1415927], dtype=torch.float32
        ).repeat(self.vocab_size, 1)

    def __call__(self, token):
        return self.rows[token]


def _declared(rows=None, **overrides):
    fields = dict(
        lookup=_Rows() if rows is None else rows,
        vocab_size=11,
        hidden_size=4,
        target_identity="synthetic-qwen3.5-Q4_K_M-test-only",
        target_sha256="a" * 64,
        capture_sha256="b" * 64,
    )
    fields.update(overrides)
    return ExternalNativeF32Embedding(**fields)


def _native_adapter(lookup):
    return NativeStepAdapter(
        _drafter(), embedding_lookup=lookup, embedding_precision="external_native_f32"
    )


class ModernEmbeddingTests(unittest.TestCase):
    def test_historical_f16_default_keeps_exact_widening_and_freezes_embedding(self):
        drafter = _drafter()
        expected = drafter.embed_tokens.weight[3].detach().float()
        adapter = NativeStepAdapter(drafter)
        self.assertEqual(adapter.embedding_precision, "f16")
        self.assertEqual(adapter.embedding_dtype, torch.float16)
        self.assertTrue(torch.equal(adapter._embedding(3), expected))
        self.assertFalse(adapter._embedding(3).requires_grad)
        self.assertFalse(drafter.embed_tokens.weight.requires_grad)

        external = _Rows()
        external.rows = external.rows.half()
        adapter = NativeStepAdapter(_drafter(), embedding_lookup=external)
        self.assertTrue(torch.equal(adapter._embedding(3), external.rows[3].float()))

    def test_f32_is_rejected_by_default_for_dense_and_external_sources(self):
        drafter = _drafter()
        drafter.embed_tokens = nn.Embedding(11, 4, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "must be F16"):
            NativeStepAdapter(drafter)
        adapter = NativeStepAdapter(_drafter(), embedding_lookup=_Rows())
        with self.assertRaisesRegex(ValueError, "frozen finite F16"):
            adapter._embedding(3)
        with self.assertRaisesRegex(ValueError, "frozen finite F16"):
            adapter.decode_step(3, torch.ones(4), 0, adapter.new_cache())
        with self.assertRaisesRegex(ValueError, "explicit external_native_f32"):
            NativeStepAdapter(_drafter(), embedding_lookup=_declared())

    def test_native_f32_requires_both_explicit_mode_and_declared_external_lookup(self):
        for lookup in (None, _Rows()):
            with self.subTest(lookup=lookup), self.assertRaisesRegex(
                ValueError, "declared ExternalNativeF32Embedding"
            ):
                _native_adapter(lookup)
        with self.assertRaisesRegex(ValueError, "unsupported embedding precision"):
            NativeStepAdapter(_drafter(), embedding_precision="f32")

    def test_native_f32_rows_remain_exact_in_decode_and_context_with_gradients_off(self):
        rows = _Rows()
        declaration = _declared(rows)
        adapter = _native_adapter(declaration)
        expected = rows.rows[3].clone()
        self.assertEqual(adapter.embedding_dtype, torch.float32)
        self.assertTrue(torch.all(expected != expected.half().float()))
        self.assertTrue(torch.equal(adapter._embedding(3), expected))
        self.assertFalse(adapter._embedding(3).requires_grad)
        self.assertIs(adapter.embedding_lookup, declaration)
        self.assertEqual(declaration.target_sha256, "a" * 64)
        with self.assertRaises(FrozenInstanceError):
            declaration.target_identity = "changed"

        raw = torch.ones(2, 12)
        cache = adapter.new_cache()
        for position, token in enumerate((3, 5)):
            taps = {}
            step = adapter.decode_step(
                token, adapter.encode_feature(raw[position]), position, cache,
                trace_callback=lambda name, value: taps.__setitem__(name, value),
            )
            self.assertTrue(torch.equal(taps["inp_embd"], rows.rows[token]))
            self.assertFalse(taps["inp_embd"].requires_grad)
            cache = step.cache
        context = adapter.build_context_cache([3, 5], raw, chunk_size=1)
        torch.testing.assert_close(context.key, cache.key, rtol=0, atol=0)
        torch.testing.assert_close(context.value, cache.value, rtol=0, atol=0)
        step.logits.square().sum().backward()
        self.assertIsNone(rows.rows.grad)
        self.assertFalse(adapter.drafter.embed_tokens.weight.requires_grad)
        self.assertIsNotNone(adapter.drafter.lm_head.scale_offset.grad)
        self.assertTrue(torch.isfinite(adapter.drafter.lm_head.scale_offset.grad).all())

    def test_missing_or_malformed_provenance_is_rejected(self):
        invalid = [
            {"target_identity": ""},
            {"target_identity": "  "},
            {"target_identity": None},
            {"target_sha256": ""},
            {"target_sha256": "z" * 64},
            {"capture_sha256": "b" * 63},
            {"capture_sha256": "B" * 64},
            {"capture_sha256": None},
        ]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "provenance"):
                _declared(**fields)

    def test_non_native_unfrozen_and_wrong_dtype_declarations_are_rejected(self):
        for fields in (
            {"source": "hf_dense"},
            {"frozen": False},
            {"frozen": 1},
            {"dtype": torch.float16},
            {"dtype": torch.bfloat16},
        ):
            with self.subTest(fields=fields), self.assertRaisesRegex(
                ValueError, "frozen native_get_rows F32"
            ):
                _declared(**fields)
        for frozen in (False, True):
            embedding = nn.Embedding(11, 4, dtype=torch.float32)
            embedding.weight.requires_grad_(not frozen)
            with self.subTest(frozen=frozen), self.assertRaisesRegex(
                ValueError, "external native row callable"
            ):
                _declared(embedding)

        class TrainableLookup(nn.Module):
            def __init__(self):
                super().__init__()
                self.row = nn.Parameter(torch.ones(4))

            def forward(self, token):
                return self.row.detach()

        with self.assertRaisesRegex(ValueError, "trainable parameters"):
            _declared(TrainableLookup())

    def test_wrong_runtime_dtype_and_trainable_rows_are_rejected_without_casting(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            rows = _Rows()
            rows.rows = rows.rows.to(dtype)
            adapter = _native_adapter(_declared(rows))
            with self.subTest(dtype=dtype), self.assertRaisesRegex(
                ValueError, "frozen finite native F32"
            ):
                adapter._embedding(3)
        rows = _Rows()
        rows.rows.requires_grad_()
        adapter = _native_adapter(_declared(rows))
        with self.assertRaisesRegex(ValueError, "frozen finite native F32"):
            adapter._embedding(3)
        with self.assertRaisesRegex(ValueError, "frozen finite native F32"):
            adapter.decode_step(3, torch.ones(4), 0, adapter.new_cache())
        self.assertTrue(rows.rows.requires_grad)

    def test_dimensions_finiteness_and_token_bounds_remain_enforced(self):
        for fields in ({"vocab_size": 0}, {"hidden_size": True}, {"hidden_size": 0}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "dimensions"):
                _declared(**fields)
        with self.assertRaisesRegex(ValueError, "incompatible dimensions"):
            _native_adapter(_declared(hidden_size=5))
        adapter = _native_adapter(_declared())
        for token in (-1, 11, True):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "token"):
                adapter._embedding(token)
        for row in (torch.ones(3), torch.tensor([1.0, 2.0, 3.0, float("nan")])):
            adapter = _native_adapter(_declared(lambda token: row))
            with self.assertRaisesRegex(ValueError, "frozen finite native F32"):
                adapter._embedding(0)


if __name__ == "__main__":
    unittest.main()

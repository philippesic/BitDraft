"""Modern packed serialization with independent prefix-RoPE negative controls."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_eagle_w1a1_gguf import audit, qk_layout_from_gguf  # noqa: E402
from export_recurrent_binary import (  # noqa: E402
    AFFINE_ARITHMETIC,
    SOURCE_NAMES,
    GGUFReader,
    GGUFWriter,
    Type,
    export_model,
    gguf_qk_row_order,
    load_affine_weights,
    sha256,
)


def prefix_order(rows, heads, rotary):
    """Reference expressed as explicit paired row assignments, including tail."""
    result = np.arange(rows)
    width = rows // heads
    for head in range(heads):
        start = head * width
        for channel in range(rotary // 2):
            result[start + 2 * channel] = start + channel
            result[start + 2 * channel + 1] = start + channel + rotary // 2
    return result


class ModernBinaryExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "base.gguf"
        self.source = self.root / "model.safetensors"
        self.checkpoint = self.root / "checkpoint.npz"
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "binary.gguf"
        self.weights = {}
        self.shapes = {}
        self.arrays = {}
        for index, (base, name) in enumerate(SOURCE_NAMES.items()):
            rows = 2560 if base.endswith("attn_q") else 640 if base.endswith("attn_k") else 9
            row, col = np.arange(rows)[:, None], np.arange(35)[None, :]
            weight = (1 - 2 * ((row >> (col % 8)) & 1)) * (1 + row % 97) / 128
            value = torch.tensor(weight, dtype=torch.bfloat16)
            self.weights[name] = value
            latent = value.float().numpy()
            self.shapes[base] = list(latent.shape)
            self.arrays[name + ".latent"] = latent
            self.arrays[name + ".scale"] = np.mean(np.abs(latent), axis=1, dtype=np.float32)
        save_file(self.weights, self.source)
        np.savez(self.checkpoint, **self.arrays)
        self.write_base()

    def write_base(self, **layout_overrides):
        fields = {
            "attention.head_count": 16,
            "attention.head_count_kv": 4,
            "attention.key_length": 160,
            "rope.dimension_count": 40,
        }
        fields.update(layout_overrides)
        writer = GGUFWriter(self.base, "eagle3")
        for name, value in fields.items():
            if value is not None:
                writer.add_uint32("eagle3." + name, value)
        writer.add_array("eagle3.target_layers", [2, 16, 29])
        writer.add_float32("eagle3.rope.freq_base", 1e7)
        for base, name in SOURCE_NAMES.items():
            raw = self.weights[name].view(torch.int16).numpy().view(np.uint8)
            writer.add_tensor(base + ".weight", raw, raw_dtype=Type.BF16)
        writer.add_tensor("output_norm.weight", np.array([1, 2, 3], dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        manifest = {
            "schema_version": 2,
            "base_gguf_sha256": sha256(self.base),
            "checkpoint_sha256": sha256(self.checkpoint),
            "scale_layout": "row",
            "activation_bits": 8,
            "activation_rule": "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive",
            "weight_rule": "hard_sign_zero_positive_clipped_identity_ste",
            "qk_row_order": "original_checkpoint",
            "export_status": "row_w1ax_requires_native_validation",
            "objective": "hard_ce",
            "projections": {
                base: {"checkpoint_name": name, "shape": self.shapes[base]}
                for base, name in SOURCE_NAMES.items()
            },
        }
        self.manifest.write_text(json.dumps(manifest))

    def export(self):
        return export_model(self.base, self.checkpoint, self.manifest, self.output)

    def test_modern_bf16_all_nine_words_scales_and_unrotated_tails(self):
        report = self.export()
        tensors = {t.name: t for t in GGUFReader(self.output).tensors}
        self.assertEqual(
            report["qk_layout"]["blk.0.attn_q"], {"heads": 16, "head_dim": 160, "rotary_dim": 40}
        )
        for base, name in SOURCE_NAMES.items():
            latent, scales = self.arrays[name + ".latent"], self.arrays[name + ".scale"]
            order = np.arange(len(latent))
            if base in ("blk.0.attn_q", "blk.0.attn_k"):
                heads = 16 if base.endswith("_q") else 4
                order = prefix_order(len(latent), heads, 40)
                np.testing.assert_array_equal(
                    order.reshape(heads, 160)[:, 40:],
                    np.arange(len(latent)).reshape(heads, 160)[:, 40:],
                )
            bits = np.unpackbits(
                tensors[base + ".w1a1_packed"].data.view(np.uint8), axis=1, bitorder="little"
            )
            np.testing.assert_array_equal(bits[:, :35], latent[order] >= 0)
            np.testing.assert_array_equal(bits[:, 35:], 0)
            np.testing.assert_array_equal(tensors[base + ".w1a1_scale"].data, scales[order])
            self.assertNotIn(base + ".weight", tensors)
        result = audit(self.source, self.output, rows_per_chunk=37)
        self.assertEqual(result["packed_tensors"], 9)
        self.assertTrue(all(row["packed_word_mismatches"] == 0 for row in result["records"]))
        np.testing.assert_array_equal(tensors["output_norm.weight"].data, [1, 2, 3])
        reader = GGUFReader(self.output)
        self.assertEqual(reader.fields["eagle3.target_layers"].contents(), [2, 16, 29])
        self.assertEqual(reader.fields["eagle3.rope.dimension_count"].contents(), 40)

    def test_prefix_projection_rope_dot_matches_and_full_head_control_fails(self):
        self.export()
        tensors = {t.name: t for t in GGUFReader(self.output).tensors}
        projections = {}
        inputs = np.sin(np.arange(35) * 0.17)
        for base in ("blk.0.attn_q", "blk.0.attn_k"):
            name = SOURCE_NAMES[base]
            original = np.where(self.arrays[name + ".latent"] >= 0, 1.0, -1.0)
            original *= self.arrays[name + ".scale"][:, None]
            bits = np.unpackbits(
                tensors[base + ".w1a1_packed"].data.view(np.uint8), axis=1, bitorder="little"
            )[:, :35]
            serialized = (2.0 * bits.astype(float) - 1) * tensors[base + ".w1a1_scale"].data[
                :, None
            ]
            projections[base] = (original @ inputs, serialized @ inputs)

        def rotate_split(x, position):
            y = x.copy()
            angle = position / (1e7 ** (np.arange(0, 40, 2) / 40))
            y[:20] = x[:20] * np.cos(angle) - x[20:40] * np.sin(angle)
            y[20:40] = x[20:40] * np.cos(angle) + x[:20] * np.sin(angle)
            return y

        def rotate_pairs(x, position):
            y = x.copy()
            angle = position / (1e7 ** (np.arange(0, 40, 2) / 40))
            y[:40:2] = x[:40:2] * np.cos(angle) - x[1:40:2] * np.sin(angle)
            y[1:40:2] = x[1:40:2] * np.cos(angle) + x[:40:2] * np.sin(angle)
            return y

        q, k = projections["blk.0.attn_q"], projections["blk.0.attn_k"]
        expected = rotate_split(q[0][:160], 3) @ rotate_split(k[0][:160], 29)
        actual = rotate_pairs(q[1][:160], 3) @ rotate_pairs(k[1][:160], 29)
        self.assertAlmostEqual(actual, expected, places=9)
        wrong = prefix_order(160, 1, 160)
        negative = rotate_pairs(q[0][:160][wrong], 3) @ rotate_pairs(k[0][:160][wrong], 29)
        self.assertGreater(abs(negative - expected), 1e-3)

    def test_initializer_audit_rejects_known_wrong_full_head_export(self):
        self.export()
        reader = GGUFReader(self.output, mode="r+")
        for base, heads in (("blk.0.attn_q", 16), ("blk.0.attn_k", 4)):
            name = SOURCE_NAMES[base]
            order = prefix_order(self.shapes[base][0], heads, 160)
            wrong = self.arrays[name + ".latent"][order] >= 0
            wrong = np.pad(wrong, ((0, 0), (0, 29)))
            packed = np.packbits(wrong, axis=1, bitorder="little").view("<i4")
            next(t for t in reader.tensors if t.name == base + ".w1a1_packed").data[:] = packed
            next(t for t in reader.tensors if t.name == base + ".w1a1_scale").data[:] = self.arrays[
                name + ".scale"
            ][order]
        del reader
        with self.assertRaisesRegex(ValueError, "packed word mismatches"):
            audit(self.source, self.output)

    def test_incomplete_wrong_width_and_invalid_rotary_metadata_fail_before_export(self):
        for changes in (
            {"attention.head_count_kv": None},
            {"attention.head_count": 15},
            {"attention.key_length": 128},
            {"rope.dimension_count": 41},
            {"rope.dimension_count": 162},
            {"rope.dimension_count": 0},
        ):
            with self.subTest(changes=changes):
                self.write_base(**changes)
                with self.assertRaisesRegex(ValueError, "Q/K|rotary"):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_affine_qk_midpoints_follow_the_same_partial_permutation(self):
        descriptor = {
            "version": 1,
            "coverage": "all",
            "arithmetic": AFFINE_ARITHMETIC,
            "tensors": {base: base + ".w1ax_midpoint" for base in SOURCE_NAMES},
        }
        midpoint = {
            name: np.arange(self.shapes[base][0], dtype=np.float32)
            for base, name in descriptor["tensors"].items()
        }
        path = self.root / "affine.npz"
        np.savez(path, **midpoint)
        expected = {base: (name, self.shapes[base]) for base, name in SOURCE_NAMES.items()}
        layout = qk_layout_from_gguf(GGUFReader(self.base), self.shapes)
        result = load_affine_weights(path, descriptor, expected, qk_layout=layout)
        for base, heads in (("blk.0.attn_q", 16), ("blk.0.attn_k", 4)):
            name = descriptor["tensors"][base]
            np.testing.assert_array_equal(
                result[name], midpoint[name][prefix_order(self.shapes[base][0], heads, 40)]
            )

    def test_legacy_full_permutation_is_byte_identical_and_bad_layout_rejected(self):
        weights = np.arange(32 * 128 * 7, dtype=np.float32).reshape(32 * 128, 7)
        old = weights.reshape(32, 2, 64, 7).swapaxes(1, 2).reshape(weights.shape)
        np.testing.assert_array_equal(gguf_qk_row_order(weights, 32), old)
        for heads, rotary in ((0, None), (True, None), (31, None), (32, 129), (32, 39)):
            with self.subTest(heads=heads, rotary=rotary), self.assertRaises(ValueError):
                gguf_qk_row_order(weights, heads, rotary)


if __name__ == "__main__":
    unittest.main()

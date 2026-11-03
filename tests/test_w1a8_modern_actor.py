"""Small CPU contract/compact-state tests; real initializer qualification is separate."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "modern_actor", PROJECT / "scripts/w1a8_generalization/modern_actor.py"
)
actor_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = actor_module
SPEC.loader.exec_module(actor_module)


class Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


def small_actor():
    drafter, linears = nn.Module(), {}
    for path in actor_module.CANDIDATE_D_BASE_TO_PATH.values():
        parts = path.split(".")
        parent = drafter
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        module = actor_module.RowBinaryLinear(
            torch.full((2, 32), 0.5), torch.tensor([0.25, 0.5]), actor_module.W1AxContract(8)
        )
        setattr(parent, parts[-1], module)
        linears[path] = module
    return actor_module.ModernActor(
        drafter=drafter,
        adapter=None,
        linears=linears,
        initializer={"path": "fixture.gguf", "sha256": "a" * 64, "bytes": 123},
        config={"path": "fixture.json", "sha256": "b" * 64, "bytes": 456},
        contract={"sha256": "c" * 64, "semantics": {"fixture": True}},
        qk_layout={},
        initial_sign_hashes={
            p: actor_module._tensor_hash(m.latent_sign) for p, m in linears.items()
        },
        initial_scale_hashes={
            p: actor_module._tensor_hash(m.initial_scale) for p, m in linears.items()
        },
    )


class CompactActorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.actor = small_actor()

    def test_scale_only_is_opt_in_and_nine_optimizer_tensors(self):
        self.assertTrue(all(m.latent_sign.requires_grad for m in self.actor.linears.values()))
        with self.assertRaisesRegex(ValueError, "explicit"):
            self.actor.scale_parameters()
        parameters = actor_module.enable_scale_only(self.actor)
        self.assertEqual(len(parameters), 9)
        self.assertEqual(
            {id(p) for p in parameters},
            {id(p) for p in self.actor.drafter.parameters() if p.requires_grad},
        )
        self.assertTrue(all(not m.latent_sign.requires_grad for m in self.actor.linears.values()))

    def test_exact_offset_roundtrip_and_state_is_detached_copy(self):
        actor_module.enable_scale_only(self.actor)
        with torch.no_grad():
            for index, module in enumerate(self.actor.linears.values()):
                module.scale_offset.copy_(torch.tensor([-0.9, 0.01 * index]))
        state = actor_module.collect_scale_state(self.actor)
        expected = {k: v.clone() for k, v in state["scale_offsets"].items()}
        with torch.no_grad():
            for module in self.actor.linears.values():
                module.scale_offset.zero_()
        actor_module.load_scale_state(self.actor, state)
        for base, path in actor_module.CANDIDATE_D_BASE_TO_PATH.items():
            self.assertTrue(torch.equal(self.actor.linears[path].scale_offset, expected[base]))
        state["ancestry"]["contract"]["semantics"]["fixture"] = False
        self.assertTrue(self.actor.contract["semantics"]["fixture"])

    def test_reject_wrong_ancestry_schema_and_unknown_scales_without_partial_load(self):
        actor_module.enable_scale_only(self.actor)
        good = actor_module.collect_scale_state(self.actor)
        for kind in (
            "initializer",
            "config",
            "contract",
            "unknown",
            "missing",
            "shape",
            "nan",
            "dtype",
            "schema",
        ):
            with self.subTest(kind=kind):
                state = copy.deepcopy(good)
                if kind in ("initializer", "config", "contract"):
                    state["ancestry"][kind]["sha256"] = "0" * 64
                elif kind == "unknown":
                    state["scale_offsets"]["unknown"] = torch.zeros(2)
                elif kind == "missing":
                    state["scale_offsets"].pop("output")
                elif kind == "shape":
                    state["scale_offsets"]["output"] = torch.zeros(3)
                elif kind == "nan":
                    state["scale_offsets"]["output"][0] = float("nan")
                elif kind == "dtype":
                    state["scale_offsets"]["output"] = torch.zeros(2, dtype=torch.float64)
                else:
                    state["schema"] = "other"
                # A valid first payload must not be copied before a later payload fails.
                state["scale_offsets"]["fc"] = torch.ones(2)
                with self.assertRaises(ValueError):
                    actor_module.load_scale_state(self.actor, state)
                self.assertTrue(torch.equal(self.actor.linears["fc"].scale_offset, torch.zeros(2)))

    def test_reject_changed_frozen_signs_initial_scales_and_optimizer_scope(self):
        actor_module.enable_scale_only(self.actor)
        first = self.actor.linears["fc"]
        with torch.no_grad():
            first.latent_sign[0, 0] = -0.5
        with self.assertRaisesRegex(ValueError, "frozen signs"):
            actor_module.collect_scale_state(self.actor)
        with torch.no_grad():
            first.latent_sign[0, 0] = 0.5
            first.initial_scale[0] += 0.01
        with self.assertRaisesRegex(ValueError, "initializer scales"):
            actor_module.collect_scale_state(self.actor)
        first.latent_sign.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "optimizer scope"):
            self.actor.scale_parameters()

    def test_partial_qk_order_inverse_and_tail(self):
        order = actor_module.qk_row_indices(2, 8, 4)
        np.testing.assert_array_equal(order, [0, 2, 1, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13, 14, 15])
        source = np.arange(16 * 3).reshape(16, 3)
        np.testing.assert_array_equal(source[order][np.argsort(order)], source)
        for geometry in ((0, 8, 4), (2, 8, 3), (2, 8, 10)):
            with self.assertRaises(ValueError):
                actor_module.qk_row_indices(*geometry)

    def test_optimizer_rng_cursor_outer_recipe_resume_is_exact(self):
        actor_module.enable_scale_only(self.actor)
        optimizer = torch.optim.Adam(self.actor.scale_parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(17)
        cursor = 0

        def update():
            nonlocal cursor
            optimizer.zero_grad()
            coefficients = torch.rand(9, generator=generator)
            loss = sum(
                p.square().sum() + c * p.sum()
                for c, p in zip(coefficients, self.actor.scale_parameters(), strict=True)
            )
            loss.backward()
            optimizer.step()
            cursor += 1
            return loss.detach().clone()

        update()
        packet = {
            "scales": actor_module.collect_scale_state(self.actor),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": generator.get_state().clone(),
            "cursor": cursor,
        }
        expected_loss = update()
        expected = actor_module.collect_scale_state(self.actor)
        actor_module.load_scale_state(self.actor, packet["scales"])
        optimizer.load_state_dict(packet["optimizer"])
        generator.set_state(packet["rng"])
        cursor = packet["cursor"]
        actual_loss = update()
        self.assertTrue(torch.equal(actual_loss, expected_loss))
        self.assertEqual(cursor, 2)
        for key, tensor in expected["scale_offsets"].items():
            self.assertTrue(
                torch.equal(
                    tensor, actor_module.collect_scale_state(self.actor)["scale_offsets"][key]
                )
            )

    def test_wrong_initializer_or_config_rejected_before_gguf_open(self):
        # Ancestry gates precede GGUF construction and model allocation.
        lookup = actor_module.ExternalNativeF32Embedding(
            lookup=lambda _: torch.zeros(2560),
            vocab_size=248320,
            hidden_size=2560,
            target_identity="synthetic",
            target_sha256="d" * 64,
            capture_sha256="e" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "not-a-gguf"
            fake.write_text("wrong initializer")
            with self.assertRaisesRegex(ValueError, "initializer ancestry"):
                actor_module.build_modern_actor(fake, fake, embedding_lookup=lookup)

    def test_modern_metadata_missing_head_or_wrong_rotary_rejected(self):
        cfg = dict(
            actor_module.MODERN_DIMENSIONS,
            rope_parameters={
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            partial_rotary_factor=0.25,
            pretraining_tp=1,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
            target_hidden_size=2560,
            max_position_embeddings=262144,
            eagle_config={"eagle_aux_hidden_state_layer_ids": [1, 15, 28]},
        )
        # Missing metadata must fail before any assumed legacy layout can be selected.
        with self.assertRaisesRegex(ValueError, "metadata"):
            actor_module.validate_metadata(SimpleNamespace(fields={}), cfg)
        cfg["num_attention_heads"] = 32
        with self.assertRaisesRegex(ValueError, "geometry"):
            actor_module.validate_metadata(SimpleNamespace(fields={}), cfg)
        cfg["num_attention_heads"] = 16
        cfg["rope_parameters"]["partial_rotary_factor"] = 1
        with self.assertRaisesRegex(ValueError, "RoPE"):
            actor_module.validate_metadata(SimpleNamespace(fields={}), cfg)


if __name__ == "__main__":
    unittest.main()

"""Small CPU selected-sign state/GGUF fixtures; no real model or CUDA allocation."""

# Fixture imports follow the explicit source/GGUF path bootstrap.
# ruff: noqa: E402
from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts/w1a8_generalization"),
    "/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py",
]
import modern_actor as actor_module
import modern_selected_sign as selected
from gguf import GGMLQuantizationType as Type
from gguf import GGUFReader, GGUFWriter


def attach(root, path, value):
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(parent, part):
            setattr(parent, part, nn.Module())
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def small_actor(directory):
    """Four-row packed projections with 35 columns exercise non-word tail bits."""
    root = nn.Module()
    linears = {}
    pattern = torch.where((torch.arange(4)[:, None] + torch.arange(35)) % 2 == 0, -0.5, 0.5)
    for base, path in actor_module.CANDIDATE_D_BASE_TO_PATH.items():
        module = actor_module.RowBinaryLinear(
            pattern, torch.tensor([0.0, 0.25, 0.5, 0.75]), actor_module.W1AxContract(8)
        )
        attach(root, path, module)
        linears[path] = module
    norm_hashes = {}
    for native, path in selected.NATIVE_NORM_PATHS.items():
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(4), requires_grad=False)
        attach(root, path, module)
        norm_hashes[native] = actor_module._tensor_hash(module.weight)
    absolute = np.array([1, 3, 5, 7], dtype="<i8")
    root.register_buffer("d2t", torch.from_numpy(absolute - np.arange(4, dtype=np.int64)))
    membership = torch.zeros(9, dtype=torch.bool)
    membership[torch.from_numpy(absolute)] = True
    root.register_buffer("t2d", membership)
    layout = {
        base: {"heads": 1, "head_dim": 4, "rotary_dim": 4}
        for base in ("blk.0.attn_q", "blk.0.attn_k")
    }
    config_path = directory / "config.json"
    config_path.write_text('{"fixture": true}\n')
    source = directory / "initializer.gguf"
    writer = GGUFWriter(source, "eagle3")
    writer.add_string("fixture.keep_metadata", "unaltered")
    for base, path in actor_module.CANDIDATE_D_BASE_TO_PATH.items():
        module = linears[path]
        signs = torch.where(module.latent_sign.detach() < 0, -1.0, 1.0).numpy()
        scales = module.initial_scale.numpy().copy()
        if base in layout:
            order = actor_module.qk_row_indices(**layout[base])
            signs, scales = signs[order], scales[order]
        writer.add_tensor(base + ".w1a1_packed", selected.pack_signs(signs), raw_dtype=Type.I32)
        writer.add_tensor(base + ".w1a1_scale", scales)
    for native, path in selected.NATIVE_NORM_PATHS.items():
        writer.add_tensor(native, root.get_submodule(path).weight.detach().numpy())
    writer.add_tensor("d2t", absolute, raw_dtype=Type.I64)
    writer.add_tensor("fixture_canary", np.array([31, 17, 99], dtype=np.int32), raw_dtype=Type.I32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    semantics = {
        "shape": {base: [4, 35] for base in actor_module.CANDIDATE_D_BASE_TO_PATH},
        "qk_layout": copy.deepcopy(layout),
        "norms": norm_hashes,
        "absolute_d2t_sha256": hashlib.sha256(absolute.tobytes()).hexdigest(),
        "fixture": True,
    }
    return actor_module.ModernActor(
        drafter=root,
        adapter=None,
        linears=linears,
        initializer=actor_module.file_pin(source),
        config=actor_module.file_pin(config_path),
        contract={"sha256": "c" * 64, "semantics": semantics},
        qk_layout=layout,
        initial_sign_hashes={
            p: actor_module._tensor_hash(m.latent_sign) for p, m in linears.items()
        },
        initial_scale_hashes={
            p: actor_module._tensor_hash(m.initial_scale) for p, m in linears.items()
        },
    )


class SelectedSignTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.actor = small_actor(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def enable(self):
        selected.enable_selected_sign(self.actor)
        return self.actor.linears[selected.SELECTED_PATH]

    def test_opt_in_exact_scope_and_old_scale_apis_remain_strict(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            selected.collect_selected_state(self.actor)
        module = self.enable()
        groups = selected.selected_parameter_groups(self.actor, sign_lr=0.01, scale_lr=1e-5)
        self.assertEqual([len(g["params"]) for g in groups], [1, 9])
        self.assertIs(groups[0]["params"][0], module.latent_sign)
        self.assertEqual([g["lr"] for g in groups], [0.01, 1e-5])
        expected = {id(module.latent_sign)} | {
            id(m.scale_offset) for m in self.actor.linears.values()
        }
        self.assertEqual(
            {id(p) for p in self.actor.drafter.parameters() if p.requires_grad}, expected
        )
        for operation in (
            self.actor.scale_parameters,
            lambda: actor_module.collect_scale_state(self.actor),
        ):
            with self.assertRaisesRegex(ValueError, "scale-only"):
                operation()
        self.assertFalse(self.actor.scale_only)

    def test_mask_shape_and_nonfinite_reject_before_flag_mutation(self):
        flags = [p.requires_grad for p in self.actor.drafter.parameters()]
        with self.assertRaisesRegex(ValueError, "attention-output"):
            selected.enable_selected_sign(self.actor, ("fc",))
        self.assertEqual(flags, [p.requires_grad for p in self.actor.drafter.parameters()])
        self.actor.contract["semantics"]["shape"]["fc"] = [4, 34]
        with self.assertRaises(ValueError):
            selected.enable_selected_sign(self.actor)
        self.assertEqual(flags, [p.requires_grad for p in self.actor.drafter.parameters()])
        self.actor.contract["semantics"]["shape"]["fc"] = [4, 35]
        with torch.no_grad():
            self.actor.linears[selected.SELECTED_PATH].latent_sign[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            selected.enable_selected_sign(self.actor)
        self.assertEqual(flags, [p.requires_grad for p in self.actor.drafter.parameters()])

    def test_exact_detached_state_roundtrip_including_zero_and_negative_offsets(self):
        module = self.enable()
        with torch.no_grad():
            module.latent_sign[0, :4] = torch.tensor([-1.0, 0.0, -0.0, 1.0])
            self.actor.linears["fc"].scale_offset.copy_(torch.tensor([0.0, -0.9, 0.0, 0.01]))
        state = selected.collect_selected_state(self.actor)
        expected = copy.deepcopy(state)
        with torch.no_grad():
            module.latent_sign.fill_(0.5)
            for item in self.actor.linears.values():
                item.scale_offset.zero_()
        selected.load_selected_state(self.actor, state)
        actual = selected.collect_selected_state(self.actor)
        for key in ("latent_signs", "scale_offsets"):
            for base in expected[key]:
                self.assertTrue(torch.equal(actual[key][base], expected[key][base]))
                self.assertEqual(actual[key][base].device.type, "cpu")
                self.assertFalse(actual[key][base].requires_grad)
        state["latent_signs"][selected.SELECTED_BASES[0]].zero_()
        state["selection"]["selected_bases"].append("fc")
        self.assertFalse(torch.equal(module.latent_sign, torch.zeros_like(module.latent_sign)))
        self.assertEqual(
            self.actor._selected_sign_contract["selected_bases"], list(selected.SELECTED_BASES)
        )

    def test_invalid_late_payloads_and_ancestry_never_partially_load(self):
        module = self.enable()
        good = selected.collect_selected_state(self.actor)
        initial = module.latent_sign.detach().clone()
        for kind in (
            "initializer",
            "config",
            "contract",
            "selection",
            "mask",
            "shape",
            "nan",
            "dtype",
            "grad",
            "bound",
            "schema",
        ):
            with self.subTest(kind=kind):
                state = copy.deepcopy(good)
                state["latent_signs"][selected.SELECTED_BASES[0]].mul_(-1)
                state["scale_offsets"]["fc"].fill_(1)
                if kind in ("initializer", "config", "contract"):
                    state["ancestry"][kind]["sha256"] = "0" * 64
                elif kind == "selection":
                    state["selection"]["helper"]["sha256"] = "0" * 64
                elif kind == "mask":
                    state["scale_offsets"]["unexpected"] = torch.zeros(4)
                elif kind == "shape":
                    state["scale_offsets"]["output"] = torch.zeros(3)
                elif kind == "nan":
                    state["scale_offsets"]["output"][0] = float("nan")
                elif kind == "dtype":
                    state["scale_offsets"]["output"] = torch.zeros(4, dtype=torch.float64)
                elif kind == "grad":
                    state["scale_offsets"]["output"].requires_grad_(True)
                elif kind == "bound":
                    state["latent_signs"][selected.SELECTED_BASES[0]][0, 0] = 1.1
                else:
                    state["schema"] = "other"
                with self.assertRaises(ValueError):
                    selected.load_selected_state(self.actor, state)
                self.assertTrue(torch.equal(module.latent_sign, initial))
                self.assertTrue(torch.equal(self.actor.linears["fc"].scale_offset, torch.zeros(4)))

    def test_aliasing_input_state_is_snapshotted_before_any_copy(self):
        module = self.enable()
        state = selected.collect_selected_state(self.actor)
        state["latent_signs"][selected.SELECTED_BASES[0]].mul_(-1)
        state["scale_offsets"]["fc"] = module.latent_sign[0, :4].detach()
        expected = state["scale_offsets"]["fc"].clone()
        selected.load_selected_state(self.actor, state)
        self.assertTrue(torch.equal(self.actor.linears["fc"].scale_offset, expected))

    def test_aliasing_projection_modules_rejected_before_opt_in(self):
        flags = [p.requires_grad for p in self.actor.drafter.parameters()]
        self.actor.linears["lm_head"] = self.actor.linears["fc"]
        with self.assertRaisesRegex(ValueError, "must not alias"):
            selected.enable_selected_sign(self.actor)
        self.assertEqual(flags, [p.requires_grad for p in self.actor.drafter.parameters()])

    def test_frozen_values_and_wrong_optimizer_scope_rejected(self):
        self.enable()
        for kind in ("sign", "initial_scale", "norm", "d2t", "t2d", "scope"):
            with self.subTest(kind=kind):
                if kind == "sign":
                    tensor = self.actor.linears["fc"].latent_sign
                elif kind == "initial_scale":
                    tensor = self.actor.linears["fc"].initial_scale
                elif kind == "norm":
                    tensor = self.actor.drafter.get_submodule(
                        next(iter(selected.NATIVE_NORM_PATHS.values()))
                    ).weight
                elif kind in ("d2t", "t2d"):
                    tensor = getattr(self.actor.drafter, kind)
                else:
                    tensor = self.actor.linears["fc"].latent_sign
                before = tensor.detach().clone()
                with torch.no_grad():
                    if kind == "scope":
                        tensor.requires_grad_(True)
                    elif kind == "t2d":
                        tensor[0] = ~tensor[0]
                    else:
                        tensor.view(-1)[0] += 1
                with self.assertRaises(ValueError):
                    selected.collect_selected_state(self.actor)
                with torch.no_grad():
                    tensor.copy_(before)
                    if kind == "scope":
                        tensor.requires_grad_(False)

    def test_clamp_only_selected_latents_and_reject_nan_before_mutation(self):
        module = self.enable()
        scales = [m.scale_offset.detach().clone() for m in self.actor.linears.values()]
        other = self.actor.linears["fc"].latent_sign.detach().clone()
        with torch.no_grad():
            module.latent_sign[0, :2] = torch.tensor([-2.0, 3.0])
        selected.clamp_selected_latents_(self.actor)
        self.assertTrue(torch.equal(module.latent_sign[0, :2], torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.equal(self.actor.linears["fc"].latent_sign, other))
        self.assertTrue(
            all(
                torch.equal(m.scale_offset, s)
                for m, s in zip(self.actor.linears.values(), scales, strict=True)
            )
        )
        with torch.no_grad():
            module.latent_sign[0, :2] = torch.tensor([float("nan"), 3.0])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            selected.clamp_selected_latents_(self.actor)
        self.assertEqual(float(module.latent_sign[0, 1].detach()), 3.0)

    def test_selected_hard_sign_ste_has_finite_nonzero_gradient(self):
        module = self.enable()
        module(torch.arange(35, dtype=torch.float32).unsqueeze(0) / 35).sum().backward()
        self.assertIsNotNone(module.latent_sign.grad)
        self.assertTrue(bool(torch.isfinite(module.latent_sign.grad).all()))
        self.assertGreater(int(module.latent_sign.grad.ne(0).sum()), 0)
        self.assertIsNone(self.actor.linears["fc"].latent_sign.grad)

    def test_zero_update_export_is_byte_identical(self):
        self.enable()
        destination = self.directory / "zero.gguf"
        report = selected.export_selected_sign_gguf(self.actor, destination)
        self.assertEqual(
            destination.read_bytes(), Path(self.actor.initializer["path"]).read_bytes()
        )
        self.assertTrue(report["all_other_bytes_unchanged"])
        self.assertFalse(report["native_qualified"])
        self.assertEqual(len(report["replaced_payloads"]), 10)

    def test_sign_flip_export_preserves_other_bytes_qk_order_signed_zero_and_tail(self):
        module = self.enable()
        with torch.no_grad():
            module.latent_sign[0, 0] = 0.0  # Negative -> positive bit0.
            module.latent_sign[0, 1] = -0.0  # Positive remains positive.
            module.latent_sign[1, 34] = -0.1  # Positive -> negative, final logical column.
            self.actor.linears["midlayer.self_attn.q_proj"].scale_offset[1] = 0.125
        destination = self.directory / "flip.gguf"
        report = selected.export_selected_sign_gguf(self.actor, destination)
        original = Path(self.actor.initializer["path"]).read_bytes()
        candidate = destination.read_bytes()
        declared = set()
        for item in report["replaced_payloads"]:
            declared.update(range(item["offset"], item["offset"] + item["bytes"]))
        self.assertTrue(
            all(
                i in declared
                for i, (a, b) in enumerate(zip(original, candidate, strict=True))
                if a != b
            )
        )
        tensors = {t.name: t for t in GGUFReader(str(destination)).tensors}
        decoded = selected.unpack_signs(tensors["blk.0.attn_output.w1a1_packed"].data, 35)
        np.testing.assert_array_equal(
            decoded, np.where(module.latent_sign.detach().numpy() < 0, -1, 1)
        )
        self.assertEqual(decoded[0, 0], 1)
        self.assertEqual(decoded[0, 1], 1)
        qmodule = self.actor.linears["midlayer.self_attn.q_proj"]
        order = actor_module.qk_row_indices(**self.actor.qk_layout["blk.0.attn_q"])
        np.testing.assert_array_equal(
            tensors["blk.0.attn_q.w1a1_scale"].data,
            qmodule.effective_scales().detach().numpy()[order],
        )
        np.testing.assert_array_equal(tensors["fixture_canary"].data, [31, 17, 99])
        metrics = selected.selected_sign_metrics(self.actor)
        self.assertEqual(metrics["hard_sign_flips"], 2)
        self.assertEqual(metrics["changed_latents"], 3)

    def test_displacement_metrics_against_initializer_and_saved_state(self):
        module = self.enable()
        reference = selected.collect_selected_state(self.actor)
        initial = selected.selected_sign_metrics(self.actor)
        self.assertEqual(initial["changed_latents"], 0)
        self.assertEqual(initial["hard_sign_flips"], 0)
        with torch.no_grad():
            module.latent_sign[0, 0] = 0.5
            module.latent_sign[0, 1] = 0.0
        current = selected.selected_sign_metrics(self.actor, reference)
        self.assertEqual(current["hard_sign_flips"], 1)
        self.assertEqual(current["changed_latents"], 2)
        self.assertAlmostEqual(current["mean_absolute_displacement"], 1.5 / 140)
        self.assertEqual(current["max_absolute_displacement"], 1.0)
        self.assertTrue(current["measured_from_current_tensors"])

    def test_payload_budget_is_explicit_estimate(self):
        self.enable()
        budget = selected.selected_state_budget(self.actor)
        self.assertEqual(budget["selected_sign_parameters"], 140)
        self.assertEqual(budget["scale_parameters"], 36)
        self.assertEqual(budget["F32_model_state_payload_bytes"], 704)
        self.assertEqual(budget["F32_model_plus_Adam_two_moments_payload_bytes"], 2112)
        self.assertTrue(budget["estimated_not_measured"])

    def test_export_refuses_overwrite_and_changed_initializer(self):
        self.enable()
        destination = self.directory / "protected.gguf"
        destination.write_bytes(b"keep me")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            selected.export_selected_sign_gguf(self.actor, destination)
        self.assertEqual(destination.read_bytes(), b"keep me")
        source = Path(self.actor.initializer["path"])
        with source.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "initializer changed"):
            selected.export_selected_sign_gguf(self.actor, self.directory / "changed.gguf")


if __name__ == "__main__":
    unittest.main()

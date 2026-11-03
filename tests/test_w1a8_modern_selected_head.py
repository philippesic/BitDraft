"""Tiny CPU oracle/export fixtures; no real head, model load or CUDA evidence."""

# ruff: noqa: E402
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "scripts/w1a8_generalization"),
    "/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py",
]
import modern_actor
import modern_selected_head as head
import modern_selected_sign
from test_w1a8_modern_selected_sign import small_actor

from w1a1_eagle.recurrent_binary import hard_sign_ste
from w1a1_eagle.recurrent_qat import hard_activation, shared_round_hard_signs


class SelectedHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.actor = small_actor(self.directory)
        self.base = self.directory / "scale-base.pt"
        self.base.write_bytes(b"pinned scale-only fixture checkpoint")
        self.manifest = self.directory / "fit-manifest.json"
        self.manifest.write_text("pinned FIT-only fixture manifest")
        self.records = [
            {"role": "fit", "group_id": "A", "labels": [1, 1, 3, 5, 99]},
            {"role": "fit", "group_id": "B", "labels": [3, 5]},
            {"role": "fit", "group_id": "C", "labels": [5, 7]},
        ]
        self.selection = head.select_head_rows(
            self.records,
            head.absolute_map(self.actor),
            source_pins=[modern_actor.file_pin(self.manifest)],
            count=2,
        )

    def enable(self, **kwargs):
        return head.SelectedHead(
            self.actor, self.selection, base_state_pin=modern_actor.file_pin(self.base), **kwargs
        )

    def test_selector_unique_groups_support_ties_and_no_dev_or_fresh(self):
        self.assertEqual(self.selection["target_ids"], [5, 3])
        self.assertEqual(self.selection["row_ids"], [2, 1])
        self.assertEqual(self.selection["unsupported_label_count"], 1)
        self.assertEqual(self.selection["selected_label_count"], 5)
        for role in ("development", "fresh", "quality", "performance"):
            records = copy.deepcopy(self.records)
            records[0]["role"] = role
            with self.assertRaisesRegex(ValueError, "FIT TRAIN"):
                head.select_head_rows(
                    records,
                    head.absolute_map(self.actor),
                    source_pins=[modern_actor.file_pin(self.manifest)],
                    count=2,
                )
        with self.assertRaisesRegex(ValueError, "not enough observed"):
            head.select_head_rows(
                self.records[:1],
                head.absolute_map(self.actor),
                source_pins=[modern_actor.file_pin(self.manifest)],
                count=4,
            )
        tied = [{"role": "fit", "group_id": "X", "labels": [7, 3]}]
        selection = head.select_head_rows(
            tied,
            head.absolute_map(self.actor),
            source_pins=[modern_actor.file_pin(self.manifest)],
            count=2,
        )
        self.assertEqual(selection["target_ids"], [3, 7])

    def test_zero_forward_exact_and_old_apis_stay_strict(self):
        x = torch.arange(210, dtype=torch.float32).reshape(2, 3, 35).mul_(0.013)
        original = self.actor.linears[head.HEAD_PATH](x).detach().clone()
        selected = self.enable()
        self.assertTrue(torch.equal(original, self.actor.linears[head.HEAD_PATH](x)))
        groups = selected.parameter_groups(sign_lr=0.03, scale_lr=0.001)
        self.assertEqual([len(g["params"]) for g in groups], [1, 9])
        self.assertEqual(tuple(groups[0]["params"][0].shape), (2, 35))
        self.assertFalse(self.actor.linears[head.HEAD_PATH].latent_sign.requires_grad)
        with self.assertRaisesRegex(ValueError, "scale-only"):
            modern_actor.collect_scale_state(self.actor)
        with self.assertRaisesRegex(ValueError, "opt-in"):
            modern_selected_sign.collect_selected_state(self.actor)
        output = self.directory / "zero.gguf"
        result = selected.export(output)
        self.assertTrue(result["all_other_bytes_unchanged"])
        self.assertEqual(output.read_bytes(), Path(self.actor.initializer["path"]).read_bytes())

    def test_dense_oracle_forward_backward_input_selected_and_nine_scales(self):
        selected = self.enable()
        generator = torch.Generator().manual_seed(51)
        x = torch.randn(2, 3, 35, generator=generator, requires_grad=True)
        oracle_x = x.detach().clone().requires_grad_()
        small = selected.latent.detach().clone().requires_grad_()
        # Include STE endpoints and zero-positive semantics.
        with torch.no_grad():
            selected.latent[0, :5] = torch.tensor([-1.0, -0.0, 0.0, 1.0, 0.2])
            small.copy_(selected.latent)
        offsets = {
            p: m.scale_offset.detach().clone().requires_grad_()
            for p, m in self.actor.linears.items()
        }
        expected, actual = [], []
        for path, module in self.actor.linears.items():
            if path == head.HEAD_PATH:
                signs = hard_sign_ste(module.latent_sign).index_copy(
                    0, selected.rows, hard_sign_ste(small)
                )
            else:
                signs = hard_sign_ste(module.latent_sign)
            quantized = hard_activation(oracle_x, 8)[0]
            scales = torch.where(
                module.initial_scale + offsets[path] >= 0,
                module.initial_scale + offsets[path],
                torch.zeros_like(offsets[path]),
            )
            expected.append(F.linear(quantized, signs) * scales + 0)
        with selected.round():
            for module in self.actor.linears.values():
                actual.append(module(x))
            for a, b in zip(actual, expected, strict=True):
                self.assertTrue(torch.equal(a, b))
            sum(t.square().sum() for t in actual).backward()
        sum(t.square().sum() for t in expected).backward()
        torch.testing.assert_close(x.grad, oracle_x.grad, rtol=1e-6, atol=2e-5)
        torch.testing.assert_close(selected.latent.grad, small.grad, rtol=1e-6, atol=2e-5)
        for path, module in self.actor.linears.items():
            self.assertIsNone(module.latent_sign.grad)
            torch.testing.assert_close(module.scale_offset.grad, offsets[path].grad, rtol=0, atol=0)

    def test_recurrent_calls_share_signs_and_keep_input_state_gradients(self):
        selected = self.enable()
        x = torch.arange(70, dtype=torch.float32).reshape(2, 35).div(90).requires_grad_()
        oracle_x = x.detach().clone().requires_grad_()
        masters = selected.latent.detach().clone().requires_grad_()
        scales = selected.head.effective_scales().detach().clone().requires_grad_()
        signs = hard_sign_ste(selected.head.latent_sign).index_copy(
            0, selected.rows, hard_sign_ste(masters)
        )

        def extend(value):
            return torch.cat([value.repeat(1, 8), value[:, :3]], dim=-1)

        with selected.round():
            first = selected.head(x)
            second = selected.head(extend(first))
            (first.square().sum() + second.square().sum()).backward()
        a = F.linear(hard_activation(oracle_x, 8)[0], signs) * scales + 0
        b = F.linear(hard_activation(extend(a), 8)[0], signs) * scales + 0
        (a.square().sum() + b.square().sum()).backward()
        self.assertTrue(torch.equal(first, a))
        self.assertTrue(torch.equal(second, b))
        torch.testing.assert_close(x.grad, oracle_x.grad, rtol=1e-6, atol=1e-4)
        torch.testing.assert_close(selected.latent.grad, masters.grad, rtol=1e-6, atol=1e-4)
        torch.testing.assert_close(
            selected.head.scale_offset.grad, scales.grad, rtol=1e-6, atol=1e-4
        )

    def test_no_selected_grad_or_adam_state_in_control(self):
        selected = self.enable(learn_signs=False)
        groups = selected.parameter_groups(scale_lr=0.001)
        self.assertEqual([g["name"] for g in groups], ["nine_raw_scale_offsets"])
        optimizer = torch.optim.Adam(groups)
        before = selected.latent.detach().clone()
        with selected.round():
            self.actor.linears[head.HEAD_PATH](torch.ones(3, 35)).sum().backward()
        optimizer.step()
        self.assertIsNone(selected.latent.grad)
        self.assertNotIn(selected.latent, optimizer.state)
        self.assertTrue(torch.equal(before, selected.latent))

    def test_clipped_identity_ste_inclusive_endpoints_and_signed_zero(self):
        selected = self.enable()
        with torch.no_grad():
            selected.latent.fill_(1.1)
            selected.latent[0, :3] = torch.tensor([-1.0, 1.0, -0.0])
        self.actor.linears[head.HEAD_PATH](torch.ones(1, 35)).sum().backward()
        torch.testing.assert_close(selected.latent.grad[0, :3], torch.full((3,), 0.5))
        self.assertTrue(torch.equal(selected.latent.grad[0, 3:], torch.zeros(32)))
        self.assertTrue(torch.equal(selected.latent.grad[1], torch.zeros(35)))
        selected.clamp_()
        self.assertTrue(bool((selected.latent.abs() <= 1).all()))

    def test_disabled_control_clipping_moments_and_export_match_scale_only(self):
        reference = copy.deepcopy(self.actor)
        modern_actor.enable_scale_only(reference)
        selected = self.enable(learn_signs=False)
        a = torch.optim.Adam(selected.parameter_groups(scale_lr=0.001))
        b = torch.optim.Adam(reference.scale_parameters(), lr=0.001)
        x = torch.arange(105, dtype=torch.float32).reshape(3, 35) / 100
        with selected.round():
            sum(module(x).square().sum() for module in self.actor.linears.values()).backward()
        sum(module(x).square().sum() for module in reference.linears.values()).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in a.param_groups for p in group["params"]], 0.01
        )
        torch.nn.utils.clip_grad_norm_(reference.scale_parameters(), 0.01)
        a.step()
        b.step()
        for path, module in self.actor.linears.items():
            other = reference.linears[path]
            self.assertTrue(torch.equal(module.scale_offset, other.scale_offset))
            self.assertTrue(
                torch.equal(
                    a.state[module.scale_offset]["exp_avg"], b.state[other.scale_offset]["exp_avg"]
                )
            )
            self.assertTrue(
                torch.equal(
                    a.state[module.scale_offset]["exp_avg_sq"],
                    b.state[other.scale_offset]["exp_avg_sq"],
                )
            )
        first, second = self.directory / "control-head.gguf", self.directory / "control-scales.gguf"
        selected.export(first)
        modern_actor.export_scale_only_gguf(reference, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_round_nesting_exception_pending_and_update_guards(self):
        selected = self.enable()
        with selected.round():
            with self.assertRaisesRegex(ValueError, "nested"):
                with selected.round():
                    pass
        self.assertIsNone(selected._cache)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with selected.round():
                raise RuntimeError("fixture failure")
        self.assertIsNone(selected._cache)
        self.assertTrue(all(m._round_hard_signs is None for m in self.actor.linears.values()))
        with self.assertRaisesRegex(ValueError, "consume backward"):
            with selected.round():
                pending = self.actor.linears[head.HEAD_PATH](torch.ones(1, 35)).sum()
        with self.assertRaisesRegex(ValueError, "stale"):
            pending.backward()
        with selected.round():
            with torch.no_grad():
                selected.latent.add_(0.001)
            with self.assertRaisesRegex(ValueError, "update within"):
                self.actor.linears[head.HEAD_PATH](torch.ones(1, 35))
        with shared_round_hard_signs(self.actor.linears):
            with self.assertRaisesRegex(ValueError, "ordinary shared"):
                self.actor.linears[head.HEAD_PATH](torch.ones(1, 35))

    def test_sparse_state_full_validation_before_mutation(self):
        selected = self.enable()
        state = selected.state()
        self.assertEqual(state["selected_latent"].numel(), 70)
        mutations = []
        bad = copy.deepcopy(state)
        bad["contract"]["selection"]["row_ids"] = [1, 1]
        mutations.append(bad)
        bad = copy.deepcopy(state)
        bad["contract"]["ancestry"]["initializer"]["sha256"] = "bad"
        mutations.append(bad)
        bad = copy.deepcopy(state)
        bad["selected_latent"] = torch.zeros(4, 35)
        mutations.append(bad)
        bad = copy.deepcopy(state)
        bad["selected_latent"] = state["selected_latent"].double()
        mutations.append(bad)
        bad = copy.deepcopy(state)
        bad["selected_latent"][0, 0] = 1.01
        mutations.append(bad)
        bad = copy.deepcopy(state)
        bad["scale_offsets"]["output"][0] = float("nan")
        mutations.append(bad)
        bad = copy.deepcopy(state)
        bad["scale_offsets"]["output"] = torch.zeros(8)[::2]
        mutations.append(bad)
        for bad in mutations:
            before = selected.state()
            with self.assertRaises(ValueError):
                selected.load_state(bad)
            self.assertTrue(torch.equal(selected.latent, before["selected_latent"]))
            for base, path in head.CANDIDATE_D_BASE_TO_PATH.items():
                self.assertTrue(
                    torch.equal(
                        self.actor.linears[path].scale_offset, before["scale_offsets"][base]
                    )
                )
        with torch.no_grad():
            selected.latent.add_(0.03)
            self.actor.linears[head.HEAD_PATH].scale_offset.add_(0.01)
        selected.load_state(state)
        self.assertTrue(torch.equal(selected.latent, state["selected_latent"]))

    def test_selected_packed_flip_only_and_zero_positive_tail(self):
        selected = self.enable()
        before = Path(self.actor.initializer["path"]).read_bytes()
        with torch.no_grad():
            # Row2,col0 begins negative; -0 must export as a positive sign.
            selected.latent[0, 0] = -0.0
            selected.latent[1, -1] = -selected.latent[1, -1]
        output = self.directory / "flip.gguf"
        result = selected.export(output)
        after = output.read_bytes()
        changes = {i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b}
        ranges = [
            r for r in result["replaced_payloads"] if r["tensor"].startswith("output.w1a1_packed[")
        ]
        allowed = {i for r in ranges for i in range(r["offset"], r["offset"] + r["bytes"])}
        self.assertTrue(changes and changes <= allowed)
        self.assertEqual(len(changes), 2)
        self.assertEqual(selected.budget()["selected_packed_patch_bytes"], 16)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            selected.export(output)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            head._publish_patches(
                Path(self.actor.initializer["path"]),
                self.actor.initializer,
                self.directory / "bad.gguf",
                [("a", 10, b"1234"), ("b", 12, b"xx")],
                {},
            )

    def test_selected_adam_memory_and_exact_resume_next_step(self):
        selected = self.enable()
        optimizer = torch.optim.Adam(selected.parameter_groups(sign_lr=0.01, scale_lr=0.001))

        def step():
            optimizer.zero_grad(set_to_none=True)
            with selected.round():
                self.actor.linears[head.HEAD_PATH](
                    torch.arange(35, dtype=torch.float32)[None] / 30
                ).square().sum().backward()
            optimizer.step()
            selected.clamp_()

        step()
        state, moments = selected.state(), copy.deepcopy(optimizer.state_dict())
        self.assertEqual(optimizer.state[selected.latent]["exp_avg"].shape, (2, 35))
        self.assertNotIn(self.actor.linears[head.HEAD_PATH].latent_sign, optimizer.state)
        step()
        expected = selected.state()
        selected.load_state(state)
        optimizer.load_state_dict(moments)
        step()
        self.assertTrue(torch.equal(selected.latent, expected["selected_latent"]))
        for base, path in head.CANDIDATE_D_BASE_TO_PATH.items():
            self.assertTrue(
                torch.equal(self.actor.linears[path].scale_offset, expected["scale_offsets"][base])
            )

    def test_initializer_frozen_sign_norm_map_and_source_pins_checked(self):
        selected = self.enable()
        with torch.no_grad():
            self.actor.linears[head.HEAD_PATH].latent_sign[0, 0] *= -1
        with self.assertRaisesRegex(ValueError, "frozen signs"):
            selected.state()

    def test_frozen_norm_map_and_file_changes_rejected(self):
        selected = self.enable()
        norm_path = next(iter(modern_selected_sign.NATIVE_NORM_PATHS.values()))
        norm = self.actor.drafter.get_submodule(norm_path).weight
        with torch.no_grad():
            norm[0] += 0.01
        with self.assertRaisesRegex(ValueError, "frozen norm"):
            selected.state()
        with torch.no_grad():
            norm[0] = 1.0
            self.actor.drafter.d2t[0] += 1
        with self.assertRaisesRegex(ValueError, "mapping|vocabulary map"):
            selected.state()
        with torch.no_grad():
            self.actor.drafter.d2t[0] -= 1
        self.manifest.write_text("changed source manifest")
        with self.assertRaisesRegex(ValueError, "pin changed"):
            selected.state()

    def test_invalid_selection_fails_before_actor_mutation(self):
        self.selection["row_ids"] = [1, 1]
        flags = [p.requires_grad for p in self.actor.drafter.parameters()]
        with self.assertRaisesRegex(ValueError, "row IDs"):
            self.enable()
        self.assertEqual([p.requires_grad for p in self.actor.drafter.parameters()], flags)
        self.assertNotIn("forward", self.actor.linears[head.HEAD_PATH].__dict__)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from typing import cast

import gguf
import torch
from torch import nn

from tests.numerical.test_features import SAMPLE, seeded, snapshot
from tests.numerical.test_model import (
    OMITTED_RECIPE,
    execute_leased,
    loader_fixture,
    row_optimization,
    tiny_model,
)
from tests.numerical.test_rounds import trained_like
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, number, record
from w1a1_eagle.data import Supervision, TailWindow, Window
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.model import (
    HE4,
    Layer,
    Projection,
    anchor_bounds,
    bound_latents,
    named_projections,
    slot_rounding,
)
from w1a1_eagle.numerical.calibration import (
    calibrate,
    choose_bound,
    layer_projections,
    recorded_inputs,
    select,
    slot_rows,
)
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.probes import RangeProbe
from w1a1_eagle.numerical.probes import Window as ProbeWindow
from w1a1_eagle.numerical.worker import TensorWorker, feature_parameters
from w1a1_eagle.records import canonical, reference
from w1a1_eagle.training import Budget, Features, Optimization, default_step_sizes

TOKENS = tuple(range(20))
MEMORY = 64 * 1024 * 1024


def slot_worker(
    identity: str, anchors: bool, features: Features | None = None, telemetry: bool = False
) -> TensorWorker:
    model, inputs = tiny_model()
    windows: tuple[Supervision, ...] = (
        Window("fixture", 2, tuple(range(3, 18))),
        Window("fixture", 4, tuple(range(5, 20))),
        TailWindow("fixture", 17, (18, 19)),
    )
    capture = Capture(TOKENS, inputs, windows)

    class Source:
        def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]:
            return capture, windows[cursor % 2 :]

    bound_latents(model, 3.0)
    with torch.no_grad():
        for _, projection in named_projections(model):
            projection.log_clip.fill_(-0.5)
    if anchors:
        anchor_bounds(model)
    return TensorWorker(
        model,
        Source(),
        identity,
        optimization=Optimization(
            {**default_step_sizes(), "sign": 0.05, "scale": 0.01, "clip": 0.05},
            latent_multiple=3.0,
        ),
        features=features,
        telemetry=telemetry,
    )


def outputs(model: HE4, features: torch.Tensor) -> list[torch.Tensor]:
    with torch.no_grad():
        return [
            model.group_hidden(features, TOKENS, (2, 5, 9), 4),
            model.group_hidden(features, TOKENS, (3, 4), 16),
            model.forward(features, TOKENS, 9, 4),
            model.forward(features, TOKENS, 3, 16),
        ]


def native_codes(
    value: torch.Tensor, gain: torch.Tensor, clip: float, anchor: float, period: int, bits: int
) -> torch.Tensor:
    maximum = (1 << (bits - 1)) - 1
    rows = torch.arange(value.shape[0]) % period == 0
    bound = torch.where(
        rows[:, None],
        torch.tensor(anchor, dtype=torch.float32),
        torch.tensor(clip, dtype=torch.float32),
    )
    inverse = torch.tensor(float(maximum), dtype=torch.float32) / bound
    scaled = (value.float() * gain) * inverse
    return scaled.clamp(-maximum, maximum).round()


class TestSlotRounding(unittest.TestCase):
    def test_anchor_bounds_start_identical_and_split_by_slot(self) -> None:
        model, features = tiny_model()
        trained_like(model)
        before = outputs(model, features)
        self.assertFalse(slot_rounding(model))
        anchor_bounds(model)
        self.assertTrue(slot_rounding(model))
        for index, (old, new) in enumerate(zip(before, outputs(model, features), strict=True)):
            self.assertTrue(torch.equal(old, new), index)
        names = {name for name, _ in model.named_parameters()}
        block = cast(Layer, model.blocks[0])
        self.assertIs(block.gate.log_anchor_clip, block.up.log_anchor_clip)
        self.assertIsNot(block.gate.log_anchor_clip, block.down.log_anchor_clip)
        self.assertIn("blocks.0.gate.log_anchor_clip", names)
        self.assertIn("blocks.0.down.log_anchor_clip", names)
        with self.assertRaisesRegex(ValueError, "already enabled"):
            anchor_bounds(model)
        with torch.no_grad():
            for _, projection in named_projections(model):
                assert projection.log_anchor_clip is not None
                projection.log_anchor_clip.fill_(-1.5)
        changed = outputs(model, features)
        for old, new in zip(before, changed, strict=True):
            self.assertFalse(torch.equal(old, new))
        grouped = torch.nn.functional.linear(changed[0][2], model.vocabulary)
        self.assertTrue(torch.allclose(grouped, changed[2], atol=1e-5))
        with torch.no_grad():
            single = model.forward(features, TOKENS, 4, 16)
        long = torch.nn.functional.linear(changed[1][1], model.vocabulary)
        self.assertTrue(torch.allclose(long, single, atol=1e-5))

    def test_quantizer_uses_the_anchor_bound_on_each_first_row_only(self) -> None:
        generator = torch.Generator().manual_seed(11)
        for bits in (4, 8):
            weight = torch.randn((5, 32), generator=generator)
            projection = Projection(
                weight,
                bits,
                nn.Parameter(torch.randn(32, generator=generator) * 0.2),
                nn.Parameter(torch.tensor(0.2)),
            )
            value = torch.randn((12, 32), generator=generator) * 2
            plain_codes, plain_step = projection.quantize(value, 4)
            projection.log_anchor_clip = nn.Parameter(torch.tensor(-0.7))
            codes, step = projection.quantize(value, 4)
            anchored = Projection(
                weight, bits, projection.log_gain, nn.Parameter(torch.tensor(-0.7))
            )
            anchor_codes, anchor_step = anchored.quantize(value)
            first = torch.arange(12) % 4 == 0
            self.assertTrue(torch.equal(codes[first], anchor_codes[first]))
            self.assertTrue(torch.equal(codes[~first], plain_codes[~first]))
            self.assertTrue(torch.equal(step[first].reshape(-1), anchor_step.expand(3)))
            self.assertTrue(torch.equal(step[~first].reshape(-1), plain_step.expand(9)))
            native = native_codes(
                value.detach(),
                projection.log_gain.detach().exp(),
                float(projection.log_clip.detach().exp()),
                float(projection.log_anchor_clip.detach().exp()),
                4,
                bits,
            )
            self.assertTrue(torch.equal(codes.detach().round(), native))
            for selected, trained, held in (
                (first, projection.log_anchor_clip, projection.log_clip),
                (~first, projection.log_clip, projection.log_anchor_clip),
            ):
                projection.zero_grad(set_to_none=True)
                projection.forward(value, 4)[selected].square().sum().backward()
                assert trained.grad is not None and held.grad is not None
                self.assertNotEqual(float(trained.grad), 0.0)
                self.assertEqual(float(held.grad), 0.0)
            for block in (None, 5, 0):
                with self.assertRaisesRegex(ValueError, "whole blocks"):
                    projection.quantize(value, block)

    def test_grouped_updates_train_both_bounds_resume_and_respect_switches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-slots-") as directory:
            root = Path(directory)
            original = slot_worker("slots", True, telemetry=True)
            clip_group = next(
                group for group in original.optimizer.param_groups if group["name"] == "clip"
            )
            anchors = [
                projection.log_anchor_clip for _, projection in named_projections(original.model)
            ]
            held = {id(value) for value in clip_group["params"]}
            self.assertTrue(all(value is not None and id(value) in held for value in anchors))
            self.assertEqual(len(held), 24)
            original.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            parent = original.checkpoint(Store(root / "slots"), 4 * 1024 * 1024)
            original.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            report = original.telemetry_record()
            assert report is not None
            bounds = record(record(report["inputs"])["blocks.0.gate"])
            gate = named_projections(original.model)[0][1]
            self.assertEqual(number(bounds["clip_bound"]), float(gate.log_clip.detach().exp()))
            assert gate.log_anchor_clip is not None
            self.assertEqual(
                number(bounds["anchor_clip_bound"]), float(gate.log_anchor_clip.detach().exp())
            )
            assert gate.log_anchor_clip is not None
            anchor, mask = float(gate.log_anchor_clip.detach()), float(gate.log_clip.detach())
            self.assertNotEqual(anchor, -0.5)
            self.assertNotEqual(mask, -0.5)
            self.assertNotEqual(anchor, mask)
            resumed = slot_worker("slots", True, telemetry=True)
            resumed.restore(parent, 8 * 1024 * 1024)
            resumed.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            expected = dict(original.model.named_parameters())
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]), name)
            with self.assertRaisesRegex(ValueError, "coverage"):
                slot_worker("plain", False).restore(parent, 8 * 1024 * 1024, "branch")
            fixed = slot_worker("fixed", True, Features(clip="fixed"))
            frozen = feature_parameters(fixed.model, "clip")
            self.assertEqual(len(frozen), 24)
            kept = [value.detach().clone() for value in frozen]
            fixed.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            for value, initial in zip(frozen, kept, strict=True):
                self.assertFalse(value.requires_grad)
                self.assertTrue(torch.equal(value, initial))

    def test_branch_from_a_shared_bound_parent_starts_identical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-slots-branch-") as directory:
            root = Path(directory)
            plain = slot_worker("plain", False)
            with torch.no_grad():
                for _, projection in named_projections(plain.model):
                    projection.log_clip.fill_(-0.25)
            plain.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            parent = plain.checkpoint(Store(root / "plain"), 4 * 1024 * 1024)
            _, features = tiny_model()
            expected = outputs(plain.model, features)
            child = slot_worker("child", True)
            child.restore(parent, 8 * 1024 * 1024, "branch")
            for name, projection in named_projections(child.model):
                assert projection.log_anchor_clip is not None
                self.assertTrue(torch.equal(projection.log_anchor_clip, projection.log_clip), name)
            for old, new in zip(expected, outputs(child.model, features), strict=True):
                self.assertTrue(torch.equal(old, new))
            child.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            for mode in ("resume", "continue"):
                with self.assertRaisesRegex(ValueError, "coverage|recipe"):
                    slot_worker("plain", True).restore(parent, 8 * 1024 * 1024, mode)

    def test_range_probe_reports_each_slot_against_its_own_bound(self) -> None:
        model, features = tiny_model()
        trained_like(model)
        anchor_bounds(model)
        with torch.no_grad():
            for _, projection in named_projections(model):
                assert projection.log_anchor_clip is not None
                projection.log_anchor_clip.fill_(-2.0)
            probe = RangeProbe(model, 4)
            probe.window = ProbeWindow("one", 9, 1)
            with probe.attached():
                logits = model.forward(features, TOKENS, 9, 4)
            self.assertTrue(torch.equal(logits, model.forward(features, TOKENS, 9, 4)))
        measured = record(record(probe.record()["layer-0"])["gate_up"])
        self.assertEqual(number(measured["anchor_bound"]), float(torch.tensor(-2.0).exp()))
        self.assertEqual(number(measured["bound"]), float(torch.tensor(-0.5).exp()))

    def test_calibration_fits_each_bound_on_its_own_slots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-slots-calibration-") as directory:
            root = Path(directory)
            corpus, model, _ = seeded(root)
            _, plain, _ = seeded(root / "plain")
            anchor_bounds(model)
            selection = select(corpus, SAMPLE, MEMORY)
            gate = layer_projections(model)[0][1][0]
            (inputs,) = recorded_inputs(model, selection, 4, [gate])
            tables = torch.cat(
                [gate.latent_sign.detach(), layer_projections(model)[0][1][1].latent_sign.detach()]
            )
            mask = choose_bound(gate, slot_rows(inputs, 4, False), tables)
            anchor = choose_bound(gate, slot_rows(inputs, 4, True), tables)
            self.assertNotEqual(mask, anchor)
            report = calibrate(model, corpus, SAMPLE, Features(), 4, MEMORY, False)
            shared = calibrate(plain, corpus, SAMPLE, Features(), 4, MEMORY, False)
            assert gate.log_anchor_clip is not None
            self.assertAlmostEqual(
                float(gate.log_anchor_clip.detach().exp()), anchor, delta=anchor * 1e-6
            )
            self.assertAlmostEqual(float(gate.log_clip.detach().exp()), mask, delta=mask * 1e-6)
            first = record(record(report["tables"])["blocks.0.gate"])
            self.assertIn("anchor_clip_bound", first)
            self.assertNotIn("anchor_clip_bound", record(record(shared["tables"])["blocks.0.gate"]))
            self.assertIn("slot_rounding", record(report["fitted"]))
            self.assertNotIn("slot_rounding", record(shared["fitted"]))
            self.assertNotEqual(report["values_sha256"], shared["values_sha256"])


class TestSlotRoundingRun(unittest.TestCase):
    def test_slot_rounding_run_exports_both_bounds_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-slots-entry-") as directory:
            root = Path(directory)
            _, _, base = seeded(root)
            intent: Record = {
                **base,
                "optimization": row_optimization(),
                "budget": {"updates": 2, "wall_seconds": 60, "attempts": 3, "retries": 0},
                "science_policy": {
                    "id": "synthetic-fixture-only",
                    "min_unique_positions": 16,
                    "max_mean_visits": 8,
                },
                "calibration": {"sample": SAMPLE.encode()},
                "telemetry": True,
                "slot_rounding": "anchor-mask",
            }
            path = root / "intent.json"
            path.write_bytes(canonical(intent))
            first = execute_leased(path, root / "first")
            events = list(Journal(root / "first").events())
            admitted = next(row for row in events if row.get("kind") == "admission")
            self.assertEqual(admitted["slot_rounding"], "anchor-mask")
            calibrated = next(row for row in events if row.get("kind") == "calibration")
            self.assertIn(
                "anchor_clip_bound", record(record(calibrated["tables"])["blocks.0.down"])
            )
            states = Store(root / "first" / "states").latest(first.recipe)
            exports = Store(root / "first" / "exports").latest(first.recipe)
            assert states is not None and exports is not None
            saved = record(torch.load(states.artifact.path, weights_only=True)["model"])
            reader = gguf.GGUFReader(str(exports.artifact.path))
            for index in range(1):
                for part in ("gate", "up", "down"):
                    prefix = f"blocks.{index}.{part}."
                    base_key = f"blk.{index}.ffn_{part}.w1ax."
                    for key, name in (("clip", "log_clip"), ("clip_anchor", "log_anchor_clip")):
                        tensor = cast(torch.Tensor, saved[prefix + name])
                        exported = reader.fields[base_key + key].contents()
                        self.assertEqual(exported, float(tensor.float().exp()))
            intent.update(
                mode="resume",
                parent=reference(states.artifact),
                budget={"updates": 3, "wall_seconds": 60, "attempts": 4, "retries": 0},
            )
            path.write_bytes(canonical(intent))
            resumed = execute_leased(path, root / "second")
            self.assertEqual(resumed.recipe, first.recipe)

    def test_omitted_setting_keeps_the_recipe_and_invalid_values_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-slots-intent-") as directory:
            root = Path(directory)
            loaded = loader_fixture(root)
            fixed = {
                key: {
                    "path": str(record(loaded.intent[key])["path"]),
                    "sha256": str(i) * 64,
                    "bytes": 7,
                }
                for i, key in enumerate(("source", "vocabulary", "corpus", "initialization"), 1)
            }
            base: Record = {**loaded.intent, **fixed, "code_sha256": "0" * 64}
            path = root / "intent.json"

            def parse(changes: Record) -> Intent:
                path.write_bytes(canonical({**base, **changes}))
                return Intent.load(path)

            omitted = parse({})
            self.assertEqual(omitted.recipe, OMITTED_RECIPE)
            self.assertIsNone(omitted.slot_rounding)
            enabled = parse({"slot_rounding": "anchor-mask"})
            self.assertEqual(enabled.slot_rounding, "anchor-mask")
            self.assertNotEqual(enabled.recipe, OMITTED_RECIPE)
            for invalid in ("per-slot", None, True):
                with self.assertRaisesRegex(ValueError, "slot_rounding"):
                    parse({"slot_rounding": invalid})

    def test_default_model_has_no_anchor_parameters(self) -> None:
        model, _ = tiny_model()
        names = [name for name, _ in model.named_parameters(remove_duplicate=False)]
        self.assertFalse(any("anchor" in name for name in names))
        self.assertFalse(any("anchor" in name for name in model.state_dict()))
        self.assertEqual(
            sorted(snapshot(model)), sorted(name for name, _ in model.named_parameters())
        )
        self.assertTrue(
            all(projection.log_anchor_clip is None for _, projection in named_projections(model))
        )

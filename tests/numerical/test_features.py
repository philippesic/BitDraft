import math
import tempfile
import unittest
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch
from torch import nn

from tests.numerical.test_model import (
    LOADER_GEOMETRY,
    OMITTED_RECIPE,
    execute_leased,
    gguf_fixture,
    loader_fixture,
    row_optimization,
    tiny_model,
)
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, number, record
from w1a1_eagle.data import Supervision, TailWindow, Window
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.model import (
    HE4,
    Layer,
    Projection,
    bound_latents,
    named_projections,
    soft_weight,
)
from w1a1_eagle.numerical.calibration import (
    calibrate,
    choose_bound,
    fit_rows,
    rounding_error,
    start_check,
)
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.entry import start_check_command
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker, feature_parameters, latent_parameters
from w1a1_eagle.records import canonical, reference
from w1a1_eagle.training import (
    Arithmetic,
    Budget,
    Features,
    GradualPrecision,
    Optimization,
    Sample,
    Schedule,
    StartCheck,
    default_step_sizes,
)

BLANK = Features("fixed", "fixed", "off", "off")
SAMPLE = Sample(1, 2, 3)


def precision(soft_updates: int = 2, maximum: float = 0.05, mean: float = 0.01) -> GradualPrecision:
    return GradualPrecision(
        0.01, 8.0, "exponential", soft_updates, 2e-3, StartCheck(SAMPLE, maximum, mean)
    )


def gradual_record(soft_updates: int, maximum: float = 0.05) -> Record:
    return {
        "start_temperature": 0.01,
        "end_temperature": 8.0,
        "shape": "exponential",
        "soft_updates": soft_updates,
        "step_size": 0.002,
        "start_check": {
            "sample": {"sequences": 1, "windows": 2, "seed": 3},
            "max_relative_error": maximum,
            "mean_relative_error": 0.01,
        },
    }


def worker(
    identity: str,
    features: Features | None = None,
    soft: GradualPrecision | None = None,
    telemetry: bool = False,
    optimization: Optimization | None = None,
    arithmetic: Arithmetic = "reference",
) -> TensorWorker:
    model, inputs = tiny_model()
    windows: tuple[Supervision, ...] = (
        Window("fixture", 2, tuple(range(3, 18))),
        Window("fixture", 4, tuple(range(5, 20))),
        TailWindow("fixture", 17, (18, 19)),
    )
    capture = Capture(tuple(range(20)), inputs, windows)

    class Source:
        def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]:
            return capture, windows[cursor % 2 :]

    recipe = optimization or Optimization(
        {**default_step_sizes(), "sign": 0.05, "scale": 0.01, "midpoint": 0.01},
        latent_multiple=3.0,
    )
    if soft is None and recipe.latent_multiple is not None:
        bound_latents(model, recipe.latent_multiple)
    return TensorWorker(
        model,
        Source(),
        identity,
        optimization=recipe,
        features=features,
        precision=soft,
        telemetry=telemetry,
        arithmetic=arithmetic,
    )


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def optimized(target: TensorWorker) -> set[int]:
    return {
        id(parameter) for group in target.optimizer.param_groups for parameter in group["params"]
    }


def seeded(root: Path) -> tuple[Corpus, HE4, Record]:
    root.mkdir(parents=True, exist_ok=True)
    loaded = loader_fixture(root)
    (root / "seeded").mkdir()
    source, vocabulary = gguf_fixture(root / "seeded", LOADER_GEOMETRY, seed=5)
    corpus = Corpus(loaded.corpus, 17, 64 * 1024 * 1024, 256, shape=LOADER_GEOMETRY)
    model = load_he4(source, vocabulary, 64 * 1024 * 1024, LOADER_GEOMETRY)
    intent = {key: value for key, value in loaded.intent.items() if key != "initialization"}
    return (
        corpus,
        model,
        {**intent, "source": reference(source), "vocabulary": reference(vocabulary)},
    )


class TestFeatureSwitches(unittest.TestCase):
    def test_each_switch_excludes_parameters_and_keeps_values(self) -> None:
        cases = {
            "scale": Features(scale="fixed"),
            "clip": Features(clip="fixed"),
            "gain": Features(gain="off"),
            "midpoint": Features(midpoint="off"),
        }
        for group, features in cases.items():
            target = worker("switch-" + group, features)
            frozen = feature_parameters(target.model, group)
            before = [value.detach().clone() for value in frozen]
            others = snapshot(target.model)
            self.assertTrue(frozen)
            self.assertFalse(optimized(target) & {id(value) for value in frozen}, group)
            self.assertNotIn(group, {g["name"] for g in target.optimizer.param_groups})
            target.run(Monotonic(), Budget(3, 60, 3), lambda: False)
            for value, initial in zip(frozen, before, strict=True):
                self.assertTrue(torch.equal(value, initial), group)
                self.assertNotIn(value, target.optimizer.state)
            if group in ("gain", "midpoint"):
                self.assertTrue(all(not bool(value.any()) for value in frozen))
            changed = [
                name
                for name, value in target.model.named_parameters()
                if not torch.equal(value, others[name])
            ]
            self.assertTrue(any(name.endswith("latent_sign") for name in changed), group)
        full = worker("full")
        self.assertEqual(
            {g["name"] for g in full.optimizer.param_groups},
            {"sign", "scale", "midpoint", "gain", "clip", "norm"},
        )
        blank = worker("blank", BLANK)
        self.assertEqual({g["name"] for g in blank.optimizer.param_groups}, {"sign", "norm"})

    def test_switch_states_resume_branch_and_refuse_nonidentity_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = worker("blank", BLANK)
            original.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            parent = original.checkpoint(Store(root / "blank"), 4 * 1024 * 1024)
            original.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            resumed = worker("blank", BLANK)
            resumed.restore(parent, 8 * 1024 * 1024)
            resumed.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(value, dict(original.model.named_parameters())[name]))
            full = worker("full-from-blank")
            full.restore(parent, 8 * 1024 * 1024, "branch")
            full.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            with self.assertRaisesRegex(ValueError, "groups or rates"):
                worker("full-continue").restore(parent, 8 * 1024 * 1024, "continue")
            trained = worker("trained")
            trained.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            learned = trained.checkpoint(Store(root / "trained"), 4 * 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "not at identity"):
                worker("blank-branch", BLANK).restore(learned, 8 * 1024 * 1024, "branch")
            fixed = worker("fixed-branch", Features("fixed", "fixed"))
            fixed.restore(learned, 8 * 1024 * 1024, "branch")
            kept = [value.detach().clone() for value in feature_parameters(fixed.model, "scale")]
            fixed.run(Monotonic(), Budget(3, 60, 3), lambda: False)
            for value, initial in zip(feature_parameters(fixed.model, "scale"), kept, strict=True):
                self.assertTrue(torch.equal(value, initial))

    def test_blank_export_equals_the_trained_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-features-") as directory:
            root = Path(directory)
            loaded = loader_fixture(root)
            corpus = Corpus(loaded.corpus, 17, 64 * 1024 * 1024, 256)
            model = load_he4(
                loaded.source,
                loaded.vocabulary,
                64 * 1024 * 1024,
                LOADER_GEOMETRY,
                loaded.initialization,
            )
            bound_latents(model, 3.0)
            with torch.no_grad():
                for _, projection in named_projections(model):
                    projection.log_clip.fill_(math.log(0.75))
            target = TensorWorker(
                model,
                corpus,
                "blank-export",
                optimization=Optimization({**default_step_sizes(), "sign": 0.05}, Schedule(), 3.0),
                features=BLANK,
            )
            target.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            exported = export_he4(
                target,
                loaded.source,
                loaded.vocabulary,
                Store(root / "exports"),
                8 * 1024 * 1024,
                32 * 1024 * 1024,
            )
            reader = gguf.GGUFReader(str(exported.artifact.path))
            tensors = {tensor.name: np.array(tensor.data) for tensor in reader.tensors}
            for name, projection in named_projections(model):
                index, part = name.split(".")[1], name.split(".")[2]
                base = f"blk.{index}.ffn_{part}"
                scale = torch.tensor(tensors[base + ".w1a1_scale"].reshape(-1))
                self.assertTrue(torch.equal(scale, projection.scale.detach()))
                self.assertFalse(bool(tensors[base + ".w1ax_midpoint"].any()))
                if part != "up":
                    self.assertTrue(bool((tensors[base + ".w1ax_gain"] == 1).all()))
                clip = reader.fields[base + ".w1ax.clip"].contents()
                self.assertEqual(clip, float(np.float32(math.exp(math.log(0.75)))))
                self.assertEqual(float(projection.log_clip.detach().exp()), clip)


def outlier_model(root: Path) -> tuple[Corpus, HE4]:
    corpus, model, _ = seeded(root)
    with torch.no_grad():
        block = cast(Layer, model.blocks[0])
        block.ffn_norm[:2] = 35.0
        cast(Projection, block.gate).latent_sign[:, :2] *= 0.01
        cast(Projection, block.up).latent_sign[:, :2] *= 0.01
    return corpus, model


def one_update(
    model: HE4, corpus: Corpus, soft: GradualPrecision | None = None
) -> tuple[float, float, float]:
    rates = {key: value / 16 for key, value in {**default_step_sizes(), "sign": 0.004}.items()}
    if soft is None:
        bound_latents(model, 3.0)
    target = TensorWorker(
        model,
        corpus,
        "one-update",
        optimization=Optimization(rates, Schedule(), 3.0),
        precision=soft,
        telemetry=True,
    )
    held = [parameter for group in target.optimizer.param_groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(value) for value in held} == {id(value) for value in trainable}
    assert len(held) == len(trainable)
    capture, windows = corpus.sample(0)

    def loss() -> float:
        target.optimizer.zero_grad(set_to_none=True)
        return float(target.group_loss(capture, windows, 16))

    before = loss()
    target.run(Monotonic(), Budget(1, 60, 1), lambda: False)
    after = loss()
    for parameter in held:
        assert any(parameter is value for value in model.parameters())
    report = target.telemetry_record()
    assert report is not None
    zero = number(record(record(report["inputs"])["blocks.0.gate"])["zero_code_share"])
    return before, after, zero


class TestCalibration(unittest.TestCase):
    def test_calibrated_start_trains_like_a_start_on_massive_activation_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-outliers-") as directory:
            root = Path(directory)
            corpus, plain = outlier_model(root / "plain")
            plain_before, plain_after, plain_zero = one_update(plain, corpus)
            self.assertLess(plain_zero, 0.5)
            _, calibrated = outlier_model(root / "calibrated")
            initial = [p.scale.detach().clone() for _, p in named_projections(calibrated)]
            calibrate(calibrated, corpus, Sample(1, 2, 3), Features(), 4, 64 * 1024 * 1024, False)
            for (name, projection), start in zip(
                named_projections(calibrated), initial, strict=True
            ):
                self.assertTrue(bool((projection.scale <= 16 * start).all()), name)
                self.assertTrue(bool((projection.midpoint.abs() <= 16 * start).all()), name)
            before, after, zero = one_update(calibrated, corpus)
            self.assertLess(zero, 0.5)
            self.assertLess(before, plain_before)
            self.assertLess(after, before)
            self.assertLess(plain_after, plain_before)
            _, soft = outlier_model(root / "soft")
            calibrate(soft, corpus, Sample(1, 2, 3), Features(), 4, 64 * 1024 * 1024, True)
            gate = named_projections(soft)[0][1]
            self.assertLess(float(gate.log_clip.detach().exp()), 8.0)
            checked = start_check(soft, corpus, StartCheck(SAMPLE, 0.01, 0.002), 0.01, 4, 1 << 26)
            self.assertIs(checked["passed"], True)
            soft_before, soft_after, soft_zero = one_update(soft, corpus, precision(2))
            self.assertLess(soft_zero, 0.5)
            self.assertLess(soft_after, soft_before)

    def test_least_squares_rows_match_a_direct_solution(self) -> None:
        generator = torch.Generator().manual_seed(4)
        weight = torch.randn((5, 12), generator=generator)
        projection = Projection(
            weight, 8, nn.Parameter(torch.zeros(12)), nn.Parameter(torch.tensor(0.5))
        )
        inputs = [torch.randn((7, 12), generator=generator) + 0.3 for _ in range(3)]
        report = fit_rows(projection, inputs, weight, True, 1)
        values = (
            torch.cat(
                [projection.quantize(value)[0] * projection.quantize(value)[1] for value in inputs]
            )
            .detach()
            .double()
        )
        signs = torch.where(weight >= 0, 1.0, -1.0).double()
        target = values @ weight.double().T
        for row in range(5):
            design = torch.stack((values @ signs[row], values.sum(-1)), -1)
            solved = torch.linalg.lstsq(design, target[:, row : row + 1]).solution.reshape(-1)
            self.assertGreater(float(solved[0]), 0.0)
            expected = (float(solved[0]), float(solved[1]))
            self.assertAlmostEqual(float(projection.scale.detach()[row]), expected[0], places=4)
            self.assertAlmostEqual(float(projection.midpoint.detach()[row]), expected[1], places=4)
        self.assertLess(number(report["relative_residual"]), 1.0)
        scale_only = Projection(
            weight, 8, nn.Parameter(torch.zeros(12)), nn.Parameter(torch.tensor(0.5))
        )
        fit_rows(scale_only, inputs, weight, False, 1)
        self.assertFalse(bool(scale_only.midpoint.any()))
        product = values @ signs.T
        direct = ((product * target).sum(0) / product.square().sum(0)).clamp_min(0)
        self.assertTrue(torch.allclose(scale_only.scale.double(), direct, rtol=1e-5))

    def test_bound_minimizes_table_output_error_not_input_error(self) -> None:
        generator = torch.Generator().manual_seed(9)
        weight = torch.randn((6, 16), generator=generator) * 0.1
        weight[:, :2] *= 0.01
        projection = Projection(
            weight, 4, nn.Parameter(torch.zeros(16)), nn.Parameter(torch.tensor(0.0))
        )
        inputs = [torch.randn((20, 16), generator=generator) for _ in range(2)]
        for value in inputs:
            value[:, :2] = 35.0 * torch.sign(value[:, :2])
        chosen = choose_bound(projection, inputs, weight)

        def error(bound: float, tables: torch.Tensor) -> float:
            value = torch.tensor(bound, dtype=torch.float32)
            return sum(float(rounding_error(item, value, 7, tables)) for item in inputs)

        identity = torch.eye(16)
        for candidate in (1.0, 4.0, 35.0):
            self.assertLessEqual(error(chosen, weight), error(candidate, weight) * (1 + 1e-6))
        self.assertLess(chosen, 8.0)
        self.assertLess(error(35.0, identity), error(chosen, identity))
        with torch.no_grad():
            projection.log_clip.fill_(math.log(chosen))
        codes, _ = projection.quantize(torch.cat(inputs))
        self.assertLess(float((codes == 0).double().mean()), 0.5)

    def test_row_fit_keeps_degenerate_rows_inside_the_trust_limit(self) -> None:
        generator = torch.Generator().manual_seed(6)
        weight = torch.randn((4, 8), generator=generator) * 0.1
        projection = Projection(
            weight, 8, nn.Parameter(torch.zeros(8)), nn.Parameter(torch.tensor(2.0))
        )
        initial = projection.scale.detach().clone()
        inputs = [torch.zeros((5, 8))]
        inputs[0][:, :2] = 0.06
        source = weight.clone()
        source[:, 0] = 5.0
        report = fit_rows(projection, inputs, source, True, 1)
        scale = projection.scale.detach()
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertTrue(bool((scale <= 16 * initial).all()))
        self.assertTrue(bool((projection.midpoint.detach().abs() <= 16 * initial).all()))
        self.assertGreater(integer(report["kept_initial_scale_rows"]), 0)

    def test_calibration_is_deterministic_and_respects_switches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-calibration-") as directory:
            root = Path(directory)
            corpus, first, _ = seeded(root)
            _, second, _ = seeded(root / "again")
            sample = Sample(1, 2, 5)
            one = calibrate(first, corpus, sample, Features(), 4, 64 * 1024 * 1024, False)
            two = calibrate(second, corpus, sample, Features(), 4, 64 * 1024 * 1024, False)
            self.assertEqual(canonical(one), canonical(two))
            for (name, value), other in zip(
                first.named_parameters(), second.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(value, other), name)
            self.assertEqual(record(one["sample"])["windows"], 2)
            self.assertEqual(record(one["sample"])["role"], "train")
            gate = named_projections(first)[0][1]
            self.assertNotEqual(float(gate.log_clip.detach()), 0.0)
            self.assertTrue(bool(gate.midpoint.any()))
            for name, projection in named_projections(first):
                self.assertTrue(
                    torch.equal(
                        projection.latent_sign,
                        dict(second.named_parameters())[name + ".latent_sign"],
                    )
                )
            _, soft, _ = seeded(root / "soft")
            initial = snapshot(soft)
            calibrate(soft, corpus, sample, Features(), 4, 64 * 1024 * 1024, True)
            for name, value in soft.named_parameters():
                if not name.endswith("log_clip"):
                    self.assertTrue(torch.equal(value, initial[name]), name)
            self.assertTrue(
                torch.equal(
                    named_projections(soft)[0][1].log_clip, named_projections(first)[0][1].log_clip
                )
            )
            _, off, _ = seeded(root / "off")
            calibrate(off, corpus, sample, BLANK, 4, 64 * 1024 * 1024, False)
            self.assertFalse(any(bool(p.midpoint.any()) for _, p in named_projections(off)))
            self.assertFalse(any(bool(p.log_gain.any()) for _, p in named_projections(off)))
            changed = calibrate(
                seeded(root / "changed")[1],
                corpus,
                Sample(1, 2, 6),
                Features(),
                4,
                64 * 1024 * 1024,
                False,
            )
            self.assertNotEqual(
                record(changed["sample"])["sha256"], record(one["sample"])["sha256"]
            )


class TestGradualPrecision(unittest.TestCase):
    def test_soft_weight_limits_and_exact_gradient(self) -> None:
        generator = torch.Generator().manual_seed(2)
        latent = torch.randn((6, 9), generator=generator) * 0.1
        scale = latent.abs().mean(-1)
        tiny = soft_weight(latent, scale, 1e-4)
        self.assertTrue(torch.allclose(tiny, latent, rtol=1e-6, atol=1e-9))
        hard = soft_weight(latent, scale, 1e4)
        self.assertTrue(torch.equal(hard, scale[:, None] * torch.where(latent >= 0, 1.0, -1.0)))
        zero = soft_weight(latent, torch.zeros(6), 0.5)
        self.assertFalse(bool(zero.any()))
        double_latent = latent.double().requires_grad_()
        double_scale = (scale.double() * 1.3).requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda w, s: soft_weight(w, s, 1.7), (double_latent, double_scale)
            )
        )
        reference = (
            double_scale[:, None]
            * torch.tanh(1.7 * double_latent / double_scale[:, None])
            / math.tanh(1.7)
        )
        upstream = torch.randn((6, 9), dtype=torch.float64, generator=generator)
        expected = torch.autograd.grad((reference * upstream).sum(), (double_latent, double_scale))
        actual = torch.autograd.grad(
            (soft_weight(double_latent, double_scale, 1.7) * upstream).sum(),
            (double_latent, double_scale),
        )
        for left, right in zip(actual, expected, strict=True):
            self.assertTrue(torch.allclose(left, right))

    def test_soft_projection_reproduces_public_table_and_trains_through_tanh(self) -> None:
        generator = torch.Generator().manual_seed(3)
        weight = torch.randn((5, 8), generator=generator) * 0.1
        projection = Projection(
            weight, 4, nn.Parameter(torch.zeros(8)), nn.Parameter(torch.tensor(0.0))
        )
        value = torch.randn((3, 8), generator=generator)
        codes, step = projection.quantize(value)
        public = (codes * step) @ weight.T
        projection.temperature = 1e-3
        soft = projection.forward(value)
        self.assertTrue(torch.allclose(soft, public, rtol=1e-5, atol=1e-7))
        soft.square().sum().backward()
        assert projection.latent_sign.grad is not None and projection.scale.grad is not None
        self.assertTrue(bool(projection.latent_sign.grad.any()))
        projection.temperature = None
        hard = Projection(weight, 4, nn.Parameter(torch.zeros(8)), nn.Parameter(torch.tensor(0.0)))
        hard.forward(value).sum().backward()
        gradient = hard.latent_sign.grad
        assert gradient is not None
        self.assertTrue(bool(((gradient != 0) <= (weight.abs() <= 1)).all()))

    def test_temperature_schedule(self) -> None:
        exponential = precision(5)
        values = [exponential.temperature(u) for u in range(7)]
        self.assertEqual(values[5:], [None, None])
        self.assertAlmostEqual(float(values[0] or 0), 0.01)
        self.assertAlmostEqual(float(values[4] or 0), 8.0)
        self.assertAlmostEqual(float(values[2] or 0), math.sqrt(0.08))
        linear = GradualPrecision(0.5, 2.5, "linear", 3, 1e-3, StartCheck(SAMPLE, 1.0, 1.0))
        self.assertEqual([linear.temperature(u) for u in range(4)], [0.5, 1.5, 2.5, None])
        for invalid in (
            lambda: GradualPrecision(0.0, 1.0, "linear", 2, 1e-3, StartCheck(SAMPLE, 1.0, 1.0)),
            lambda: GradualPrecision(2.0, 1.0, "linear", 2, 1e-3, StartCheck(SAMPLE, 1.0, 1.0)),
            lambda: GradualPrecision(0.1, 1.0, "linear", 0, 1e-3, StartCheck(SAMPLE, 1.0, 1.0)),
            lambda: StartCheck(SAMPLE, 0.0, 1.0),
            lambda: Sample(0, 1, 1),
        ):
            with self.assertRaises(ValueError):
                invalid()

    def test_switch_preserves_signs_and_resumes_exactly_across_it(self) -> None:
        soft = precision(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = worker("gradual", soft=soft)
            source = [value.detach().clone() for value in latent_parameters(original.model)]
            parents = []
            losses = []
            for updates in range(1, 5):
                original.run(Monotonic(), Budget(updates, 60, updates), lambda: False)
                losses.append(original.progress.last_loss)
                parents.append(original.checkpoint(Store(root / str(updates)), 4 * 1024 * 1024))
                if updates <= 2:
                    self.assertEqual(original.phase, "soft")
                    sign_rate = next(
                        g["lr"] for g in original.optimizer.param_groups if g["name"] == "sign"
                    )
                    self.assertEqual(sign_rate, 2e-3)
                    self.assertFalse(
                        all(
                            torch.equal(a, b)
                            for a, b in zip(source, latent_parameters(original.model), strict=True)
                        )
                    )
                if updates == 3:
                    self.assertEqual((original.phase, original.latent_step_offset), ("hard", 2))
                    for value in latent_parameters(original.model):
                        self.assertLessEqual(float(value.detach().abs().max()), 1.0)
            converted = torch.load(parents[2].artifact.path, weights_only=True)
            self.assertEqual(converted["precision"], {"phase": "hard", "latent_step_offset": 2})
            soft_state = torch.load(parents[1].artifact.path, weights_only=True)
            self.assertEqual(soft_state["precision"], {"phase": "soft", "latent_step_offset": 0})
            expected = snapshot(original.model)
            for start in (0, 1, 2):
                resumed = worker("gradual", soft=soft)
                resumed.restore(parents[start], 8 * 1024 * 1024)
                resumed.run(Monotonic(), Budget(4, 60, 4), lambda: False)
                self.assertEqual(resumed.progress.last_loss, losses[-1], start)
                for name, value in resumed.model.named_parameters():
                    self.assertTrue(torch.equal(value, expected[name]), (start, name))
            branched = worker("gradual-branch", soft=soft)
            branched.restore(parents[0], 8 * 1024 * 1024, "branch")
            branched.run(Monotonic(), Budget(4, 60, 4), lambda: False)
            self.assertEqual((branched.phase, branched.latent_step_offset), ("hard", 1))
            child = branched.checkpoint(Store(root / "branch"), 4 * 1024 * 1024)
            again = worker("gradual-branch", soft=soft)
            again.restore(child, 8 * 1024 * 1024)
            self.assertEqual(again.latent_step_offset, 1)
            with self.assertRaisesRegex(ValueError, "soft-phase state"):
                worker("plain").restore(parents[0], 8 * 1024 * 1024, "branch")
            plain = worker("plain")
            plain.restore(parents[2], 8 * 1024 * 1024, "continue")
            plain.run(Monotonic(), Budget(5, 60, 5), lambda: False)
            with self.assertRaisesRegex(ValueError, "phase disagrees"):
                worker("plain-parent", soft=precision(9)).restore(
                    parents[2], 8 * 1024 * 1024, "branch"
                )

    def test_signs_survive_the_conversion(self) -> None:
        soft = precision(1)
        target = worker("convert", soft=soft)
        target.run(Monotonic(), Budget(1, 60, 1), lambda: False)
        before = [value.detach().clone() for value in latent_parameters(target.model)]
        target.harden()
        for value, previous in zip(latent_parameters(target.model), before, strict=True):
            self.assertTrue(torch.equal(value.detach() >= 0, previous >= 0))
            mean = previous.abs().mean(-1, keepdim=True)
            self.assertTrue(torch.equal(value.detach(), (previous / (3 * mean)).clamp(-1, 1)))
            state = target.optimizer.state[value]
            self.assertTrue(all(not bool(item.any()) for item in state.values()))

    def test_start_check_passes_and_fails_as_declared(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-start-") as directory:
            root = Path(directory)
            corpus, model, _ = seeded(root)
            check = StartCheck(SAMPLE, 1e-3, 1e-4)
            passed = start_check(model, corpus, check, 1e-3, 4, 64 * 1024 * 1024)
            self.assertIs(passed["passed"], True)
            self.assertEqual(len(record(passed["tables"])), 3)
            self.assertTrue(all(p.temperature is None for _, p in named_projections(model)))
            failed = start_check(model, corpus, check, 3.0, 4, 64 * 1024 * 1024)
            self.assertIs(failed["passed"], False)
            self.assertGreater(number(failed["max_relative_error"]), 1e-3)


class TestTelemetry(unittest.TestCase):
    def test_telemetry_keeps_parameters_and_reports_each_family(self) -> None:
        for soft in (None, precision(2)):
            plain = worker("telemetry", soft=soft)
            observed = worker("telemetry", soft=soft, telemetry=True)
            for target in (plain, observed):
                target.run(Monotonic(), Budget(3, 60, 3), lambda: False)
            self.assertEqual(plain.progress.last_loss, observed.progress.last_loss)
            for name, value in observed.model.named_parameters():
                self.assertTrue(torch.equal(value, dict(plain.model.named_parameters())[name]))
            self.assertTrue(all(p.statistics is None for _, p in named_projections(observed.model)))
            report = observed.telemetry_record()
            assert report is not None
            norms = record(report["gradient_norms"])
            combined = math.sqrt(
                sum(
                    number(record(value)["last"]) ** 2
                    for family, value in norms.items()
                    if family != "soft"
                )
            )
            self.assertAlmostEqual(combined, observed.last_gradient_norm, places=4)
            self.assertTrue({"scale", "midpoint", "gain", "clip", "norm", "sign"} <= set(norms))
            if soft is not None:
                self.assertIn("soft", norms)
                self.assertEqual(record(report["precision"])["phase"], "hard")
            self.assertEqual(report["updates"], 3)
            clipping = record(report["clipping"])
            self.assertLessEqual(number(clipping["mean_factor"]), 1.0)
            signs = record(report["signs"])
            self.assertGreaterEqual(integer(signs["flipped"]), 0)
            self.assertEqual(len(record(signs["by_projection"])), 18)
            inputs = record(record(report["inputs"])["blocks.0.gate"])
            self.assertGreater(integer(inputs["inputs"]), 0)
            self.assertLessEqual(number(inputs["zero_code_share"]), 1.0)
            self.assertIsNone(report["memory"])
            follow = observed.telemetry_record()
            assert follow is not None
            self.assertEqual((follow["since_updates"], follow["updates"]), (3, 0))
            self.assertEqual(record(follow["signs"])["flipped"], 0)
        self.assertIsNone(worker("off").telemetry_record())


class TestIntentSettings(unittest.TestCase):
    def test_omitted_and_default_settings_keep_the_recipe_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-intent-") as directory:
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

            default_features: Record = {
                "features": {
                    "scale": "learned",
                    "clip": "learned",
                    "gain": "learned",
                    "midpoint": "learned",
                }
            }
            unchanged: list[Record] = [
                {},
                default_features,
                {"telemetry": True},
                {"telemetry": False, "initial_export": True},
            ]
            for changes in unchanged:
                self.assertEqual(parse(changes).recipe, OMITTED_RECIPE, changes)
            omitted = parse({})
            self.assertTrue(omitted.features.default)
            self.assertIsNone(omitted.calibration)
            self.assertIsNone(omitted.gradual_precision)
            self.assertFalse(omitted.telemetry or omitted.initial_export)
            blank = parse({"features": BLANK.encode()})
            self.assertEqual(blank.features, BLANK)
            calibrated = parse({"calibration": {"sample": SAMPLE.encode()}})
            self.assertEqual(calibrated.calibration, SAMPLE)
            budget = {"updates": 3, "wall_seconds": 60, "attempts": 4, "retries": 0}
            gradual = parse({"gradual_precision": gradual_record(2), "budget": budget})
            self.assertEqual(gradual.gradual_precision, precision(2))
            integral = parse(
                {
                    "gradual_precision": {**gradual_record(2), "end_temperature": 8},
                    "budget": budget,
                }
            )
            self.assertEqual(integral.recipe, gradual.recipe)
            recipes = {
                OMITTED_RECIPE,
                blank.recipe,
                calibrated.recipe,
                gradual.recipe,
                parse({"features": Features(scale="fixed").encode()}).recipe,
                parse({"calibration": {"sample": Sample(1, 2, 4).encode()}}).recipe,
                parse({"gradual_precision": gradual_record(1), "budget": budget}).recipe,
            }
            self.assertEqual(len(recipes), 7)
            kept = parse(
                {
                    "optimization": {
                        **row_optimization(),
                        "step_sizes": {**default_step_sizes(), "sign": 0.004},
                        "schedule": {
                            "decay": "cosine",
                            "warmup_updates": 1,
                            "start_update": 0,
                            "final_update": 50,
                        },
                    },
                    "blocks": {"slots": {"4": 1.0}, "position_weights": "decay"},
                    "grouping": {
                        "max_windows": 32,
                        "activation_bytes": 3 * 1024**3,
                        "device_bytes": 14 * 1024**3,
                    },
                    "trained_tables": False,
                    "frozen_format": "baseline-k",
                    "frozen_baseline": fixed["source"],
                }
            )
            self.assertEqual(kept.optimization.step_sizes["sign"], 0.004)
            self.assertEqual(kept.optimization.latent_multiple, 3.0)
            self.assertEqual(kept.blocks.fixed, 4)
            self.assertEqual(kept.grouping.max_windows, 32)
            refusals: list[tuple[Record, str]] = [
                ({"features": {"scale": "fixed"}}, "features require"),
                ({"features": {**BLANK.encode(), "gain": "fixed"}}, "learned or off"),
                ({"features": {**BLANK.encode(), "scale": "off"}}, "learned or fixed"),
                ({"telemetry": 1}, "true or false"),
                ({"calibration": {"sample": {"sequences": 1, "windows": 2}}}, "sample requires"),
                ({"calibration": {"sample": {**SAMPLE.encode(), "windows": 0}}}, "windows"),
                ({"gradual_precision": gradual_record(2)}, "hard-sign updates"),
                (
                    {
                        "gradual_precision": {**gradual_record(2), "shape": "cosine"},
                        "budget": budget,
                    },
                    "linear or exponential",
                ),
            ]
            for refused, message in refusals:
                with self.assertRaisesRegex(ValueError, message):
                    parse(refused)
            parent = {"path": str(path), "sha256": "9" * 64, "bytes": 1}
            modes: list[tuple[str, Record, str]] = [
                ("branch", {"calibration": {"sample": SAMPLE.encode()}}, "calibration"),
                (
                    "continue",
                    {"gradual_precision": gradual_record(2), "budget": budget},
                    "fresh, resume",
                ),
                ("resume", {"initial_export": True}, "initial_export"),
            ]
            for mode, changes, message in modes:
                with self.assertRaisesRegex(ValueError, message):
                    parse({**changes, "mode": mode, "parent": parent})


class TestEntry(unittest.TestCase):
    def test_calibrated_gradual_run_exports_update_zero_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-entry-") as directory:
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
                "gradual_precision": gradual_record(1, maximum=0.5),
                "telemetry": True,
                "initial_export": True,
                "features": Features(clip="fixed").encode(),
            }
            path = root / "intent.json"
            path.write_bytes(canonical(intent))
            measured = start_check_command(path)
            self.assertIs(measured["passed"], True)
            self.assertEqual(len(record(measured["tables"])), 3)
            self.assertIn("calibration_values_sha256", measured)
            first = execute_leased(path, root / "first")
            events = list(Journal(root / "first").events())
            kinds = [row.get("kind") for row in events]
            for kind in ("calibration", "gradual-precision-start-check", "initial-export"):
                self.assertIn(kind, kinds)
            calibrated = next(row for row in events if row.get("kind") == "calibration")
            self.assertEqual(calibrated["values_sha256"], measured["calibration_values_sha256"])
            check = next(
                row for row in events if row.get("kind") == "gradual-precision-start-check"
            )
            self.assertIs(check["passed"], True)
            self.assertEqual(check["max_relative_error"], measured["max_relative_error"])
            initial = next(row for row in events if row.get("kind") == "initial-export")
            self.assertEqual(initial["updates"], 0)
            self.assertEqual(record(initial["precision"])["phase"], "soft")
            exports = Store(root / "first" / "exports").latest(first.recipe)
            states = Store(root / "first" / "states").latest(first.recipe)
            assert exports is not None and states is not None
            self.assertEqual(states.updates, 2)
            saved = torch.load(states.artifact.path, weights_only=True)
            self.assertEqual(saved["precision"]["phase"], "hard")
            intent.update(
                mode="resume",
                parent=reference(states.artifact),
                initial_export=False,
                budget={"updates": 3, "wall_seconds": 60, "attempts": 4, "retries": 0},
            )
            path.write_bytes(canonical(intent))
            resumed = execute_leased(path, root / "second")
            self.assertEqual(resumed.recipe, first.recipe)
            intent.update(
                mode="fresh",
                parent=None,
                budget={"updates": 2, "wall_seconds": 60, "attempts": 3, "retries": 0},
                gradual_precision={
                    **gradual_record(1),
                    "start_temperature": 3.0,
                    "start_check": {
                        "sample": SAMPLE.encode(),
                        "max_relative_error": 1e-6,
                        "mean_relative_error": 1e-6,
                    },
                },
            )
            path.write_bytes(canonical(intent))
            with self.assertRaisesRegex(ValueError, "start check exceeds"):
                execute_leased(path, root / "refused")
            refused = [
                row
                for row in Journal(root / "refused").events()
                if row.get("kind") == "gradual-precision-start-check"
            ]
            self.assertEqual([row["passed"] for row in refused], [False])

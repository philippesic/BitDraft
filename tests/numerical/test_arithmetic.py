import math
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest import mock

import torch

from tests.numerical.test_features import snapshot, worker
from tests.numerical.test_model import OMITTED_RECIPE, loader_fixture
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.intent import Intent
from w1a1_eagle.model import (
    BYPASS_CHANNELS,
    DeferredChecks,
    Layer,
    Projection,
    cache_storage,
    hard_sign,
    rope,
    tensor_cores,
)
from w1a1_eagle.numerical.worker import NONFINITE_NORM, TensorWorker
from w1a1_eagle.records import canonical
from w1a1_eagle.training import Arithmetic, Budget

FAMILIES = ("sign", "scale", "midpoint", "gain", "clip", "norm")
MODES: tuple[Arithmetic, ...] = ("reference", "fast-exact", "fast-exact-tf32ff")


def projection(rows: int, inputs: int, bits: int, seed: int) -> Projection:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn((rows, inputs), generator=generator) * 0.1
    gain = torch.nn.Parameter(torch.randn(inputs, generator=generator) * 0.1)
    clip = torch.nn.Parameter(torch.tensor(0.3))
    result = Projection(weight, bits, gain, clip)
    with torch.no_grad():
        result.midpoint.copy_(torch.randn(rows, generator=generator) * 0.01)
    return result


def run_projection(
    module: Projection, inputs: torch.Tensor, upstream: torch.Tensor, single: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    module.zero_grad(set_to_none=True)
    value = inputs.clone().requires_grad_()
    module.single_product = single
    try:
        output = module.forward(value)
    finally:
        module.single_product = False
    (output * upstream).sum().backward()
    gradients = {"input": value.grad}
    for name, parameter in module.named_parameters():
        gradients[name] = parameter.grad
    return output.detach(), {name: grad for name, grad in gradients.items() if grad is not None}


def relative(first: torch.Tensor, second: torch.Tensor) -> float:
    scale = float(first.abs().max())
    return float((first - second).abs().max()) / (scale if scale > 0 else 1.0)


def tf32(value: torch.Tensor) -> torch.Tensor:
    bits = value.view(torch.int32)
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & ~0x1FFF
    return rounded.view(torch.float32)


def family_gradients(target: TensorWorker) -> dict[str, list[torch.Tensor]]:
    result: dict[str, list[torch.Tensor]] = {}
    for group in target.optimizer.param_groups:
        result[str(group["name"])] = [
            parameter.grad.detach().clone()
            for parameter in group["params"]
            if parameter.grad is not None
        ]
    return result


def failing(target: TensorWorker, damage: Callable[[TensorWorker], None]) -> BaseException:
    damage(target)
    before = snapshot(target.model)
    try:
        target.step(0)
    except (ValueError, RuntimeError) as error:
        caught: BaseException = error
    else:
        raise AssertionError("the update did not fail")
    for name, value in snapshot(target.model).items():
        if not torch.equal(value.view(torch.int32), before[name].view(torch.int32)):
            raise AssertionError("parameters changed before the error: " + name)
    if target.optimizer.state or target.optimizer_updates or target.updates:
        raise AssertionError("the optimizer stepped before the error")
    return caught


class TestGateUpBypass(unittest.TestCase):
    def test_exact_columns_bypass_quantization_and_train_the_remaining_weights(self) -> None:
        module = projection(8, 864, 4, 8101)
        unchanged = projection(8, 864, 4, 8101)
        module.enable_bypass()
        values = torch.randn((4, 864), generator=torch.Generator().manual_seed(8102))
        values[:, list(BYPASS_CHANNELS)] *= 35
        masked = values.clone()
        masked[:, list(BYPASS_CHANNELS)] = 0
        exact_weights = unchanged.latent_sign.detach()[:, list(BYPASS_CHANNELS)]
        side = torch.nn.functional.linear(values[:, list(BYPASS_CHANNELS)], exact_weights)
        expected = unchanged(masked) + side
        actual = module(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        gradient = module.latent_sign.grad
        assert gradient is not None
        self.assertEqual(int(torch.count_nonzero(gradient[:, list(BYPASS_CHANNELS)])), 0)
        self.assertGreater(int(torch.count_nonzero(gradient[:, 1:850])), 0)


class TestSingleProduct(unittest.TestCase):
    def test_forward_is_bit_identical_and_gradients_agree(self) -> None:
        for bits, rows, inputs in ((4, 24, 16), (8, 16, 24)):
            module = projection(rows, inputs, bits, bits)
            generator = torch.Generator().manual_seed(bits + 1)
            value = torch.randn((9, inputs), generator=generator)
            upstream = torch.randn((9, rows), generator=generator)
            paired, paired_gradients = run_projection(module, value, upstream, False)
            single, single_gradients = run_projection(module, value, upstream, True)
            self.assertTrue(torch.equal(paired, single))
            self.assertEqual(set(paired_gradients), set(single_gradients))
            self.assertEqual(
                set(paired_gradients),
                {"input", "latent_sign", "scale", "midpoint", "log_gain", "log_clip"},
            )
            for name, gradient in paired_gradients.items():
                self.assertGreater(float(gradient.abs().max()), 0.0, name)
                self.assertLess(relative(gradient, single_gradients[name]), 1e-5, name)

    def test_soft_path_is_unchanged(self) -> None:
        module = projection(12, 16, 4, 3)
        module.temperature = 2.0
        generator = torch.Generator().manual_seed(4)
        value = torch.randn((5, 16), generator=generator)
        upstream = torch.randn((5, 12), generator=generator)
        paired, paired_gradients = run_projection(module, value, upstream, False)
        module.tensor_cores = True
        single, single_gradients = run_projection(module, value, upstream, True)
        self.assertTrue(torch.equal(paired, single))
        for name, gradient in paired_gradients.items():
            self.assertTrue(torch.equal(gradient, single_gradients[name]), name)

    def test_whole_update_matches_reference(self) -> None:
        reference = worker("reference")
        losses: dict[str, float] = {"reference": reference.step(0)}
        expected = family_gradients(reference)
        parameters = snapshot(reference.model)
        modes: tuple[Arithmetic, ...] = ("fast-exact", "fast-exact-tf32ff")
        for mode in modes:
            fast = worker(mode, arithmetic=mode)
            losses[mode] = fast.step(0)
            self.assertEqual(losses[mode], losses["reference"], mode)
            self.assertEqual(fast.last_gradient_norm, reference.last_gradient_norm, mode)
            measured = family_gradients(fast)
            self.assertEqual(set(measured), set(FAMILIES))
            for family in FAMILIES:
                first = torch.cat([value.reshape(-1) for value in expected[family]])
                second = torch.cat([value.reshape(-1) for value in measured[family]])
                difference = float((first - second).norm() / first.norm())
                self.assertLess(difference, 1e-5, (mode, family))
            for name, value in snapshot(fast.model).items():
                self.assertTrue(
                    torch.allclose(value, parameters[name], rtol=1e-5, atol=1e-7), (mode, name)
                )
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            for module in fast.model.modules():
                if isinstance(module, Projection):
                    self.assertIsNone(module.checks)
                    self.assertFalse(module.single_product or module.tensor_cores)
                if isinstance(module, Layer):
                    self.assertIsNone(module.checks)

    def test_one_device_read_per_update(self) -> None:
        reads: list[int] = []
        original = DeferredChecks.read

        def counted(self: DeferredChecks, *values: torch.Tensor) -> tuple[float, ...]:
            reads.append(len(self.failed))
            return original(self, *values)

        fast = worker("fast", arithmetic="fast-exact")
        with mock.patch.object(DeferredChecks, "read", counted):
            fast.step(0)
            fast.step(1)
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads, [18 * 2 + 6 * 4 + 1] * 2)

    def test_resume_is_exact_in_the_new_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = worker("fast", arithmetic="fast-exact-tf32ff")
            original.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            parent = original.checkpoint(Store(Path(directory)), 4 * 1024 * 1024)
            original.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            resumed = worker("fast", arithmetic="fast-exact-tf32ff")
            resumed.restore(parent, 8 * 1024 * 1024)
            resumed.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            self.assertEqual(resumed.progress.last_loss, original.progress.last_loss)
            expected = snapshot(original.model)
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]), name)


class TestDeferredChecks(unittest.TestCase):
    def assert_same_failure(
        self, damage: Callable[[TensorWorker], None], kind: type[BaseException], message: str
    ) -> None:
        errors = {mode: failing(worker(mode, arithmetic=mode), damage) for mode in MODES}
        for mode, error in errors.items():
            self.assertIsInstance(error, kind, mode)
            if kind is ValueError:
                self.assertEqual(str(error), message, mode)
        self.assertEqual(str(errors["fast-exact"]), str(errors["fast-exact-tf32ff"]))

    def test_nonfinite_quantizer_input(self) -> None:
        def damage(target: TensorWorker) -> None:
            with torch.no_grad():
                block = target.model.blocks[0]
                assert isinstance(block, Layer)
                block.ffn_norm.fill_(math.inf)

        self.assert_same_failure(damage, ValueError, "nonfinite quantizer input")

    def test_invalid_bound(self) -> None:
        for value in (-math.inf, math.inf, math.nan):

            def damage(target: TensorWorker, value: float = value) -> None:
                with torch.no_grad():
                    block = target.model.blocks[1]
                    assert isinstance(block, Layer)
                    block.gate.log_clip.fill_(value)

            self.assert_same_failure(damage, ValueError, "invalid learned clipping bound")

    def test_f16_cache_overflow(self) -> None:
        def damage(target: TensorWorker) -> None:
            with torch.no_grad():
                block = target.model.blocks[2]
                assert isinstance(block, Layer)
                block.attn_norm.fill_(1e10)

        self.assert_same_failure(damage, ValueError, "F16 cache storage overflow")

    def test_nonfinite_loss(self) -> None:
        def damage(target: TensorWorker) -> None:
            with torch.no_grad():
                target.model.output_norm.fill_(math.inf)

        self.assert_same_failure(damage, ValueError, "nonfinite loss")

    def test_nonfinite_gradient_norm(self) -> None:
        def damage(target: TensorWorker) -> None:
            block = target.model.blocks[0]
            assert isinstance(block, Layer)
            block.down.scale.register_hook(lambda gradient: torch.full_like(gradient, math.inf))

        self.assert_same_failure(damage, RuntimeError, "")
        error = failing(worker("fast", arithmetic="fast-exact"), damage)
        self.assertEqual(str(error), NONFINITE_NORM)

    def test_checks_raise_the_first_failure_in_order(self) -> None:
        checks = DeferredChecks()
        checks.require(torch.tensor(True), "first")
        checks.require(torch.tensor(False), "second")
        checks.require(torch.tensor(False), "third")
        with self.assertRaisesRegex(ValueError, "^second$"):
            checks.read(torch.tensor(1.0))
        passing = DeferredChecks()
        passing.require(torch.tensor(True), "never")
        self.assertEqual(passing.read(torch.tensor(1.5), torch.tensor(2.0)), (1.5, 2.0))
        deferred = DeferredChecks()
        cache_storage(torch.tensor([70000.0]), deferred)
        with self.assertRaisesRegex(ValueError, "F16 cache storage overflow"):
            deferred.read()


class TestExactKernels(unittest.TestCase):
    def test_tensor_core_scope_restores_the_global_flag(self) -> None:
        matmul = torch.backends.cuda.matmul
        self.assertFalse(matmul.allow_tf32)
        with tensor_cores(True):
            self.assertTrue(matmul.allow_tf32)
        self.assertFalse(matmul.allow_tf32)
        with self.assertRaises(KeyError), tensor_cores(True):
            raise KeyError("inside")
        self.assertFalse(matmul.allow_tf32)
        with tensor_cores(False):
            self.assertFalse(matmul.allow_tf32)

    def test_full_size_integer_product_is_exact_in_tf32(self) -> None:
        generator = torch.Generator().manual_seed(11)
        for bits, inputs, outputs in ((8, 9216, 2560), (4, 2560, 9216)):
            module = projection(outputs, inputs, bits, bits)
            maximum = module.maximum
            with torch.no_grad():
                module.log_clip.fill_(0.0)
                value = torch.randn((48, inputs), generator=generator) * 2
                value[0] = 50.0
                value[1] = -50.0
                codes, _ = module.quantize(value)
                signs = hard_sign(module.latent_sign)
                signs[: outputs // 2] = 1.0
            self.assertTrue(torch.equal(codes, codes.round()))
            self.assertEqual(float(codes.abs().max()), float(maximum))
            self.assertTrue(torch.equal(tf32(codes).view(torch.int32), codes.view(torch.int32)))
            self.assertTrue(torch.equal(tf32(signs).view(torch.int32), signs.view(torch.int32)))
            single = torch.nn.functional.linear(codes, signs)
            exact = torch.nn.functional.linear(codes.double(), signs.double())
            self.assertTrue(torch.equal(single.double(), exact))
            self.assertEqual(float(exact.abs().max()), float(maximum * inputs))
            self.assertLess(maximum * inputs, 2**24)
            reversed_order = torch.nn.functional.linear(codes.flip(-1), signs.flip(-1))
            self.assertTrue(torch.equal(single, reversed_order))

    def test_tf32_emulation_rounds_ordinary_values(self) -> None:
        value = torch.tensor([1.0 + 2**-20, 127.0, -1.0, 0.1])
        rounded = tf32(value)
        self.assertEqual(float(rounded[0]), 1.0)
        self.assertEqual(float(rounded[1]), 127.0)
        self.assertNotEqual(float(rounded[3]), float(value[3]))

    def test_rope_and_sign_surrogate_keep_their_values(self) -> None:
        generator = torch.Generator().manual_seed(5)
        value = torch.randn((7, 3, 8), generator=generator)
        positions = torch.arange(100, 107)
        width = value.shape[-1]
        frequency = torch.tensor(1e7, dtype=torch.float32).pow(
            -torch.arange(0, width, 2, dtype=torch.float32) / width
        )
        angle = positions.float()[:, None] * frequency
        angle = torch.cat((angle, angle), -1)[:, None, :]
        rotated = torch.cat((-value[..., 4:], value[..., :4]), -1)
        expected = value * angle.cos() + rotated * angle.sin()
        self.assertTrue(torch.equal(rope(value, positions, 1e7), expected))
        latent = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]).requires_grad_()
        upstream = torch.tensor([1.0, -0.0, math.nan, math.inf, -math.inf, -2.0, -3.0, math.nan])
        hard_sign(latent).backward(upstream)
        assert latent.grad is not None
        previous = upstream * (latent.detach().abs() <= 1).to(upstream.dtype)
        self.assertTrue(torch.equal(latent.grad.view(torch.int32), previous.view(torch.int32)))


class TestArithmeticSetting(unittest.TestCase):
    def test_omitted_and_reference_keep_the_recipe_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-arithmetic-") as directory:
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

            self.assertEqual(parse({}).arithmetic, "reference")
            self.assertEqual(parse({}).recipe, OMITTED_RECIPE)
            self.assertEqual(parse({"arithmetic": "reference"}).recipe, OMITTED_RECIPE)
            fast = parse({"arithmetic": "fast-exact"})
            cores = parse({"arithmetic": "fast-exact-tf32ff"})
            self.assertEqual(
                (fast.arithmetic, cores.arithmetic), ("fast-exact", "fast-exact-tf32ff")
            )
            self.assertEqual(len({OMITTED_RECIPE, fast.recipe, cores.recipe}), 3)
            for refused in ("tf32", True, None):
                with self.assertRaisesRegex(ValueError, "arithmetic is"):
                    parse({"arithmetic": refused})

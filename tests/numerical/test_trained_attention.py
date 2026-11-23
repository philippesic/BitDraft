import tempfile
import unittest
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch

from tests.numerical.test_model import fixture_corpus, gguf_fixture
from tests.numerical.test_plain import CFG, MEMORY, baseline_file, payloads, quantizer
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Layer, k_rounded
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.k_tables import (
    plain_tables,
    q4_k_codes,
    q6_k_codes,
    quantizer_binary,
    trained_rows,
)
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.training import TABLE_GROUP, Budget, Optimization, default_step_sizes

ATTENTION = ("attn_q", "attn_k", "attn_v", "attn_output")


def attention_model(source: Artifact, vocabulary: Artifact, baseline: Artifact) -> HE4:
    return load_he4(
        source, vocabulary, MEMORY, CFG, None, "baseline-k", baseline, trained_attention=True
    )


def rates() -> Optimization:
    return Optimization({**default_step_sizes(), TABLE_GROUP: 1e-3})


class TestTrainedAttention(unittest.TestCase):
    def test_attention_tables_train_export_as_baseline_types_and_resume(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-attention-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, CFG, seed=5)
            built = quantizer(root)
            baseline = baseline_file(root, "coded.gguf", source, built)
            model = attention_model(source, vocabulary, baseline)
            trained = {name for name, value in model.named_parameters() if name.endswith("operand")}
            self.assertEqual(
                trained,
                {f"blocks.0.{part}.operand" for part in ("query", "key", "value", "output")},
            )
            self.assertFalse(model.fusion.operand.requires_grad)
            reference = {
                item.name: item.tensor_type.name
                for item in gguf.GGUFReader(str(baseline.path)).tensors
            }
            self.assertEqual(
                dict(model.attention_types),
                {f"blk.0.{name}.weight": reference[f"blk.0.{name}.weight"] for name in ATTENTION},
            )
            corpus = Corpus(fixture_corpus(root, 256), 17, MEMORY, 256)
            worker = TensorWorker(model, corpus, "attention", optimization=rates())
            self.assertIn(
                TABLE_GROUP, {str(group["name"]) for group in worker.optimizer.param_groups}
            )
            binary = quantizer_binary(built).path
            store = Store(root / "exports")
            with self.assertRaisesRegex(ValueError, "table quantizer"):
                export_he4(worker, source, vocabulary, store, 8 << 20, 32 << 20)
            initial = export_he4(worker, source, vocabulary, store, 8 << 20, 32 << 20, binary)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            self.assertEqual(
                worker.run(Monotonic(), Budget(1, 60, 1), lambda: False), "update-budget"
            )
            after = dict(model.named_parameters())
            for name in trained:
                self.assertFalse(torch.equal(before[name], after[name]), name)
            exported = export_he4(worker, source, vocabulary, store, 8 << 20, 32 << 20, binary)
            first, second = payloads(initial.artifact.path), payloads(exported.artifact.path)
            self.assertEqual(set(first), set(second))
            served = plain_tables(model, binary)
            for name in ATTENTION:
                tensor = f"blk.0.{name}.weight"
                kind_name, data = second[tensor]
                self.assertEqual(kind_name, reference[tensor])
                self.assertNotEqual(first[tensor][1], data, tensor)
                operand = served[
                    "blocks.0."
                    + {
                        "attn_q": "query",
                        "attn_k": "key",
                        "attn_v": "value",
                        "attn_output": "output",
                    }[name]
                    + ".operand"
                ]
                rows = np.frombuffer(data, dtype=np.uint8).reshape(operand.shape[0], -1)
                restored = gguf.dequantize(rows, getattr(gguf.GGMLQuantizationType, kind_name))
                self.assertTrue(np.array_equal(restored.reshape(operand.shape), operand.cpu()))
            self.assertEqual(first["fc.weight"], second["fc.weight"])
            self.assertEqual(first["fc.weight"][1], payloads(baseline.path)["fc.weight"][1])
            state = worker.checkpoint(Store(root / "states"), MEMORY)
            resumed = TensorWorker(
                attention_model(source, vocabulary, baseline),
                corpus,
                "attention",
                optimization=rates(),
            )
            resumed.restore(state, MEMORY)
            self.assertEqual(resumed.frozen_sha256, worker.frozen_sha256)
            frozen = TensorWorker(
                load_he4(source, vocabulary, MEMORY, CFG, None, "baseline-k", baseline),
                corpus,
                "frozen",
            )
            self.assertNotEqual(frozen.frozen_sha256, worker.frozen_sha256)


class TestTrainedVocabularyRows(unittest.TestCase):
    def test_rows_train_tied_and_export_into_the_output_table(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-rows-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, CFG, seed=5)
            built = quantizer(root)
            baseline = baseline_file(root, "coded.gguf", source, built)
            rows = (2, 7, 9, 13)
            model = load_he4(
                source, vocabulary, MEMORY, CFG, None, "baseline-k", baseline, head_rows=rows
            )
            self.assertTrue(torch.equal(model.head_values, model.vocabulary[list(rows)]))
            corpus = Corpus(fixture_corpus(root, 256), 17, MEMORY, 256)
            worker = TensorWorker(model, corpus, "rows", optimization=rates())
            binary = quantizer_binary(built).path
            store = Store(root / "exports")
            initial = export_he4(worker, source, vocabulary, store, 8 << 20, 32 << 20, binary)
            before = model.head_values.detach().clone()
            self.assertEqual(
                worker.run(Monotonic(), Budget(1, 60, 1), lambda: False), "update-budget"
            )
            self.assertFalse(torch.equal(before, model.head_values))
            hidden = torch.randn(5, CFG.hidden, generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                scores = model.scores(hidden)
                self.assertTrue(torch.allclose(scores[:, list(rows)], hidden @ model.head_values.T))
                self.assertTrue(torch.allclose(scores[:, 4], hidden @ model.vocabulary[4]))
                embedded = model.embed(torch.tensor([7, 4]))
                self.assertTrue(torch.equal(embedded[0], model.head_values[1]))
                self.assertTrue(torch.equal(embedded[1], model.vocabulary[4]))
            exported = export_he4(worker, source, vocabulary, store, 8 << 20, 32 << 20, binary)
            first = payloads(initial.artifact.path)["output.weight"][1]
            second = payloads(exported.artifact.path)["output.weight"][1]
            width = len(first) // model.vocabulary.shape[0]
            coded = trained_rows(model, binary)
            for row in range(model.vocabulary.shape[0]):
                old = first[row * width : (row + 1) * width]
                new = second[row * width : (row + 1) * width]
                if row in rows:
                    self.assertEqual(new, coded[rows.index(row)].tobytes(), row)
                else:
                    self.assertEqual(old, new, row)
            self.assertNotEqual(first, second)
            fields = gguf.GGUFReader(str(exported.artifact.path)).fields
            self.assertEqual(fields["dflash.vocab_trained_rows"].contents(), list(rows))
            state = worker.checkpoint(Store(root / "states"), MEMORY)
            resumed = TensorWorker(
                load_he4(
                    source, vocabulary, MEMORY, CFG, None, "baseline-k", baseline, head_rows=rows
                ),
                corpus,
                "rows",
                optimization=rates(),
            )
            resumed.restore(state, MEMORY)
            self.assertTrue(torch.equal(resumed.model.head_values, model.head_values))


class TestRoundedAttention(unittest.TestCase):
    def test_rounding_keeps_the_grid_and_passes_gradients_straight_through(self) -> None:
        generator = torch.Generator().manual_seed(1)
        for kind in ("Q4_K", "Q6_K"):
            value = (0.02 * torch.randn(4, 512, generator=generator)).requires_grad_()
            rounded = k_rounded(value, kind)
            self.assertFalse(torch.equal(rounded.detach(), value.detach()))
            again = k_rounded(rounded.detach(), kind)
            self.assertLess(
                float((again - rounded.detach()).norm() / rounded.detach().norm()), 1e-3, kind
            )
            weights = torch.randn(4, 512, generator=generator)
            (rounded * weights).sum().backward()
            assert value.grad is not None
            self.assertTrue(torch.equal(value.grad, weights))

    def test_rounded_setting_marks_every_attention_table(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-rounded-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, CFG, seed=5)
            baseline = baseline_file(root, "coded.gguf", source, quantizer(root))
            model = load_he4(
                source,
                vocabulary,
                MEMORY,
                CFG,
                None,
                "baseline-k",
                baseline,
                trained_attention=True,
                attention_rounding=True,
            )
            block = model.blocks[0]
            kinds = {
                part: getattr(block, part).rounding for part in ("query", "key", "value", "output")
            }
            self.assertEqual(set(kinds.values()), {"Q4_K"})
            self.assertIsNone(model.fusion.rounding)


class TestRoundedExport(unittest.TestCase):
    def test_packed_codes_decode_to_the_trained_values(self) -> None:
        generator = torch.Generator().manual_seed(2)
        for kind, pack in (("Q4_K", q4_k_codes), ("Q6_K", q6_k_codes)):
            value = 0.02 * torch.randn(4, 512, generator=generator)
            value[0, :256] = 0
            codes = pack(value)
            self.assertEqual(codes.shape, (4, 2 * (144 if kind == "Q4_K" else 210)))
            decoded = gguf.dequantize(codes, getattr(gguf.GGMLQuantizationType, kind))
            expected = k_rounded(value, kind).detach().numpy()
            self.assertLess(float(np.abs(decoded.reshape(4, 512) - expected).max()), 1e-8, kind)


class TestRoundingRefresh(unittest.TestCase):
    def test_each_update_sees_the_new_tables_rounded(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-refresh-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, CFG, seed=5)
            baseline = baseline_file(root, "coded.gguf", source, quantizer(root))
            model = load_he4(
                source,
                vocabulary,
                MEMORY,
                CFG,
                None,
                "baseline-k",
                baseline,
                trained_attention=True,
                attention_rounding=True,
            )
            corpus = Corpus(fixture_corpus(root, 256), 17, MEMORY, 256)
            worker = TensorWorker(model, corpus, "refresh", optimization=rates())
            table = cast(Layer, model.blocks[0]).query
            probe = torch.randn(3, CFG.hidden, generator=torch.Generator().manual_seed(4))
            for budget in (1, 2):
                self.assertEqual(
                    worker.run(Monotonic(), Budget(budget, 60, budget), lambda: False),
                    "update-budget",
                )
                with torch.no_grad():
                    served = table.forward(probe)
                    fresh = probe @ k_rounded(table.operand.detach(), "Q4_K").T
                self.assertTrue(torch.allclose(served, fresh, atol=1e-6), budget)

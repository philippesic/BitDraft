import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch

from tests.numerical.test_model import fixture_corpus, gguf_fixture
from tests.numerical.test_plain import CFG, MEMORY, baseline_file, payloads, quantizer
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.k_tables import plain_tables, quantizer_binary
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

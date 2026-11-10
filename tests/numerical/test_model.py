import hashlib
import importlib.metadata
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch
from torch import nn

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact, Window
from w1a1_eagle.identity import source_identity
from w1a1_eagle.model import HE4, Geometry, Layer, Projection, hard_sign, rms, rope
from w1a1_eagle.numerical.captures import Capture, CaptureCache, CaptureSpec, load_capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.weights import load_he4, shapes
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import canonical, reference
from w1a1_eagle.supervisor import execute
from w1a1_eagle.training import Budget


class TestNumerical(unittest.TestCase):
    def reference(self, path: Path) -> Artifact:
        return Artifact(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)

    def fixture_corpus(self, root: Path, hidden: int) -> Artifact:
        files = []
        for tap in (2, 6, 10, 14, 18, 22, 26, 30):
            path = root / f"layer-{tap}.f32"
            path.write_bytes(np.full((24, hidden), tap / 32, dtype="<f4").tobytes())
            files.append(reference(self.reference(path)))
        metadata = root / "capture.json"
        metadata.write_bytes(
            canonical(
                {
                    "taps": [2, 6, 10, 14, 18, 22, 26, 30],
                    "feature_semantics": "zero_based_pre_block_input_all_tokens",
                    "input_tokens": [0] * 24,
                    "prompt_tokens": 8,
                    "logit_positions": list(range(8, 24)),
                    "greedy_ids": [1] * 15 + [19],
                    "eog_ids": [19],
                }
            )
        )
        manifest = root / "corpus.json"
        manifest.write_bytes(
            canonical(
                {
                    "schema": 1,
                    "target_sha256": "a" * 64,
                    "protected": ["heldout24"],
                    "expected_train": 1,
                    "records": [
                        {
                            "id": "one",
                            "source": "synthetic-fixture",
                            "group": "one",
                            "topic": None,
                            "aliases": [],
                            "role": "train",
                            "ancestry": "verified",
                            "metadata": reference(self.reference(metadata)),
                            "files": files,
                        }
                    ],
                }
            )
        )
        return self.reference(manifest)

    def test_projection_scalar_oracle(self) -> None:
        policy = tomllib.loads(Path("policy/numerical.toml").read_text())
        for name in ("torch", "gguf", "numpy"):
            self.assertEqual(importlib.metadata.version(name), policy[name])
        layer = Projection(
            torch.tensor([[0.2, -0.4]]),
            4,
            nn.Parameter(torch.zeros(2)),
            nn.Parameter(torch.tensor(0.0)),
        )
        value = layer.forward(torch.tensor([[0.1, 0.7]]))
        self.assertAlmostEqual(float(value.detach()), (1 - 5) * 0.3 / 7, places=6)
        value.sum().backward()
        assert layer.midpoint.grad is not None
        assert layer.latent_sign.grad is not None
        self.assertTrue(torch.allclose(layer.midpoint.grad, torch.tensor([6 / 7])))
        self.assertTrue(torch.allclose(layer.latent_sign.grad, torch.tensor([[0.3 / 7, 1.5 / 7]])))

    def test_sign_gradient_and_rotary_identity(self) -> None:
        source = torch.tensor([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0], requires_grad=True)
        result = hard_sign(source)
        self.assertEqual(result.detach().tolist(), [-1, -1, 1, 1, 1, 1])
        result.sum().backward()
        assert source.grad is not None
        self.assertEqual(source.grad.tolist(), [0, 1, 1, 1, 1, 0])
        value = torch.tensor([[[3.0, 4.0]]])
        self.assertTrue(torch.equal(rope(value, torch.tensor([0]), 1e7), value))
        expected = value / (12.5 + 1e-6) ** 0.5
        self.assertTrue(torch.allclose(rms(value, torch.ones(2), 1e-6), expected))

    def tiny_model(self) -> tuple[HE4, torch.Tensor]:
        torch.set_num_threads(1)
        cfg = Geometry(hidden=4, intermediate=8, heads=2, kv_heads=1, head_dim=2)
        generator = torch.Generator().manual_seed(17)
        tensors = {
            "fc.weight": torch.randn((4, 32), generator=generator) * 0.1,
            "enc.output_norm.weight": torch.ones(4),
            "output_norm.weight": torch.ones(4),
        }
        for index in range(6):
            for name, shape in {
                "attn_norm": (4,),
                "attn_q_norm": (2,),
                "attn_k_norm": (2,),
                "ffn_norm": (4,),
                "attn_q": (4, 4),
                "attn_k": (2, 4),
                "attn_v": (2, 4),
                "attn_output": (4, 4),
                "ffn_gate": (8, 4),
                "ffn_up": (8, 4),
                "ffn_down": (4, 8),
            }.items():
                tensors[f"blk.{index}.{name}.weight"] = (
                    torch.ones(shape)
                    if len(shape) == 1
                    else torch.randn(shape, generator=generator) * 0.1
                )
        model = HE4(tensors, torch.randn((20, 4), generator=generator), 19, cfg)
        features = torch.randn((20, 32), generator=generator)
        return model, features

    def test_tiny_six_layer_forward_and_update(self) -> None:
        model, features = self.tiny_model()
        for module in model.blocks:
            block = cast(Layer, module)
            self.assertIs(block.gate.log_gain, block.up.log_gain)
            self.assertIs(block.gate.log_clip, block.up.log_clip)
        logits = model.forward(features, tuple(range(20)), 2)
        self.assertEqual(tuple(logits.shape), (16, 20))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        first = cast(Layer, model.blocks[0])
        before = first.gate.latent_sign.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0, foreach=False)
        loss = torch.nn.functional.cross_entropy(logits[1:], torch.arange(3, 18))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, first.gate.latent_sign))
        self.assertFalse(torch.cuda.is_initialized())

    def test_gguf_loading_and_preflight(self) -> None:
        cfg = Geometry(hidden=256, intermediate=256, heads=4, kv_heads=1, head_dim=8, layers=1)
        with tempfile.TemporaryDirectory(prefix="binary-loader-") as directory:
            root = Path(directory)
            source = root / "source.gguf"
            writer = gguf.GGUFWriter(str(source), "dflash")
            writer.add_array("dflash.target_layers", [2, 6, 10, 14, 18, 22, 26, 30])
            writer.add_uint32("tokenizer.ggml.mask_token_id", 19)
            for name, shape in shapes(cfg).items():
                data = np.ones(shape, dtype=np.float32)
                encoded = gguf.quantize(data, gguf.GGMLQuantizationType.BF16)
                writer.add_tensor(name, encoded, raw_dtype=gguf.GGMLQuantizationType.BF16)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            vocabulary = root / "vocabulary.gguf"
            writer = gguf.GGUFWriter(str(vocabulary), "qwen35")
            writer.add_tensor(
                "token_embd.weight",
                np.zeros((20, 144), dtype=np.uint8),
                raw_dtype=gguf.GGMLQuantizationType.Q4_K,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            source_ref = Artifact(
                source, hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_size
            )
            vocabulary_ref = Artifact(
                vocabulary,
                hashlib.sha256(vocabulary.read_bytes()).hexdigest(),
                vocabulary.stat().st_size,
            )
            with self.assertRaisesRegex(ValueError, "memory"):
                load_he4(source_ref, vocabulary_ref, 0, cfg)
            model = load_he4(source_ref, vocabulary_ref, 64 * 1024 * 1024, cfg)
            self.assertEqual(model.mask, 19)
            block = cast(Layer, model.blocks[0])
            self.assertTrue(torch.equal(block.gate.latent_sign, torch.ones((256, 256))))
            self.assertTrue(torch.equal(block.gate.scale, torch.ones(256)))
            corpus_ref = self.fixture_corpus(root, 256)
            corpus = Corpus(corpus_ref, 17, 64 * 1024 * 1024, 256)
            self.assertEqual(corpus.count, 2)
            self.assertEqual(corpus.unique_positions, 16)
            expected = [corpus.locate(i)[1] for i in range(5)]
            self.assertEqual(
                expected,
                [Corpus(corpus_ref, 17, 64 * 1024 * 1024, 256).locate(i)[1] for i in range(5)],
            )
            self.assertEqual(corpus.exposure(4)["mean_visits"], 60 / 16)
            intent: Record = {
                "schema": 1,
                "objective": "exercise CPU source integration",
                "hypothesis": "one finite update can be restored",
                "decision": "CPU plumbing qualification only",
                "risk": "synthetic source cannot establish native parity or learning usefulness",
                "authority_scope": "disposable-CPU-test",
                "source": reference(source_ref),
                "vocabulary": reference(vocabulary_ref),
                "corpus": reference(corpus_ref),
                "parent": None,
                "mode": "fresh",
                "budget": {"updates": 1, "wall_seconds": 60, "attempts": 2, "retries": 0},
                "memory_bytes": 64 * 1024 * 1024,
                "state_bytes": 8 * 1024 * 1024,
                "disk_reserve_bytes": 1,
                "backing_available_bytes": 16 * 1024 * 1024,
                "seed": 17,
                "device": "cpu",
                "science_policy": {
                    "id": "synthetic-fixture-only",
                    "min_unique_positions": 16,
                    "max_mean_visits": 2,
                },
                "geometry": {
                    "hidden": 256,
                    "intermediate": 256,
                    "heads": 4,
                    "kv_heads": 1,
                    "head_dim": 8,
                    "layers": 1,
                },
                "code_sha256": source_identity(Path.cwd()),
            }
            intent_path = root / "intent.json"
            intent_path.write_bytes(canonical(intent))
            report = execute(intent_path, root / "attempt", "disposable-CPU-test")
            self.assertEqual(record(report["latest"])["outcome"], "complete", report)
            self.assertEqual(record(report["worker"])["updates"], 1)
            self.assertIn("unknown", str(report["scientific"]))
            latest = Store(root / "attempt/states").latest(
                str(record(report["admission"])["recipe"])
            )
            self.assertEqual(latest.updates, 1)

    def test_capture_tail_budget_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-capture-") as directory:
            root = Path(directory)

            def reference(path: Path) -> Artifact:
                return Artifact(
                    path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
                )

            files = []
            for tap in (2, 6, 10, 14, 18, 22, 26, 30):
                path = root / f"layer-{tap}.f32"
                path.write_bytes(np.full((24, 4), tap, dtype="<f4").tobytes())
                files.append(reference(path))
            for name, values in (
                ("inputs.npy", list(range(24))),
                ("label_positions.npy", list(range(8, 24))),
                ("labels.npy", list(range(100, 115)) + [999]),
            ):
                path = root / name
                np.save(path, np.asarray(values, dtype="int32"), allow_pickle=False)
                files.append(reference(path))
            metadata = {
                "taps": [2, 6, 10, 14, 18, 22, 26, 30],
                "feature_semantics": "zero_based_pre_block_input_all_tokens",
                "input_tokens": list(range(24)),
                "prompt_tokens": 8,
                "logit_positions": list(range(8, 24)),
                "greedy_ids": list(range(100, 115)) + [999],
                "eog_ids": [999],
            }
            path = root / "capture.json"
            path.write_text(json.dumps(metadata))
            original_metadata = path.read_bytes()
            spec = CaptureSpec("one", reference(path), tuple(files))
            capture = load_capture(spec, 128 * 1024, hidden=4)
            self.assertEqual([w.anchor for w in capture.windows], [8, 9])
            self.assertEqual(capture.windows[-1].labels[-1], 999)
            self.assertEqual(float(capture.features[0, 4]), 6)
            self.assertLessEqual(capture.bytes, 128 * 1024)
            with self.assertRaisesRegex(ValueError, "budget"):
                load_capture(spec, 32, hidden=4)
            cache = CaptureCache(128 * 1024, hidden=4)
            for index in range(5):
                cache.get(CaptureSpec(str(index), spec.metadata, spec.files))
                self.assertEqual(len(cache.entries), 1)
            cast(list[int], metadata["greedy_ids"])[0] = 1000
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "disagreement"):
                load_capture(
                    CaptureSpec("changed", reference(path), tuple(files)), 128 * 1024, hidden=4
                )
            path.write_bytes(original_metadata)
            files[0].path.write_bytes(b"x" * files[0].size)
            with self.assertRaisesRegex(ValueError, "content"):
                load_capture(spec, 128 * 1024, hidden=4)

    def fixture_worker(self, identity: str) -> TensorWorker:
        model, features = self.tiny_model()
        window = Window("fixture", 2, tuple(range(3, 18)))
        capture = Capture(tuple(range(20)), features, (window,))

        class Source:
            def sample(self, cursor: int) -> tuple[Capture, Window]:
                if cursor < 0:
                    raise ValueError("negative cursor")
                return capture, window

        return TensorWorker(model, Source(), identity)

    def test_tensor_resume_next_update_and_branch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-resume-") as directory:
            store = Store(Path(directory))
            original = self.fixture_worker("recipe")
            original.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            parent = original.checkpoint(store, 4 * 1024 * 1024)
            parent_bytes = parent.artifact.path.read_bytes()
            original.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            expected_loss = original.progress.last_loss
            expected = {
                name: value.detach().clone() for name, value in original.model.named_parameters()
            }
            resumed = self.fixture_worker("recipe")
            resumed.restore(parent, 8 * 1024 * 1024)
            self.assertEqual(resumed.updates, 1)
            resumed.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            self.assertEqual(resumed.progress.last_loss, expected_loss)
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]), name)
            self.assertEqual(parent.artifact.path.read_bytes(), parent_bytes)
            spent = original.progress.wall_seconds
            original.restore(parent, 8 * 1024 * 1024)
            self.assertEqual(
                (
                    original.progress.updates,
                    original.progress.attempts,
                    original.progress.discarded,
                ),
                (1, 2, 1),
            )
            self.assertGreaterEqual(original.progress.wall_seconds, spent)
            original.run(Monotonic(), Budget(2, 60, 3), lambda: False)
            for name, value in original.model.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]), name)

            branch = self.fixture_worker("changed-recipe")
            branch.restore(parent, 8 * 1024 * 1024, mode="branch")
            self.assertEqual((branch.updates, branch.cursor, branch.optimizer_updates), (1, 0, 0))
            self.assertFalse(branch.optimizer.state)
            branch.run(Monotonic(), Budget(2, 60, 2), lambda: False)
            self.assertEqual((branch.updates, branch.cursor, branch.optimizer_updates), (2, 1, 1))
            self.assertIn(parent.artifact.sha256, branch.lineage)

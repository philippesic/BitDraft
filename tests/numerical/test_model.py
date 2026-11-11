import hashlib
import importlib.metadata
import json
import struct
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
from w1a1_eagle.contracts import Record, number, record, sequence
from w1a1_eagle.data import Artifact, Window
from w1a1_eagle.identity import source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.model import HE4, Geometry, Layer, Projection, hard_sign, rms, rope
from w1a1_eagle.numerical.captures import Capture, CaptureCache, CaptureSpec, load_capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.entry import admission
from w1a1_eagle.numerical.export import export_he4, packed_signs
from w1a1_eagle.numerical.weights import load_he4, shapes
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import canonical, read, reference
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
        for name, values in (
            ("inputs.npy", [0] * 24),
            ("label_positions.npy", list(range(8, 24))),
            ("labels.npy", [1] * 15 + [19]),
        ):
            path = root / name
            np.save(path, np.asarray(values, dtype="int32"), allow_pickle=False)
            files.append(reference(self.reference(path)))
        source = root / "source.json"
        source.write_bytes(
            canonical(
                {
                    "id": "one",
                    "group_id": "one",
                    "source": "synthetic-fixture",
                    "role": "fit",
                    "source_ancestry": {"fixture": "only"},
                    "identities": {"rendered_sha256": "c" * 64},
                    "pool": "unused_modern_reserved",
                }
            )
        )
        files.append(reference(self.reference(source)))
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
                            "prior_use": "unobserved",
                            "metadata": reference(self.reference(metadata)),
                            "files": files,
                        }
                    ],
                }
            )
        )
        return self.reference(manifest)

    def fixture_index(self, root: Path, corpus: Artifact) -> tuple[Artifact, Record]:
        row = record(sequence(read(corpus.path)["records"])[0])
        metadata_ref = record(row["metadata"])
        metadata = read(Path(str(metadata_ref["path"])))
        exclusions = root / "exclusions.json"
        exclusions.write_bytes(canonical({"identities": ["heldout24"]}))
        receipt: Record = {
            "status": "VERIFIED_NATIVE_HARD_LABEL_CAPTURE",
            "id": "one",
            "group_id": "one",
            "metadata": metadata_ref,
            "files": row["files"],
            "greedy_vs_full_logits_argmax": True,
            "no_post_eog_labels": True,
            "taps": [2, 6, 10, 14, 18, 22, 26, 30],
            "feature_semantics": "zero_based_pre_block_input_all_tokens",
            "feature_shape": [24, 256],
            "windows": 2,
            "unique_label_positions_in_windows": 16,
            "unique_label_positions": 16,
            "verified_rows": 16,
            "sequences": 1,
        }
        proof = root / "proof.json"
        proof.write_bytes(canonical(receipt))
        index = root / "index.json"
        index.write_bytes(
            canonical(
                {
                    "status": "ADMITTED_METADATA",
                    "manifest_sha256": "b" * 64,
                    "target_sha256": "a" * 64,
                    "exclusions": reference(self.reference(exclusions)),
                    "rows": [
                        {
                            "id": "one",
                            "receipt_pin": reference(self.reference(proof)),
                            "receipt": receipt,
                            "metadata": metadata,
                            "anchors": [8, 9],
                        }
                    ],
                    "sequences": 1,
                    "windows": 2,
                    "unique_labels_in_windows": 16,
                    "note": "synthetic producer contract, not native qualification",
                }
            )
        )
        adapter: Record = {
            "kind": "he4fresh-index-v1",
            "logical_root": str(root),
            "metadata_root": str(root),
            "data_root": str(root),
            "manifest_sha256": "b" * 64,
            "target_sha256": "a" * 64,
            "exclusions": reference(self.reference(exclusions)),
            "sequences": 1,
            "windows": 2,
            "unique_positions": 16,
        }
        return self.reference(index), adapter

    def test_default_geometry_against_retained_header(self) -> None:
        header = read(Path("evidence/he4-initializer-header.json"))
        expected = {name: list(shape) for name, shape in shapes(Geometry()).items()}
        self.assertEqual(expected, header["tensor_shapes"])
        self.assertEqual(header["target_layers"], [2, 6, 10, 14, 18, 22, 26, 30])

    def test_projection_scalar_oracle(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())
        versions = dict(
            requirement.split("==", 1)
            for requirement in project["project"]["optional-dependencies"]["numerical"]
        )
        for name in ("torch", "gguf", "numpy"):
            self.assertEqual(importlib.metadata.version(name), versions[name])
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

    def test_sign_packing_independent_words(self) -> None:
        value = torch.full((1, 33), -1.0, dtype=torch.float32)
        value[0, 0] = -0.0
        value[0, 31] = 1
        value[0, 32] = 1
        actual = packed_signs(value).tobytes()
        self.assertEqual(actual, struct.pack("<II", 0x80000001, 1))
        self.assertEqual(struct.unpack("<ii", actual), (-2147483647, 1))

    def test_historical_arithmetic_characterization(self) -> None:
        captured = record(
            cast(object, json.loads(Path("evidence/arithmetic-reference.json").read_text()))
        )
        expected_projection = record(captured["projection"])
        weight = torch.tensor([[0.2, -0.4, -1.0], [1.0, 1.4, -0.0]])
        value = torch.tensor([[0.1, 0.7, -0.2], [-0.3, 0.5, 1.2]], requires_grad=True)
        projection = Projection(
            weight,
            4,
            nn.Parameter(torch.tensor([0.1, -0.2, 0.3])),
            nn.Parameter(torch.tensor(0.7).log()),
        )
        with torch.no_grad():
            projection.midpoint.copy_(torch.tensor([0.05, -0.03]))
        output = projection.forward(value)
        output.sum().backward()
        self.assertTrue(
            torch.allclose(
                output, torch.tensor(expected_projection["output"]), atol=1e-7, rtol=1e-6
            )
        )
        assert value.grad is not None
        self.assertTrue(
            torch.allclose(
                value.grad, torch.tensor(expected_projection["input_grad"]), atol=1e-7, rtol=1e-6
            )
        )
        gradients = record(expected_projection["gradients"])
        for name, parameter in projection.named_parameters():
            assert parameter.grad is not None
            self.assertTrue(
                torch.allclose(parameter.grad, torch.tensor(gradients[name]), atol=1e-7, rtol=1e-6),
                name,
            )
        optimizer = torch.optim.AdamW(
            projection.parameters(), lr=1e-4, weight_decay=0, foreach=False
        )
        optimizer.step()
        updated = record(expected_projection["updated"])
        for name, parameter in projection.named_parameters():
            self.assertTrue(
                torch.allclose(parameter, torch.tensor(updated[name]), atol=1e-7, rtol=1e-6), name
            )
        cfg = Geometry(hidden=4, intermediate=8, heads=2, kv_heads=1, head_dim=2)
        tensor_shapes = {
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
        }
        tensors = {
            "blk.0." + name + ".weight": torch.ones(shape)
            if len(shape) == 1
            else torch.sin(torch.arange(int(np.prod(shape)), dtype=torch.float32) * 0.37).reshape(
                shape
            )
            * 0.1
            for name, shape in tensor_shapes.items()
        }
        layer = Layer(tensors, "blk.0.", cfg)
        encoded = (
            torch.cos(torch.arange(12, dtype=torch.float32) * 0.23).reshape(3, 4) * 0.1
        ).requires_grad_()
        state = (
            torch.sin(torch.arange(64, dtype=torch.float32) * 0.13).reshape(16, 4) * 0.1
        ).requires_grad_()
        result = layer.forward(
            state,
            layer.context(encoded, torch.arange(3)),
            torch.arange(3, 19),
            torch.ones((16, 19), dtype=torch.bool),
        )
        result.sum().backward()
        expected_layer = record(captured["layer"])
        assert state.grad is not None and encoded.grad is not None
        for name, actual in (
            ("output", result),
            ("state_grad", state.grad),
            ("encoded_grad", encoded.grad),
        ):
            self.assertTrue(
                torch.allclose(actual, torch.tensor(expected_layer[name]), atol=1e-7, rtol=1e-6),
                name,
            )

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
            export_store = Store(root / "exports")
            exporter = TensorWorker(model, corpus, "export-fixture")
            exported = export_he4(
                exporter,
                source_ref,
                vocabulary_ref,
                export_store,
                8 * 1024 * 1024,
                32 * 1024 * 1024,
            )
            tensors = {
                tensor.name: tensor
                for tensor in gguf.GGUFReader(str(exported.artifact.path)).tensors
            }
            self.assertEqual(
                tensors["blk.0.ffn_gate.w1a1_packed"].data.tobytes(), b"\xff" * (256 * 256 // 8)
            )
            self.assertEqual(tensors["output.weight"].tensor_type.name, "Q4_K")
            self.assertEqual(tensors["output.weight"].data.tobytes(), b"\0" * (20 * 144))
            self.assertNotIn("token_embd.weight", tensors)
            with self.assertRaisesRegex(ValueError, "byte budget"):
                export_he4(
                    exporter, source_ref, vocabulary_ref, export_store, 100, 32 * 1024 * 1024
                )
            self.assertEqual(export_store.latest("export-fixture"), exported)
            self.assertEqual(corpus.count, 2)
            self.assertEqual(corpus.unique_positions, 16)
            expected = [corpus.locate(i)[1] for i in range(5)]
            self.assertEqual(corpus.locate(0)[1], corpus.locate(corpus.count)[1])
            self.assertEqual(
                expected,
                [Corpus(corpus_ref, 17, 64 * 1024 * 1024, 256).locate(i)[1] for i in range(5)],
            )
            self.assertEqual(corpus.exposure(4)["mean_visits"], 60 / 16)
            intent: Record = {
                "schema": 2,
                "objective": "exercise CPU source integration",
                "checkpoint_updates": 1,
                "checkpoint_seconds": 30,
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
                "backing_available_bytes": 256 * 1024 * 1024,
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
            sealed = Intent.load(intent_path)
            editable = sealed.raw
            editable["authority_scope"] = "altered"
            self.assertEqual(sealed.raw["authority_scope"], "disposable-CPU-test")
            with self.assertRaisesRegex(ValueError, "scope"):
                execute(intent_path, root / "not-authorized", "unrelated-task")
            self.assertFalse((root / "not-authorized").exists())
            invalid_policy = dict(intent)
            invalid_policy["science_policy"] = {
                "id": "deliberate-refusal",
                "min_unique_positions": 17,
                "max_mean_visits": 2,
            }
            intent_path.write_bytes(canonical(invalid_policy))
            with self.assertRaisesRegex(ValueError, "distinct-data"):
                admission(Intent.load(intent_path), corpus)
            refused = execute(intent_path, root / "refused-retry", "disposable-CPU-test")
            self.assertEqual(record(refused["latest"])["outcome"], "refused")
            self.assertIsNone(refused["worker"])
            self.assertFalse((root / "refused-retry/states/latest.json").exists())
            intent_path.write_bytes(canonical(intent))
            recovered = execute(intent_path, root / "refused-retry", "disposable-CPU-test")
            self.assertEqual(recovered["attempts"], 2)
            self.assertEqual(record(recovered["latest"])["outcome"], "complete", recovered)
            self.assertGreater(
                number(recovered["cumulative_attempt_seconds"]),
                number(refused["cumulative_attempt_seconds"]),
            )
            invalid_policy["schema"] = 1
            intent_path.write_bytes(canonical(invalid_policy))
            with self.assertRaisesRegex(ValueError, "schema is 2"):
                Intent.load(intent_path)
            intent_path.write_bytes(canonical(intent))

            def mutate_caller_intent(_admission: Record) -> None:
                changed = dict(intent)
                changed["authority_scope"] = "caller-changed-after-preflight"
                intent_path.write_bytes(canonical(changed))

            report = execute(
                intent_path, root / "attempt", "disposable-CPU-test", mutate_caller_intent
            )
            sealed_path = next((root / "attempt/intents").glob("*.json"))
            self.assertEqual(
                Intent.load_sealed(sealed_path).raw["authority_scope"], "disposable-CPU-test"
            )
            self.assertEqual(Intent.load_sealed(sealed_path).base, intent_path.parent)
            snapshot = next((root / "attempt/sources").iterdir())
            self.assertEqual(source_identity(snapshot), intent["code_sha256"])
            intent_path.write_bytes(canonical(intent))
            self.assertEqual(record(report["latest"])["outcome"], "complete", report)
            self.assertEqual(record(report["worker"])["updates"], 1)
            self.assertIn("unknown", str(report["scientific"]))
            latest = Store(root / "attempt/states").latest(
                str(record(report["admission"])["recipe"])
            )
            self.assertEqual(latest.updates, 1)
            intent["mode"] = "resume"
            intent["parent"] = reference(latest.artifact)
            intent["budget"] = {"updates": 3, "wall_seconds": 60, "attempts": 4, "retries": 0}
            intent["science_policy"] = {
                "id": "synthetic-fixture-only",
                "min_unique_positions": 16,
                "max_mean_visits": 5,
            }
            intent_path.write_bytes(canonical(intent))
            resumed_report = execute(intent_path, root / "attempt", "disposable-CPU-test")
            self.assertEqual(
                record(resumed_report["latest"])["outcome"], "complete", resumed_report
            )
            self.assertEqual(record(resumed_report["worker"])["updates"], 3)
            self.assertEqual(resumed_report["attempts"], 2)
            self.assertTrue(
                any(
                    row.get("kind") == "checkpoint" and row.get("updates") == 2
                    for row in Journal(root / "attempt").events()
                )
            )
            self.assertGreater(
                number(resumed_report["cumulative_attempt_seconds"]),
                number(report["cumulative_attempt_seconds"]),
            )
            index_ref, adapter = self.fixture_index(root, corpus_ref)
            intent.update(
                mode="fresh",
                parent=None,
                corpus=reference(index_ref),
                input_adapter=adapter,
                budget={"updates": 1, "wall_seconds": 60, "attempts": 2, "retries": 0},
            )
            intent_path.write_bytes(canonical(intent))
            indexed_report = execute(intent_path, root / "indexed-attempt", "disposable-CPU-test")
            self.assertEqual(
                record(indexed_report["latest"])["outcome"], "complete", indexed_report
            )
            self.assertEqual(record(indexed_report["worker"])["updates"], 1)

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
            broken = self.fixture_worker("recipe")
            broken.progress.needs_recovery = True
            with self.assertRaisesRegex(ValueError, "incomplete"):
                broken.checkpoint(Store(Path(directory) / "invalid"), 4 * 1024 * 1024)
            state = torch.load(parent.artifact.path, map_location="cpu", weights_only=True)
            self.assertEqual(state["schema"], 3)
            self.assertNotIn("vocabulary", state["model"])
            self.assertNotIn("fusion.operand", state["model"])
            self.assertEqual(state["frozen_sha256"], original.frozen_sha256)
            first_state = next(iter(state["optimizer"]["state"].values()))
            first_state["exp_avg"] = torch.zeros((1,))
            malformed = Store(Path(directory) / "malformed").commit_with(
                lambda output: torch.save(state, output), "recipe", 1, 4 * 1024 * 1024
            )
            with self.assertRaisesRegex(ValueError, "moment geometry"):
                self.fixture_worker("recipe").restore(malformed, 8 * 1024 * 1024)

    def test_frozen_state_guards_and_schema_refusal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-frozen-") as directory:
            root = Path(directory)
            worker = self.fixture_worker("recipe")
            parent = worker.checkpoint(Store(root / "states"), 4 * 1024 * 1024)
            changed = self.fixture_worker("recipe")
            with torch.no_grad():
                changed.model.vocabulary[0, 0] += 1
            with self.assertRaisesRegex(ValueError, "frozen source"):
                changed.checkpoint(Store(root / "changed"), 4 * 1024 * 1024)
            self.assertFalse(changed.ready)
            other = self.fixture_worker("recipe")
            with torch.no_grad():
                other.model.vocabulary[0, 0] += 1
            replacement = TensorWorker(other.model, other.samples, "recipe")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                replacement.restore(parent, 8 * 1024 * 1024)
            state = torch.load(parent.artifact.path, map_location="cpu", weights_only=True)
            state["schema"] = 2
            old = Store(root / "old").commit_with(
                lambda output: torch.save(state, output), "recipe", 0, 4 * 1024 * 1024
            )
            with self.assertRaisesRegex(ValueError, "unsupported tensor state"):
                self.fixture_worker("recipe").restore(old, 8 * 1024 * 1024)

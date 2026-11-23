import tempfile
import unittest
from pathlib import Path
from typing import TypeAlias

import gguf
import numpy as np
import torch

from tests.numerical.test_model import (
    OMITTED_RECIPE,
    authenticated,
    execute_leased,
    fixture_corpus,
    gguf_fixture,
    loader_fixture,
)
from tests.numerical.test_rounds import q4_k
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact
from w1a1_eagle.intent import Intent
from w1a1_eagle.model import HE4, FeedForwardTable, Geometry, Projection
from w1a1_eagle.numerical.baseline import load_baseline
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.export import export_he4, export_plain
from w1a1_eagle.numerical.k_tables import plain_tables, quantize, quantizer_binary
from w1a1_eagle.numerical.weights import is_table, load_he4, shapes
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import canonical, reference
from w1a1_eagle.training import Budget, Optimization

Floats: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
MEMORY = 64 * 1024 * 1024
QUANTIZER = """#!/usr/bin/env python3
import hashlib, struct, sys
widths = {"Q4_K": 144, "Q6_K": 210}
source, target, rows, columns, path, output = sys.argv[1:]
rows, columns, width = int(rows), int(columns), widths[target]
data = open(path, "rb").read()
step = columns * 4
scale = struct.pack("<e", 0.01)
result = bytearray()
for row in range(rows):
    seed = data[row * step:(row + 1) * step]
    for block in range(columns // 256):
        digest = b""
        while len(digest) < width:
            digest += hashlib.sha256(seed + bytes([block, len(digest) % 251])).digest()
        chunk = bytearray(digest[:width])
        if target == "Q4_K":
            chunk[0:4] = scale + bytes(2)
        else:
            chunk[208:210] = scale
        result += chunk
open(output, "wb").write(bytes(result))
"""
PLAIN_RATES = {"norm": 1e-3, "feed_forward": 1e-2}
CFG = Geometry(hidden=256, intermediate=256, heads=32, kv_heads=8, head_dim=8, layers=1)


def quantizer(root: Path) -> Artifact:
    script = root / "quantizer"
    script.write_text(QUANTIZER)
    script.chmod(0o755)
    built = root / "quantizer.json"
    built.write_bytes(canonical({"binary": reference(authenticated(script))}))
    return authenticated(built)


def kind(name: str) -> str:
    return "Q6_K" if name.endswith("ffn_down.weight") else "Q4_K"


def source_values(source: Artifact) -> dict[str, Floats]:
    return {
        item.name: gguf.dequantize(item.data, item.tensor_type)
        .astype(np.float32)
        .reshape(tuple(int(v) for v in reversed(item.shape)))
        for item in gguf.GGUFReader(str(source.path)).tensors
    }


def baseline_file(root: Path, name: str, source: Artifact, built: Artifact | None) -> Artifact:
    values = source_values(source)
    generator = np.random.default_rng(11)
    path = root / name
    writer = gguf.GGUFWriter(str(path), "dflash")
    writer.add_string("general.name", "fixture Baseline")
    writer.add_array("dflash.target_layers", [2, 6, 10, 14, 18, 22, 26, 30])
    writer.add_uint32("tokenizer.ggml.mask_token_id", 19)
    binary = None if built is None else quantizer_binary(built)
    kinds = gguf.GGMLQuantizationType
    for tensor, shape in shapes(CFG).items():
        value = values[tensor]
        if len(shape) == 1:
            writer.add_tensor(tensor, value, raw_dtype=kinds.F32)
        elif is_table(tensor):
            writer.add_tensor(tensor, q4_k(shape[0], shape[1], generator), raw_dtype=kinds.Q4_K)
        elif binary is None:
            writer.add_tensor(tensor, gguf.quantize(value, kinds.BF16), raw_dtype=kinds.BF16)
        else:
            value.tofile(root / "table.f32")
            coded = quantize(
                binary.path,
                "F32",
                kind(tensor),
                shape[0],
                shape[1],
                root / "table.f32",
                root / "table.bin",
            )
            target = kinds.Q6_K if kind(tensor) == "Q6_K" else kinds.Q4_K
            writer.add_tensor(tensor, coded, raw_dtype=target)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return authenticated(path)


def plain_model(source: Artifact, vocabulary: Artifact, baseline: Artifact) -> HE4:
    return load_he4(
        source,
        vocabulary,
        MEMORY,
        CFG,
        None,
        "baseline-k",
        baseline,
        plain_feed_forward=True,
    )


def payloads(path: Path) -> dict[str, tuple[str, bytes]]:
    return {
        item.name: (item.tensor_type.name, np.ascontiguousarray(item.data).tobytes())
        for item in gguf.GGUFReader(str(path)).tensors
    }


def fields(path: Path) -> dict[str, object]:
    return {
        key: field.contents()
        for key, field in gguf.GGUFReader(str(path)).fields.items()
        if not key.startswith("GGUF.")
    }


class TestPlainModel(unittest.TestCase):
    def test_initial_plain_model_computes_the_baseline_graph(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-plain-model-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, CFG, seed=5)
            built = quantizer(root)
            coded = baseline_file(root, "coded.gguf", source, built)
            plain_baseline = baseline_file(root, "plain.gguf", source, None)
            model = plain_model(source, vocabulary, coded)
            reference_model = load_baseline(plain_baseline, vocabulary, 1 << 30, CFG)
            self.assertFalse(any(isinstance(m, Projection) for m in model.modules()))
            tables = [m for m in model.modules() if isinstance(m, FeedForwardTable)]
            self.assertEqual(len(tables), 3)
            self.assertTrue(all(table.operand.requires_grad for table in tables))
            self.assertEqual(
                dict(model.feed_forward_types),
                {
                    f"blk.0.ffn_{part}.weight": kind(f"blk.0.ffn_{part}.weight")
                    for part in ("gate", "up", "down")
                },
            )
            self.assertEqual(model.mask, reference_model.mask)
            features = torch.randn((12, 8 * 256), generator=torch.Generator().manual_seed(2))
            with torch.no_grad():
                for anchor, slots in ((6, 4), (3, 16)):
                    trained = model.forward(features, tuple(range(12)), anchor, slots)
                    served = reference_model.forward(features, tuple(range(12)), anchor, slots)
                    self.assertTrue(torch.equal(trained, served))
            with self.assertRaisesRegex(ValueError, "baseline-k"):
                load_he4(source, vocabulary, MEMORY, CFG, plain_feed_forward=True)


class TestPlainUpdate(unittest.TestCase):
    def test_one_update_changes_tables_and_export_and_reads_back(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-plain-update-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, CFG, seed=5)
            built = quantizer(root)
            baseline = baseline_file(root, "coded.gguf", source, built)
            model = plain_model(source, vocabulary, baseline)
            corpus = Corpus(fixture_corpus(root, 256), 17, MEMORY, 256)
            worker = TensorWorker(
                model, corpus, "plain", optimization=Optimization(dict(PLAIN_RATES))
            )
            names = {str(group["name"]) for group in worker.optimizer.param_groups}
            self.assertEqual(names, {"norm", "feed_forward"})
            store = Store(root / "exports")
            initial = export_plain(worker, baseline, built, store, 8 << 20, 32 << 20)
            self.assertEqual(payloads(initial.artifact.path), payloads(baseline.path))
            self.assertEqual(fields(initial.artifact.path), fields(baseline.path))
            self.assertFalse(any(key.startswith("dflash.w1ax.") for key in fields(baseline.path)))
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            self.assertEqual(
                worker.run(Monotonic(), Budget(1, 60, 1), lambda: False), "update-budget"
            )
            after = dict(model.named_parameters())
            for name, value in before.items():
                self.assertFalse(torch.equal(value, after[name]), name)
            exported = export_plain(worker, baseline, built, store, 8 << 20, 32 << 20)
            first, second = payloads(initial.artifact.path), payloads(exported.artifact.path)
            self.assertEqual(set(first), set(second))
            for name in first:
                changed = ".ffn_" in name or "norm" in name
                self.assertEqual(first[name][0], second[name][0])
                self.assertEqual(first[name][1] != second[name][1], changed, name)
            served = plain_tables(model, quantizer_binary(built).path)
            for tensor, (kind_name, data) in second.items():
                if ".ffn_" in tensor and not tensor.endswith("norm.weight"):
                    operand = served["blocks.0." + tensor.split("_")[1].split(".")[0] + ".operand"]
                    rows = np.frombuffer(data, dtype=np.uint8).reshape(operand.shape[0], -1)
                    restored = gguf.dequantize(rows, getattr(gguf.GGMLQuantizationType, kind_name))
                    self.assertTrue(np.array_equal(restored.reshape(operand.shape), operand))
            with self.assertRaisesRegex(ValueError, "one-bit"):
                export_he4(worker, source, vocabulary, Store(root / "he4"), 8 << 20, 32 << 20)
            state = worker.checkpoint(Store(root / "states"), MEMORY)
            resumed = TensorWorker(
                plain_model(source, vocabulary, baseline),
                corpus,
                "plain",
                optimization=Optimization(dict(PLAIN_RATES)),
            )
            resumed.restore(state, MEMORY)
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(value, after[name]), name)
            self.assertEqual(
                resumed.run(Monotonic(), Budget(2, 60, 2), lambda: False), "update-budget"
            )
            self.assertEqual(resumed.updates, 2)
            with self.assertRaisesRegex(ValueError, "fresh, resume and continue"):
                TensorWorker(
                    plain_model(source, vocabulary, baseline),
                    corpus,
                    "plain",
                    optimization=Optimization(dict(PLAIN_RATES)),
                ).restore(state, MEMORY, "branch")
            with self.assertRaisesRegex(ValueError, "feed_forward"):
                TensorWorker(plain_model(source, vocabulary, baseline), corpus, "plain")


def plain_intent(base: Record, baseline: Artifact, built: Artifact) -> Record:
    return {
        **{key: value for key, value in base.items() if key != "initialization"},
        "frozen_format": "baseline-k",
        "frozen_baseline": reference(baseline),
        "feed_forward": "plain",
        "table_quantizer": reference(built),
        "optimization": {
            "step_sizes": PLAIN_RATES,
            "schedule": {
                "decay": "constant",
                "warmup_updates": 0,
                "start_update": 0,
                "final_update": None,
            },
            "latent_scale": {"kind": "raw"},
        },
    }


class TestPlainIntent(unittest.TestCase):
    def test_setting_is_opt_in_and_refuses_one_bit_options(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-plain-intent-") as directory:
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

            def parse(value: Record) -> Intent:
                path.write_bytes(canonical(value))
                return Intent.load(path)

            self.assertEqual(parse(base).recipe, OMITTED_RECIPE)
            self.assertEqual(parse({**base, "feed_forward": "one-bit"}).recipe, OMITTED_RECIPE)
            self.assertFalse(parse(base).plain_feed_forward)
            baseline = Artifact(root / "baseline.gguf", "8" * 64, 7)
            built = Artifact(root / "quantizer.json", "9" * 64, 7)
            plain = plain_intent(base, baseline, built)
            admitted = parse(plain)
            self.assertTrue(admitted.plain_feed_forward)
            self.assertEqual(admitted.optimization.groups, ("norm", "feed_forward"))
            self.assertNotEqual(admitted.recipe, OMITTED_RECIPE)
            other = parse(
                {**plain, "table_quantizer": reference(Artifact(built.path, "7" * 64, 7))}
            )
            self.assertNotEqual(other.recipe, admitted.recipe)
            optimization = record(plain["optimization"])
            refusals: list[tuple[Record, str]] = [
                ({**plain, "feed_forward": "binary"}, "one-bit or plain"),
                (
                    {key: value for key, value in plain.items() if key != "table_quantizer"},
                    "table_quantizer",
                ),
                ({**base, "table_quantizer": reference(built)}, "table_quantizer"),
                (
                    {key: value for key, value in plain.items() if key != "optimization"},
                    "norm and feed_forward",
                ),
                ({**base, "optimization": optimization}, "norm and feed_forward"),
                (
                    {
                        **plain,
                        "optimization": {
                            **optimization,
                            "latent_scale": {"kind": "row", "multiple": 3},
                        },
                    },
                    "sign latents",
                ),
                ({**plain, "initialization": base["initialization"]}, "initialization"),
                (
                    {
                        **plain,
                        "features": {
                            "scale": "fixed",
                            "clip": "fixed",
                            "gain": "off",
                            "midpoint": "off",
                        },
                    },
                    "features",
                ),
                ({**plain, "slot_rounding": "anchor-mask"}, "slot_rounding"),
                ({**plain, "arithmetic": "fast-exact"}, "arithmetic"),
                (
                    {
                        key: value
                        for key, value in plain.items()
                        if key not in ("frozen_format", "frozen_baseline")
                    },
                    "baseline-k",
                ),
            ]
            for changes, message in refusals:
                with self.assertRaisesRegex(ValueError, message):
                    parse(changes)


class TestPlainRun(unittest.TestCase):
    def test_fresh_run_writes_initial_and_final_exports(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-plain-run-") as directory:
            root = Path(directory)
            loaded = loader_fixture(root)
            sized = root / "sized"
            sized.mkdir()
            source, vocabulary = gguf_fixture(sized, CFG, seed=5)
            built = quantizer(root)
            baseline = baseline_file(root, "coded.gguf", source, built)
            intent = {
                **plain_intent(loaded.intent, baseline, built),
                "source": reference(source),
                "vocabulary": reference(vocabulary),
                "geometry": {
                    "hidden": 256,
                    "intermediate": 256,
                    "heads": 32,
                    "kv_heads": 8,
                    "head_dim": 8,
                    "layers": 1,
                },
                "initial_export": True,
            }
            intent_path = root / "plain-intent.json"
            intent_path.write_bytes(canonical(intent))
            output = root / "run"
            execute_leased(intent_path, output)
            exports = sorted(
                path
                for path in (output / "exports").iterdir()
                if path.is_file() and path.read_bytes()[:4] == b"GGUF"
            )
            self.assertEqual(len(exports), 2)
            kinds = {name: value[0] for name, value in payloads(baseline.path).items()}
            initial = [path for path in exports if payloads(path) == payloads(baseline.path)]
            self.assertEqual(len(initial), 1)
            final = [path for path in exports if path not in initial][0]
            self.assertEqual({k: v[0] for k, v in payloads(final).items()}, kinds)
            self.assertEqual(fields(final), fields(baseline.path))

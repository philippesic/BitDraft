import tempfile
import unittest
from pathlib import Path
from typing import cast

import gguf
import numpy as np
import torch
from torch import nn

from tests.numerical.test_model import authenticated, fixture_corpus, gguf_fixture, tiny_model
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.drafting import generated_anchors, simulate, tally
from w1a1_eagle.model import HE4, Geometry, Layer, Projection, Table
from w1a1_eagle.numerical.baseline import load_baseline
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.probes import (
    ABLATIONS,
    AblatedProjection,
    Ablation,
    RangeProbe,
    Window,
    ablated,
    apply_ablation,
    substituted,
)
from w1a1_eagle.numerical.round_diagnostics import propose
from w1a1_eagle.numerical.weights import load_he4, shapes
from w1a1_eagle.numerical.worker import TensorWorker


def q4_k(
    rows: int, columns: int, generator: np.random.Generator
) -> "np.ndarray[tuple[int, ...], np.dtype[np.uint8]]":
    data = generator.integers(0, 256, (rows, columns // 256 * 144), dtype=np.uint8)
    blocks = data.reshape(rows, columns // 256, 144)
    blocks[:, :, 0:4] = np.array([0.01, 0.002], dtype=np.float16).view(np.uint8)
    return data


def baseline_fixture(root: Path, cfg: Geometry, untied: bool = False) -> tuple[Path, Path]:
    generator = np.random.default_rng(3)
    drafter = root / "baseline.gguf"
    writer = gguf.GGUFWriter(str(drafter), "dflash")
    writer.add_array("dflash.target_layers", [2, 6, 10, 14, 18, 22, 26, 30])
    writer.add_uint32("tokenizer.ggml.mask_token_id", 19)
    for name, shape in shapes(cfg).items():
        if len(shape) == 1:
            values = (1 + generator.standard_normal(shape) * 0.1).astype(np.float32)
            writer.add_tensor(name, values, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor(
                name, q4_k(shape[0], shape[1], generator), raw_dtype=gguf.GGMLQuantizationType.Q4_K
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    target = root / "target.gguf"
    writer = gguf.GGUFWriter(str(target), "qwen35")
    writer.add_tensor(
        "token_embd.weight",
        q4_k(20, cfg.hidden, generator),
        raw_dtype=gguf.GGMLQuantizationType.Q4_K,
    )
    if untied:
        writer.add_tensor(
            "output.weight",
            q4_k(20, cfg.hidden, generator),
            raw_dtype=gguf.GGMLQuantizationType.Q4_K,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return drafter, target


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def trained_like(model: HE4) -> HE4:
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for module in model.blocks:
            block = cast(Layer, module)
            for projection in (block.gate, block.down):
                projection.log_gain.copy_(
                    torch.randn(projection.log_gain.shape, generator=generator) * 0.3
                )
                projection.log_clip.fill_(-0.5)
            for projection in (block.gate, block.up, block.down):
                projection.midpoint.copy_(
                    torch.randn(projection.midpoint.shape, generator=generator) * 0.01
                )
    return model


class TestBaselineModel(unittest.TestCase):
    def test_baseline_tables_are_dequantized_plain_linear_layers(self) -> None:
        torch.set_num_threads(1)
        cfg = Geometry(hidden=256, intermediate=256, heads=32, kv_heads=8, head_dim=8, layers=2)
        with tempfile.TemporaryDirectory(prefix="binary-baseline-") as directory:
            root = Path(directory)
            drafter_path, target_path = baseline_fixture(root, cfg)
            model = load_baseline(
                authenticated(drafter_path), authenticated(target_path), 1 << 30, cfg
            )
            reader = gguf.GGUFReader(str(drafter_path))
            for item in reader.tensors:
                shape = tuple(int(v) for v in reversed(item.shape))
                expected = torch.tensor(
                    gguf.dequantize(item.data, item.tensor_type).reshape(shape),
                    dtype=torch.float32,
                )
                key = name(item.name)
                actual = (
                    model.get_buffer(key) if key.endswith(".operand") else model.get_parameter(key)
                )
                self.assertTrue(torch.equal(actual, expected), item.name)
            table = gguf.GGUFReader(str(target_path)).tensors[0]
            vocabulary = gguf.dequantize(table.data, table.tensor_type).reshape(20, 256)
            self.assertTrue(torch.equal(model.vocabulary, torch.tensor(vocabulary)))
            self.assertEqual(model.mask, 19)
            block = cast(Layer, model.blocks[0])
            for part in ("gate", "up", "down"):
                self.assertIsInstance(getattr(block, part), Table)
            value = torch.randn((4, 256), generator=torch.Generator().manual_seed(1))
            linear = torch.nn.functional.linear
            silu = torch.nn.functional.silu
            with torch.no_grad():
                served = block.down.forward(
                    silu(block.gate.forward(value)) * block.up.forward(value)
                )
                gate, up, down = (
                    model.get_buffer(f"blocks.0.{part}.operand") for part in ("gate", "up", "down")
                )
                reference = linear(silu(linear(value, gate)) * linear(value, up), down)
            self.assertTrue(torch.equal(served, reference))
            self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
            features = torch.randn((12, 8 * 256), generator=torch.Generator().manual_seed(2))
            with torch.no_grad():
                logits = model.forward(features, tuple(range(12)), 6, 4)
            self.assertEqual(tuple(logits.shape), (4, 20))
            self.assertTrue(bool(torch.isfinite(logits).all()))
            untied = root / "untied"
            untied.mkdir()
            drafter_path, target_path = baseline_fixture(untied, cfg, untied=True)
            with self.assertRaisesRegex(ValueError, "output table"):
                load_baseline(authenticated(drafter_path), authenticated(target_path), 1 << 30, cfg)
            with self.assertRaisesRegex(ValueError, "coverage"):
                load_baseline(
                    authenticated(drafter_path),
                    authenticated(target_path),
                    1 << 30,
                    Geometry(
                        hidden=256, intermediate=256, heads=32, kv_heads=8, head_dim=8, layers=1
                    ),
                )


def name(tensor: str) -> str:
    if tensor == "fc.weight":
        return "fusion.operand"
    if tensor == "enc.output_norm.weight":
        return "encoder_norm"
    if tensor == "output_norm.weight":
        return "output_norm"
    _, index, part, _ = tensor.split(".")
    modules = {
        "attn_q": "query.operand",
        "attn_k": "key.operand",
        "attn_v": "value.operand",
        "attn_output": "output.operand",
        "ffn_gate": "gate.operand",
        "ffn_up": "up.operand",
        "ffn_down": "down.operand",
        "attn_norm": "attn_norm",
        "attn_q_norm": "q_norm",
        "attn_k_norm": "k_norm",
        "ffn_norm": "ffn_norm",
    }
    return f"blocks.{index}.{modules[part]}"


def field(value: object, *keys: str) -> object:
    for key in keys:
        value = cast(dict[str, object], value)[key]
    return value


class TestRoundSimulation(unittest.TestCase):
    def test_hand_built_logits_drive_rounds_and_recall(self) -> None:
        vocabulary = 8
        labels = {
            anchor: tuple((anchor + 1 + i) % vocabulary for i in range(3))
            for anchor in range(3, 10)
        }
        logits: dict[int, torch.Tensor] = {}
        for anchor, truth in labels.items():
            scores = torch.zeros((3, vocabulary))
            for slot, token in enumerate(truth):
                wrong = anchor == 5 and slot == 1
                scores[slot, token] = 1.0 if wrong else 2.0
                scores[slot, (token + 3) % vocabulary] = 1.5 if wrong else 0.5
            logits[anchor] = scores
        ranked = {
            anchor: tuple(
                tuple(int(v) for v in row) for row in value.topk(2, dim=-1).indices.tolist()
            )
            for anchor, value in logits.items()
        }
        rounds = simulate(labels, ranked, 3)
        self.assertEqual([(r.anchor, r.accepted) for r in rounds], [(4, 3), (8, 3)])
        record = tally(generated_anchors(labels), labels, ranked, 3, 2).record()
        self.assertEqual(record["first_choice_recall"], [1.0, 5 / 6, 1.0])
        self.assertEqual(record["top_recall"], [1.0, 1.0, 1.0])

    def test_proposals_match_single_window_forward(self) -> None:
        model, features = tiny_model()
        tokens = tuple(range(20))
        capture = Capture(tokens, features, ())
        anchors = (3, 4, 9, 15, 19)
        ranked = propose(model, capture, anchors, 3, 4, 2)
        with torch.no_grad():
            for anchor in anchors:
                single = model.forward(features, tokens, anchor, 4)[1:]
                self.assertEqual([slot[0] for slot in ranked[anchor]], single.argmax(-1).tolist())
                self.assertEqual(len(ranked[anchor][0]), 4)


class TestProbesAndAblations(unittest.TestCase):
    def test_range_statistics_match_direct_inputs(self) -> None:
        model, features = tiny_model()
        trained_like(model)
        seen: dict[tuple[int, str], torch.Tensor] = {}

        class Recorder(nn.Module):
            def __init__(self, original: Projection, key: tuple[int, str]) -> None:
                super().__init__()
                self.original = original
                self.key = key

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                seen[self.key] = value.detach().clone()
                return self.original.forward(value)

        tokens = tuple(range(20))
        with torch.no_grad():
            with substituted(
                model, lambda index, part, projection: Recorder(projection, (index, part))
            ):
                expected_logits = model.forward(features, tokens, 9, 4)
            probe = RangeProbe(model, 4)
            probe.window = Window("one", 9, 1)
            with probe.attached():
                logits = model.forward(features, tokens, 9, 4)
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertTrue(
            all(
                isinstance(getattr(model.blocks[0], part), Projection)
                for part in ("gate", "up", "down")
            )
        )
        record = probe.record()
        for index in range(6):
            block = cast(Layer, model.blocks[index])
            for part, kind in (("gate", "gate_up"), ("down", "down")):
                projection = getattr(block, part)
                ratio = (
                    (seen[(index, part)] * projection.log_gain.exp()).abs()
                    / projection.log_clip.exp()
                ).detach()
                measured = cast(dict[str, object], field(record, f"layer-{index}", kind))
                clipped = int((ratio > 1).sum())
                self.assertEqual(measured["clipped"], clipped)
                self.assertEqual(measured["values"], ratio.numel())
                slots = cast(list[dict[str, object]], measured["slots"])
                for slot in range(4):
                    self.assertAlmostEqual(
                        cast(float, slots[slot]["max_over_bound"]),
                        float(ratio[slot].max()),
                        places=5,
                    )
                    self.assertEqual(
                        slots[slot]["halves_clipped_values"], [0, int((ratio[slot] > 1).sum())]
                    )
                self.assertEqual(
                    field(measured, "clipped_by_context_band", "<512", "values"), ratio.numel()
                )
                if clipped:
                    self.assertAlmostEqual(
                        cast(float, measured["clipped_share_anchor_slot"]),
                        int((ratio[0] > 1).sum()) / clipped,
                    )
                self.assertEqual(len(cast(list[object], measured["largest_spikes"])), 4)
        self.assertTrue(any(field(record, f"layer-{i}", "gate_up", "clipped") for i in range(6)))

    def test_ablation_switches_leave_the_loaded_state_unchanged(self) -> None:
        model, features = tiny_model()
        trained_like(model)
        tokens = tuple(range(20))
        before = snapshot(model)
        versions = {name: value._version for name, value in model.state_dict().items()}
        with torch.no_grad():
            baseline = model.forward(features, tokens, 9, 4)
            block = cast(Layer, model.blocks[0])
            value = torch.randn((4, 4), generator=torch.Generator().manual_seed(3))
            self.assertTrue(
                torch.equal(
                    AblatedProjection(block.gate, Ablation()).forward(value),
                    block.gate.forward(value),
                )
            )
            outputs = {}
            for name in ABLATIONS:
                with ablated(model, name):
                    outputs[name] = model.forward(features, tokens, 9, 4)
        self.assertTrue(torch.equal(outputs["unchanged"], baseline))
        for name in ABLATIONS:
            if name != "unchanged":
                self.assertFalse(torch.equal(outputs[name], baseline), name)
        after = snapshot(model)
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertEqual(
            versions, {name: value._version for name, value in model.state_dict().items()}
        )
        for module in model.blocks:
            for part in ("gate", "up", "down"):
                self.assertIsInstance(getattr(module, part), Projection)
        with self.assertRaisesRegex(ValueError, "unknown"):
            with ablated(model, "missing"):
                pass
        with self.assertRaisesRegex(ValueError, "exported"):
            apply_ablation(model, "inputs-unrounded")

    def test_exported_ablations_carry_unit_gains_and_zero_midpoints(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-ablation-") as directory:
            root = Path(directory)
            cfg = Geometry(hidden=256, intermediate=256, heads=32, kv_heads=8, head_dim=8, layers=1)
            source, vocabulary = gguf_fixture(root, cfg, seed=4)
            corpus = Corpus(fixture_corpus(root, 256), 17, 64 * 1024 * 1024, 256)
            model = trained_like(load_he4(source, vocabulary, 64 * 1024 * 1024, cfg))
            parent = TensorWorker(model, corpus, "parent").checkpoint(
                Store(root / "states"), 64 * 1024 * 1024
            )
            for name, gains, midpoints in (
                ("gains-one", True, False),
                ("midpoints-zero", False, True),
            ):
                worker = TensorWorker(model, corpus, "variant-" + name)
                worker.restore(parent, 64 * 1024 * 1024, "branch")
                apply_ablation(model, name)
                exported = export_he4(
                    worker, source, vocabulary, Store(root / name), 8 << 20, 32 << 20
                )
                tensors = {t.name: t for t in gguf.GGUFReader(str(exported.artifact.path)).tensors}
                gain = np.asarray(tensors["blk.0.ffn_gate.w1ax_gain"].data)
                midpoint = np.asarray(tensors["blk.0.ffn_up.w1ax_midpoint"].data)
                self.assertEqual(bool((gain == 1).all()), gains)
                self.assertEqual(bool((midpoint == 0).all()), midpoints)
            restored = TensorWorker(model, corpus, "again")
            restored.restore(parent, 64 * 1024 * 1024, "branch")
            block = cast(Layer, model.blocks[0])
            self.assertFalse(bool((block.gate.log_gain == 0).all()))
            self.assertFalse(bool((block.up.midpoint == 0).all()))

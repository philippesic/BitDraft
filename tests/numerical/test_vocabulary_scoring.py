import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch

from tests.numerical.test_memory import GEOMETRY, specification, write
from tests.numerical.test_model import fixture_corpus, tiny_model
from w1a1_eagle.drafting import generated_anchors
from w1a1_eagle.numerical import round_diagnostics
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.labels import LabelFrequencies
from w1a1_eagle.numerical.memory_plan import Shapes, plan
from w1a1_eagle.numerical.round_diagnostics import propose
from w1a1_eagle.numerical.vocabulary_scoring import (
    Scoring,
    ScoringSettings,
    Tables,
    activation_codes,
    converted,
    export_host_bytes,
    rescored,
    restricted,
    scoring_bytes,
    scoring_settings,
)
from w1a1_eagle.records import read, reference

GIB = 1024**3


class TestScoringArithmetic(unittest.TestCase):
    def test_restricted_choice_keeps_the_full_choice_inside_the_set(self) -> None:
        logits = torch.tensor([[0.0, 3.0, 2.0, 1.0], [5.0, 5.0, 1.0, 0.0]])
        full = torch.tensor([1, 1])
        allowed = torch.tensor([[True, False, True, True], [True, True, False, False]])
        self.assertEqual(restricted(logits, allowed, full).tolist(), [2, 1])
        candidates = torch.tensor([[3, 2], [0, 1]])
        self.assertEqual(rescored(logits, candidates, full).tolist(), [2, 1])

    def test_one_bit_scores_follow_the_runtime_integer_pass(self) -> None:
        generator = torch.Generator().manual_seed(5)
        table = torch.randn((40, 64), generator=generator)
        hidden = torch.randn((3, 64), generator=generator)
        tables = Tables(table, ScoringSettings(one_bit=(4,)), None)
        codes = activation_codes(hidden)
        self.assertTrue(bool((codes.abs() <= 127).all()))
        self.assertTrue(bool((codes.abs().amax(-1) == 127).all()))
        signs = np.where(table.numpy() >= 0, 1, -1).astype(np.int64)
        integer = codes.numpy().astype(np.int64) @ signs.T
        expected = torch.tensor(integer, dtype=torch.float32) * table.abs().mean(-1)
        self.assertTrue(torch.equal(tables.one_bit_scores(hidden), expected))

    def test_full_rank_factor_reproduces_scores(self) -> None:
        generator = torch.Generator().manual_seed(6)
        table = torch.randn((50, 16), generator=generator)
        hidden = torch.randn((4, 16), generator=generator)
        tables = Tables(table, ScoringSettings(rank=16, low_rank=(4,)), None)
        self.assertTrue(
            torch.allclose(tables.low_rank_scores(hidden), hidden @ table.T, atol=2e-2, rtol=2e-2)
        )
        top = Tables(table, ScoringSettings(rank=1, low_rank=(4,)), None)
        self.assertEqual(tuple(top.left.shape), (50, 1))

    def test_low_bit_tables_use_their_levels_and_beat_one_bit_error(self) -> None:
        generator = torch.Generator().manual_seed(7)
        table = torch.randn((30, 64), generator=generator)
        tables = Tables(table, ScoringSettings(tables=("2-bit", "1-bit")), None)
        two = converted(table, "2-bit", tables.one_bit_scale, tables.two_bit_scale)
        one = converted(table, "1-bit", tables.one_bit_scale, tables.two_bit_scale)
        levels = torch.round(two / tables.two_bit_scale[:, None] * 2) / 2
        self.assertEqual(set(levels.unique().tolist()), {-1.5, -0.5, 0.5, 1.5})
        self.assertEqual(set((one / tables.one_bit_scale[:, None]).unique().tolist()), {-1.0, 1.0})
        self.assertLess(float((two - table).square().sum()), float((one - table).square().sum()))
        hidden = torch.randn((5, 64), generator=generator)
        chosen = tables.converted_argmax(hidden, "1-bit")
        self.assertEqual(chosen.tolist(), (hidden @ one.T).argmax(-1).tolist())

    def test_settings_refuse_incomplete_or_unknown_values(self) -> None:
        root = Path("/")
        cases: list[dict[str, object]] = [
            {},
            {"frequent": [8]},
            {"request_tokens": True},
            {"tables": ["3-bit"]},
            {"low_rank": {"rank": 0, "keep": [4]}},
            {"one_bit": [4, 4]},
            {"one_bit": [4], "export": {"rounds_per_request": 1}},
            {"one_bit": [4], "other": 1},
        ]
        for raw in cases:
            with self.assertRaises(ValueError, msg=str(raw)):
                scoring_settings(raw, root, 16)
        parsed = scoring_settings({"one_bit": [64, 4], "tables": ["1-bit"]}, root, 16)
        self.assertEqual(
            parsed.variants(),
            ("one-bit-4", "one-bit-64", "table-1-bit-scoring", "table-1-bit-drafter"),
        )


class TestScoringMemory(unittest.TestCase):
    def test_scoring_adds_only_to_the_round_steps(self) -> None:
        base = Shapes(GEOMETRY, 20, 30, 64, 32, 4, 3, 1, True)
        scored = Shapes(GEOMETRY, 20, 30, 64, 32, 4, 3, 1, True, 0, 1000, 70)
        for cuda in (True, False):
            before = {step.name: step for step in plan(base, cuda)}
            after = {step.name: step for step in plan(scored, cuda)}
            for name, step in before.items():
                rounds = name.endswith("rounds")
                self.assertEqual(
                    after[name].device - step.device, 1000 if rounds and cuda else 0, name
                )
                self.assertEqual(
                    after[name].host - step.host,
                    (70 + (0 if cuda else 1000)) if rounds else 0,
                    name,
                )
        settings = ScoringSettings(one_bit=(4,), rank=256, low_rank=(4,), export_rounds=1)
        self.assertGreater(scoring_bytes(settings, 248320, 2560, 96), 248320 * 256 * 4)
        self.assertEqual(export_host_bytes(ScoringSettings(one_bit=(4,)), 248320, 2560), 0)


class TestScoringHook(unittest.TestCase):
    def test_hook_leaves_proposals_and_table_unchanged(self) -> None:
        model, features = tiny_model()
        tokens = (3, 5, 7, 9, 11, 3, 5, 2, 8, 1, 4, 6, 10, 12, 14, 0, 16, 18, 13, 15)
        windows = {anchor: tokens[anchor + 1 : anchor + 4] for anchor in range(8, 19)}
        capture = Capture(tokens, features, ())
        anchors = generated_anchors(windows)
        before = model.vocabulary.clone()
        plain = propose(model, capture, anchors, 3, 4, 3)
        ranked = (1, 5, 7) + tuple(token for token in range(20) if token not in {1, 5, 7})
        frequencies = LabelFrequencies("m", ranked, (20,) * 20, 400)
        scoring = Scoring(
            ScoringSettings(
                frequent=(1, 20),
                request_tokens=True,
                one_bit=(2, 20),
                rank=2,
                low_rank=(20,),
                tables=("2-bit", "1-bit"),
                export_rounds=1,
            ),
            frequencies,
            model,
            3,
        )
        hook = scoring.request(capture, windows)
        hooked = propose(model, capture, anchors, 3, 4, 3, hook)
        self.assertEqual(plain, hooked)
        self.assertTrue(torch.equal(model.vocabulary, before))
        first = {anchor: tuple(slot[0] for slot in plain[anchor]) for anchor in anchors}
        for name in ("frequent-20", "one-bit-20", "low-rank-2-20"):
            self.assertEqual(hook.choices[name].first, first, name)
        self.assertTrue(all(set(row) == {1} for row in hook.choices["frequent-1"].first.values()))
        widened = hook.choices["frequent-1+request"]
        for anchor, row in widened.first.items():
            for token in row:
                self.assertTrue(token == 1 or token in tokens[: anchor + 1])
        for anchor in anchors:
            labels = windows[anchor]
            self.assertEqual(
                hook.choices["frequent-1"].label_inside[anchor][: len(labels)],
                tuple(label == 1 for label in labels),
            )
        self.assertEqual(set(hook.hidden), set(anchors))


class TestScoringReport(unittest.TestCase):
    def setUp(self) -> None:
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.addCleanup(quiet.close)

    def test_rounds_without_scoring_are_unchanged_and_exports_are_complete(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-scoring-") as directory:
            root = Path(directory)
            spec = specification(root, 4 * GIB, True)
            train = root / "labels"
            train.mkdir()
            corpus = read(fixture_corpus(train, 256).path)
            rows = corpus["records"]
            assert isinstance(rows, list)
            corpus["records"] = [{**row, "role": "train"} for row in rows]
            corpus["target_sha256"] = "t"
            source = write(train / "train-corpus.json", corpus)
            mixture = write(
                root / "mixture.json",
                {
                    "schema": "corpus-mixture-v1",
                    "target_sha256": "t",
                    "components": [{"kind": "corpus-v1", "source": reference(source)}],
                },
            )
            self.assertEqual(round_diagnostics.run_diagnostics(spec, root / "plain", GEOMETRY), 0)
            raw = json.loads(spec.read_text())
            raw["rounds"]["scoring"] = {
                "mixture": reference(mixture),
                "frequent": [1, 64],
                "request_tokens": True,
                "one_bit": [2, 8],
                "low_rank": {"rank": 8, "keep": [2, 8]},
                "tables": ["2-bit", "1-bit"],
                "export": {"rounds_per_request": 2},
            }
            scored_spec = write(root / "scored.json", raw).path
            self.assertEqual(
                round_diagnostics.run_diagnostics(scored_spec, root / "scored", GEOMETRY), 0
            )
            plain = json.loads((root / "plain" / "report.json").read_text())["rounds"]
            scored = json.loads((root / "scored" / "report.json").read_text())["rounds"]
            for drafter in ("candidate", "baseline"):
                section = scored[drafter]
                scoring = section.pop("scoring")
                self.assertEqual(section, plain[drafter])
                variants = scoring["variants"]
                self.assertEqual(variants["frequent-64"]["accepted_per_round_difference"], 0.0)
                self.assertEqual(variants["frequent-64"]["first_choice_changed"][0], 0.0)
                self.assertEqual(variants["frequent-1"]["label_outside_candidates"][0], 0.0)
                self.assertIsNone(variants["table-1-bit-drafter"]["label_outside_candidates"])
                self.assertEqual(scoring["train_labels"]["distinct_tokens"], 2)
                self.assertEqual(scoring["train_labels"]["coverage"]["1"], 15 / 16)
                export = root / "scored" / ("scoring-" + drafter)
                manifest = json.loads((export / "manifest.json").read_text())
                self.assertEqual(manifest["table"]["tensor"], "token_embd.weight")
                sets = len(manifest["sets"])
                self.assertGreater(sets, 0)
                sizes = {
                    "hidden.f32": sets * 3 * 256 * 4,
                    "signs.u32": 20 * 8 * 4,
                    "scales.f32": 20 * 4,
                    "low-rank-left.f16": 20 * 8 * 2,
                    "low-rank-right.f16": 8 * 256 * 2,
                    "frequent.i32": 2 * 4,
                }
                for name, size in sizes.items():
                    self.assertEqual(manifest["files"][name]["bytes"], size, name)
                    self.assertEqual((export / name).stat().st_size, size, name)
                words = np.frombuffer((export / "signs.u32").read_bytes(), dtype="<u4")
                bits = np.unpackbits(words.view(np.uint8), bitorder="little").reshape(20, 256)
                table = next(
                    item
                    for item in gguf.GGUFReader(manifest["table"]["model"]["path"]).tensors
                    if item.name == "token_embd.weight"
                )
                values = gguf.dequantize(table.data, table.tensor_type).reshape(20, 256)
                self.assertTrue(np.array_equal(bits.astype(bool), values >= 0))
                scales = np.frombuffer((export / "scales.f32").read_bytes(), dtype="<f4")
                self.assertTrue(np.allclose(scales, np.abs(values).mean(-1), rtol=1e-5))
                ranked = np.frombuffer((export / "frequent.i32").read_bytes(), dtype="<i4")
                self.assertEqual(ranked.tolist(), [1, 19])
            text = (root / "scored" / "report.txt").read_text()
            self.assertIn("scoring candidate frequent-1:", text)

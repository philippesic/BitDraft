import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import torch

from tests.numerical.test_features import SAMPLE, gradual_record
from tests.numerical.test_model import (
    LOADER_GEOMETRY,
    OMITTED_RECIPE,
    execute_leased,
    gguf_fixture,
    loader_fixture,
    row_optimization,
    scenario_intent,
    table_optimization,
)
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, record, sequence
from w1a1_eagle.data import Artifact
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.model import HE4, named_projections
from w1a1_eagle.numerical.calibration import BOUND_CANDIDATES, BOUND_DECADES, calibrate
from w1a1_eagle.numerical.captures import load_capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.fusing import fuse
from w1a1_eagle.numerical.mixture import mixture
from w1a1_eagle.numerical.weights import fusion_identity, load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.training import Budget, Features, Grouping, Sample

BUDGET = 64 * 1024 * 1024
HIDDEN = LOADER_GEOMETRY.hidden
TAPS = (2, 6, 10, 14, 18, 22, 26, 30)


def authenticated(path: Path) -> Artifact:
    return Artifact(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)


def eight_tap_corpus(
    root: Path, prefix: str, lengths: tuple[int, ...], seed: int, role: str = "train"
) -> Artifact:
    root.mkdir(parents=True)
    generator = np.random.default_rng(seed)
    records: list[Record] = []
    for index, length in enumerate(lengths):
        folder = root / f"{prefix}-{index}"
        folder.mkdir()
        half = index % 2 == 1
        suffix = "f16" if half else "f32"
        files = []
        for tap in TAPS:
            path = folder / f"layer-{tap}.{suffix}"
            values = generator.standard_normal((length, HIDDEN)).astype("<f2" if half else "<f4")
            path.write_bytes(values.tobytes())
            files.append(reference(authenticated(path)))
        support = folder / "source.json"
        support.write_bytes(canonical({"id": f"{prefix}-{index}"}))
        files.append(reference(authenticated(support)))
        fields: Record = {
            "taps": list(TAPS),
            "feature_semantics": "zero_based_pre_block_input_all_tokens",
            "input_tokens": [(seed + index + p) % 18 for p in range(length)],
            "prompt_tokens": 8,
            "logit_positions": list(range(7, length)),
            "greedy_ids": [(seed + index + p) % 18 for p in range(8, length)] + [19],
            "eog_ids": [19],
            "window_policy": "up-to-15",
        }
        if half:
            fields["feature_dtype"] = "float16"
        metadata = folder / "capture.json"
        metadata.write_bytes(canonical(fields))
        records.append(
            {
                "id": f"{prefix}-{index}",
                "source": "synthetic-fixture",
                "group": f"{prefix}-group-{index}",
                "topic": None,
                "aliases": [],
                "role": role,
                "ancestry": "verified",
                "prior_use": "unobserved",
                "metadata": reference(authenticated(metadata)),
                "files": files,
            }
        )
    manifest = root / "corpus.json"
    manifest.write_bytes(
        canonical(
            {
                "schema": 1,
                "target_sha256": "a" * 64,
                "protected": ["heldout24"],
                "expected_train": len(lengths) if role == "train" else 0,
                "records": records,
            }
        )
    )
    return authenticated(manifest)


def specification(
    root: Path,
    name: str,
    corpus: Artifact,
    source: Artifact,
    vocabulary: Artifact,
    **changes: object,
) -> Path:
    raw: Record = {
        "schema": "fused-conversion-v1",
        "corpus": reference(corpus),
        "corpus_format": "corpus-v1",
        "source": reference(source),
        "vocabulary": reference(vocabulary),
        "frozen_format": "Q4_0",
        "fused_dtype": "float16",
        "maximum_relative_l2": 0.01,
        "capture_limit_bytes": 64 * 1024 * 1024,
        "memory_bytes": BUDGET,
        "device": "cpu",
        "rows": 7,
        "geometry": {
            "hidden": 256,
            "intermediate": 256,
            "heads": 4,
            "kv_heads": 1,
            "head_dim": 8,
            "layers": 1,
        },
        "disk_reserve_bytes": 0,
        **changes,
    }
    path = root / f"{name}.json"
    path.write_bytes(canonical(raw))
    return path


def model(source: Artifact, vocabulary: Artifact) -> HE4:
    return load_he4(source, vocabulary, BUDGET, LOADER_GEOMETRY)


def train(worker: TensorWorker, updates: int) -> list[float]:
    losses = []
    for target in range(1, updates + 1):
        worker.run(Monotonic(), Budget(target, 60, 1000), lambda: False)
        losses.append(cast(float, worker.progress.last_loss))
    return losses


def corpus(manifest: Artifact, fused: bool = False) -> Corpus:
    return Corpus(
        manifest, 3, BUDGET, HIDDEN, grouping=Grouping(3), shape=LOADER_GEOMETRY, fused=fused
    )


class TestFusedStorage(unittest.TestCase):
    def test_round_trip_identity_and_refusals(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, LOADER_GEOMETRY, 5)
            original = eight_tap_corpus(root / "original", "row", (21, 33, 18), 1)
            before = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (root / "original").rglob("*")
                if path.is_file()
            }
            spec = specification(root, "fuse", original, source, vocabulary)
            heights: list[int] = []
            product = torch.nn.functional.linear

            def bounded(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
                heights.append(value.shape[0])
                return product(value, weight)

            with mock.patch.object(torch.nn.functional, "linear", bounded):
                result = fuse(spec, root / "fused")
            self.assertEqual(max(heights), 7)
            self.assertEqual(sum(heights), 21 + 33 + 18)
            self.assertEqual((result["status"], result["arrays"]), ("converted", 3))
            self.assertFalse(result["source_deleted"])
            self.assertLess(cast(float, result["maximum_relative_l2"]), 2.0**-10)
            self.assertEqual(result["feature_bytes"], (21 + 33 + 18) * HIDDEN * 2)
            after = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (root / "original").rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)
            operand = model(source, vocabulary).fusion.operand
            identity = fusion_identity(operand)
            self.assertEqual(record(result["fusion"])["sha256"], identity)
            fused_ref = artifact(result["corpus"], root)
            plain = corpus(original)
            fused = corpus(fused_ref, True)
            self.assertEqual(fused.fusion, identity)
            self.assertEqual(fused.count, plain.count)
            for plain_row, fused_row in zip(plain.rows, fused.rows, strict=True):
                self.assertEqual(plain_row.anchors, fused_row.anchors)
                eight = load_capture(plain_row.capture, BUDGET, HIDDEN)
                stored = load_capture(fused_row.capture, BUDGET, HIDDEN, fusion=identity)
                self.assertEqual(stored.features.dtype, torch.float16)
                self.assertEqual(stored.features.shape, (len(eight.tokens), HIDDEN))
                self.assertEqual((stored.tokens, stored.windows), (eight.tokens, eight.windows))
                exact = torch.nn.functional.linear(eight.features.float(), operand)
                difference = stored.features.float() - exact
                self.assertLess(float(difference.norm() / exact.norm()), 2.0**-11)
                self.assertLess(float(difference.abs().max()), float(exact.abs().max()) * 2.0**-10)
                with self.assertRaisesRegex(ValueError, "fusion table identity"):
                    load_capture(fused_row.capture, BUDGET, HIDDEN, fusion="0" * 64)
                with self.assertRaisesRegex(ValueError, "semantics"):
                    load_capture(fused_row.capture, BUDGET, HIDDEN)
                with self.assertRaisesRegex(ValueError, "fusion table identity"):
                    load_capture(plain_row.capture, BUDGET, HIDDEN, fusion=identity)
            with self.assertRaisesRegex(ValueError, "unsupported corpus fields"):
                corpus(fused_ref)
            with self.assertRaisesRegex(ValueError, "fused-corpus-v1"):
                corpus(original, True)
            with self.assertRaisesRegex(ValueError, "new root"):
                fuse(spec, root / "fused")
            small = specification(
                root, "small", original, source, vocabulary, capture_limit_bytes=1024
            )
            with self.assertRaisesRegex(ValueError, "forecast exceeds the declared bound"):
                fuse(small, root / "small")
            self.assertFalse((root / "small").exists())
            tight = specification(
                root, "tight", original, source, vocabulary, maximum_relative_l2=1e-9
            )
            with self.assertRaisesRegex(ValueError, "numerical margin"):
                fuse(tight, root / "tight")
            for broken in ({"device": "tpu"}, {"fused_dtype": "int8"}, {"rows": 0}):
                path = specification(root, "broken", original, source, vocabulary, **broken)
                with self.assertRaises(ValueError):
                    fuse(path, root / "broken")
            altered = root / "original" / "row-1" / "layer-2.f16"
            data = bytearray(altered.read_bytes())
            data[-1] ^= 1
            altered.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "content mismatch"):
                fuse(spec, root / "altered")

    def test_fused_training_matches_eight_tap_training(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, LOADER_GEOMETRY, 5)
            original = eight_tap_corpus(root / "original", "row", (24, 40, 32), 2)
            exact = artifact(
                fuse(
                    specification(
                        root, "exact", original, source, vocabulary, fused_dtype="float32"
                    ),
                    root / "exact",
                )["corpus"],
                root,
            )
            rounded = artifact(
                fuse(specification(root, "half", original, source, vocabulary), root / "half")[
                    "corpus"
                ],
                root,
            )
            reference_worker = TensorWorker(model(source, vocabulary), corpus(original), "recipe")
            expected = train(reference_worker, 6)
            results = {}
            for name, manifest in (("exact", exact), ("half", rounded)):
                worker = TensorWorker(model(source, vocabulary), corpus(manifest, True), "recipe")
                worker.fusion_sha256 = fusion_identity(worker.model.fusion.operand)
                results[name] = (worker, train(worker, 6))
            exact_worker, exact_losses = results["exact"]
            np.testing.assert_allclose(exact_losses, expected, rtol=1e-6)
            for (name, value), (_, other) in zip(
                exact_worker.model.named_parameters(),
                reference_worker.model.named_parameters(),
                strict=True,
            ):
                torch.testing.assert_close(value, other, rtol=1e-5, atol=1e-6, msg=name)
            half_losses = results["half"][1]
            half_gap = max(abs(a - b) / b for a, b in zip(half_losses, expected, strict=True))
            exact_gap = max(abs(a - b) / b for a, b in zip(exact_losses, expected, strict=True))
            self.assertLess(half_gap, 1e-3)
            self.assertGreater(half_gap, exact_gap)
            plain_model = model(source, vocabulary)
            capture, windows = corpus(original).sample(0)
            anchors = tuple(window.anchor for window in windows)
            last = max(anchors)
            fused_values = plain_model.fusion.forward(capture.features[:last].float())
            padded = torch.cat(
                (fused_values, fused_values.new_zeros(len(capture.tokens) - last, HIDDEN))
            )
            with torch.no_grad():
                self.assertTrue(
                    torch.equal(
                        plain_model.group_hidden(capture.features, capture.tokens, anchors, 4),
                        plain_model.group_hidden(padded, capture.tokens, anchors, 4, True),
                    )
                )
            stranger = TensorWorker(model(source, vocabulary), corpus(rounded, True), "recipe")
            with self.assertRaisesRegex(ValueError, "fusion table identities differ"):
                train(stranger, 1)
            tables = load_he4(source, vocabulary, BUDGET, LOADER_GEOMETRY, None, "Q4_0", None, True)
            capture, windows = corpus(rounded, True).sample(0)
            with self.assertRaisesRegex(ValueError, "frozen fusion table"):
                tables.group_hidden(capture.features, capture.tokens, (windows[0].anchor,), 4, True)

    def test_fused_mixture_and_intent_formats(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, LOADER_GEOMETRY, 5)
            components = [
                eight_tap_corpus(root / f"part-{index}", f"part{index}", (20, 26), 7 * index + 3)
                for index in range(2)
            ]
            mixed = root / "mixture.json"
            mixed.write_bytes(
                canonical(
                    {
                        "schema": "corpus-mixture-v1",
                        "target_sha256": "a" * 64,
                        "components": [
                            {"kind": "corpus-v1", "source": reference(value)}
                            for value in components
                        ],
                    }
                )
            )
            spec = specification(
                root,
                "mixture-spec",
                authenticated(mixed),
                source,
                vocabulary,
                corpus_format="corpus-mixture-v1",
            )
            result = fuse(spec, root / "fused-mixture")
            self.assertEqual(result["corpus_format"], "fused-mixture-v1")
            manifest = artifact(result["corpus"], root)
            identity = fusion_identity(model(source, vocabulary).fusion.operand)
            dataset = mixture(manifest, 3, BUDGET, HIDDEN, True)
            self.assertEqual((dataset.fusion, len(dataset.rows)), (identity, 4))
            plain = mixture(authenticated(mixed), 3, BUDGET, HIDDEN)
            self.assertIsNone(plain.fusion)
            loaded = Corpus(dataset, 3, BUDGET, HIDDEN, shape=LOADER_GEOMETRY, fused=True)
            self.assertEqual(loaded.count, Corpus(plain, 3, BUDGET, HIDDEN).count)
            with self.assertRaisesRegex(ValueError, "feature format"):
                Corpus(dataset, 3, BUDGET, HIDDEN, shape=LOADER_GEOMETRY)
            with self.assertRaisesRegex(ValueError, "unsupported corpus mixture"):
                mixture(manifest, 3, BUDGET, HIDDEN)
            with self.assertRaisesRegex(ValueError, "fused-mixture-v1"):
                mixture(authenticated(mixed), 3, BUDGET, HIDDEN, True)
            (root / "intent-fixture").mkdir()
            loaded_fixture = loader_fixture(root / "intent-fixture")
            fixed = {
                key: {
                    "path": str(record(loaded_fixture.intent[key])["path"]),
                    "sha256": str(i) * 64,
                    "bytes": 7,
                }
                for i, key in enumerate(("source", "vocabulary", "corpus", "initialization"), 1)
            }
            base: Record = {**loaded_fixture.intent, **fixed, "code_sha256": "0" * 64}
            path = root / "intent.json"

            def parse(changes: Record) -> Intent:
                path.write_bytes(canonical({**base, **changes}))
                return Intent.load(path)

            default = parse({})
            self.assertEqual(default.recipe, OMITTED_RECIPE)
            self.assertFalse(default.fused_input or default.mixture_input)
            recipes = {OMITTED_RECIPE}
            for value, fused, mixed_input in (
                ("corpus-mixture-v1", False, True),
                ("fused-corpus-v1", True, False),
                ("fused-mixture-v1", True, True),
            ):
                parsed = parse({"corpus_format": value})
                self.assertEqual((parsed.fused_input, parsed.mixture_input), (fused, mixed_input))
                recipes.add(parsed.recipe)
            self.assertEqual(len(recipes), 4)
            with self.assertRaisesRegex(ValueError, "frozen fusion table"):
                parse(
                    {
                        "corpus_format": "fused-corpus-v1",
                        "trained_tables": True,
                        "optimization": table_optimization(),
                    }
                )
            with self.assertRaises(ValueError):
                parse({"corpus_format": "fused-v2"})
            kinds = [record(value)["kind"] for value in sequence(read(manifest.path)["components"])]
            self.assertEqual(kinds, ["fused-corpus-v1", "fused-corpus-v1"])


class TestFusedRun(unittest.TestCase):
    def test_fused_intent_trains_and_refuses_other_fusion_tables(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-fused-") as directory:
            root = Path(directory)
            loaded, intent, path = scenario_intent(root)
            fused_ref = artifact(
                fuse(
                    specification(
                        root,
                        "fuse",
                        loaded.corpus,
                        loaded.source,
                        loaded.vocabulary,
                        initialization=reference(loaded.initialization),
                    ),
                    root / "fused",
                )["corpus"],
                root,
            )
            intent.update(corpus=reference(fused_ref), corpus_format="fused-corpus-v1")
            path.write_bytes(canonical(intent))
            admitted = execute_leased(path, root / "run")
            admission_row = record(Journal(root / "run").report()["admission"])
            self.assertIsNotNone(admission_row)
            state = Store(root / "run" / "states").latest(admitted.recipe)
            assert state is not None
            self.assertEqual(state.updates, 1)
            (root / "other").mkdir()
            other_source, other_vocabulary = gguf_fixture(root / "other", LOADER_GEOMETRY, 9)
            other = artifact(
                fuse(
                    specification(root, "other", loaded.corpus, other_source, other_vocabulary),
                    root / "other-fused",
                )["corpus"],
                root,
            )
            intent.update(corpus=reference(other))
            path.write_bytes(canonical(intent))
            with self.assertRaisesRegex(ValueError, "fusion table identities differ"):
                execute_leased(path, root / "refused")


def calibrated_values(model: HE4) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{part}": getattr(projection, part).detach().clone()
        for name, projection in named_projections(model)
        for part in ("scale", "midpoint", "log_clip")
    }


def largest_relative(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(
        float((left[key] - right[key]).abs().max() / right[key].abs().max().clamp_min(1e-30))
        for key in right
    )


def development(root: Path) -> Artifact:
    dev = eight_tap_corpus(root / "dev", "dev", (48, 48, 48), 40, "dev")
    exclusions = root / "exclusions.json"
    exclusions.write_bytes(canonical({"identities": []}))
    windows = [
        {"id": f"dev-{index}", "anchor": anchor} for index in range(3) for anchor in range(7, 47)
    ]
    path = root / "diagnostics.json"
    path.write_bytes(
        canonical(
            {
                "schema": "training-diagnostics-v1",
                "corpus": reference(dev),
                "exclusions": reference(authenticated(exclusions)),
                "windows": windows,
            }
        )
    )
    return authenticated(path)


class TestFusedCalibration(unittest.TestCase):
    def test_calibration_on_fused_input_matches_eight_tap_input(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-fused-calibration-") as directory:
            root = Path(directory)
            source, vocabulary = gguf_fixture(root, LOADER_GEOMETRY, 5)
            original = eight_tap_corpus(root / "original", "row", (24, 40, 32), 2)
            manifests = {
                dtype: artifact(
                    fuse(
                        specification(root, dtype, original, source, vocabulary, fused_dtype=dtype),
                        root / dtype,
                    )["corpus"],
                    root,
                )
                for dtype in ("float32", "float16")
            }
            sample = Sample(3, 6, 5)

            def run(manifest: Artifact, fused: bool) -> tuple[Record, dict[str, torch.Tensor]]:
                target = model(source, vocabulary)
                result = calibrate(
                    target, corpus(manifest, fused), sample, Features(), 4, BUDGET, False
                )
                return result, calibrated_values(target)

            plain, expected = run(original, False)
            exact, exact_values = run(manifests["float32"], True)
            rounded, rounded_values = run(manifests["float16"], True)
            for result in (exact, rounded):
                self.assertEqual(
                    [record(row)["anchors"] for row in sequence(record(result["sample"])["rows"])],
                    [record(row)["anchors"] for row in sequence(record(plain["sample"])["rows"])],
                )
            self.assertLess(largest_relative(exact_values, expected), 1e-5)
            step = math.pow(10.0, BOUND_DECADES / BOUND_CANDIDATES)
            for key, value in expected.items():
                if not key.endswith("log_clip"):
                    continue
                ratio = float((rounded_values[key] - value).exp())
                self.assertLess(abs(math.log(ratio)), math.log(step) * 1.05, key)
                prefix = key.removesuffix("log_clip")
                gap = largest_relative(
                    {part: rounded_values[prefix + part] for part in ("scale", "midpoint")},
                    {part: expected[prefix + part] for part in ("scale", "midpoint")},
                )
                self.assertLess(gap, 1e-2 if abs(ratio - 1.0) < 1e-3 else 5e-2, key)

    def test_calibrated_fresh_intent_trains_on_a_fused_corpus(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-fused-entry-") as directory:
            root = Path(directory)
            (root / "fixture").mkdir()
            loaded = loader_fixture(root / "fixture")
            source, vocabulary = gguf_fixture(root, LOADER_GEOMETRY, 5)
            original = eight_tap_corpus(root / "original", "row", (24, 40, 32), 2)
            fused = artifact(
                fuse(specification(root, "fuse", original, source, vocabulary), root / "fused")[
                    "corpus"
                ],
                root,
            )
            base = {key: value for key, value in loaded.intent.items() if key != "initialization"}
            intent: Record = {
                **base,
                "source": reference(source),
                "vocabulary": reference(vocabulary),
                "corpus": reference(fused),
                "corpus_format": "fused-corpus-v1",
                "development": reference(development(root)),
                "development_cadence_updates": 1,
                "optimization": row_optimization(),
                "budget": {"updates": 2, "wall_seconds": 120, "attempts": 3, "retries": 0},
                "science_policy": {
                    "id": "synthetic-fixture-only",
                    "min_unique_positions": 16,
                    "max_mean_visits": 100,
                },
                "calibration": {"sample": SAMPLE.encode()},
                "gradual_precision": gradual_record(1, maximum=0.5),
                "initial_export": True,
                "features": Features(clip="fixed").encode(),
            }
            path = root / "intent.json"
            path.write_bytes(canonical(intent))
            admitted = execute_leased(path, root / "run")
            events = list(Journal(root / "run").events())
            kinds = [row.get("kind") for row in events]
            for kind in (
                "calibration",
                "gradual-precision-start-check",
                "initial-export",
                "development-diagnostic",
            ):
                self.assertIn(kind, kinds)
            states = Store(root / "run" / "states").latest(admitted.recipe)
            assert states is not None
            self.assertEqual(states.updates, 2)

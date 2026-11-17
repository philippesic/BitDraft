import contextlib
import gc
import io
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from typing import Callable, Literal
from unittest import mock

import torch

from tests.numerical.test_model import authenticated, fixture_corpus, gguf_fixture
from tests.numerical.test_rounds import baseline_fixture
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.data import Artifact
from w1a1_eagle.model import HE4, Geometry
from w1a1_eagle.numerical import round_diagnostics
from w1a1_eagle.numerical.baseline import load_baseline
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.memory_plan import (
    DEVICE_MARGIN,
    FatalMemory,
    Guard,
    Shapes,
    baseline_elements,
    he4_elements,
    plan,
    refusal,
)
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import canonical, read, reference

GEOMETRY = Geometry(hidden=256, intermediate=256, heads=32, kv_heads=8, head_dim=8, layers=1)
GIB = 1024**3


def write(path: Path, value: object) -> Artifact:
    path.write_bytes(canonical(value))
    return authenticated(path)


def specification(root: Path, memory: int, candidate: bool) -> Path:
    source, vocabulary = gguf_fixture(root, GEOMETRY, seed=4)
    drafter, target = baseline_fixture(root, GEOMETRY)
    train = root / "train"
    train.mkdir()
    corpus = Corpus(fixture_corpus(train, 256), 17, 64 * 1024 * 1024, 256)
    model = load_he4(source, vocabulary, 64 * 1024 * 1024, GEOMETRY)
    state = TensorWorker(model, corpus, "parent").checkpoint(Store(root / "states"), 1 << 26)
    del model
    development = root / "dev"
    development.mkdir()
    manifest = read(fixture_corpus(development, 256).path)
    records = manifest["records"]
    assert isinstance(records, list)
    manifest["records"] = [{**row, "role": "dev"} for row in records]
    dev = write(development / "dev-corpus.json", manifest)
    exclusions = write(root / "exclusions.json", {"identities": []})
    raw: dict[str, object] = {
        "schema": "round-diagnostics-v1",
        "memory_bytes": memory,
        "baseline": {
            "drafter": reference(authenticated(drafter)),
            "target": reference(authenticated(target)),
        },
        "rounds": {
            "corpus": reference(dev),
            "exclusions": reference(exclusions),
            "depth": 3,
            "top": 4,
            "chunk": 4,
            "seed": 1,
            "resamples": 10,
        },
    }
    if candidate:
        raw["candidate"] = {
            "source": reference(source),
            "vocabulary": reference(vocabulary),
            "state": reference(state.artifact),
            "mode": "continue",
        }
    return write(root / "spec.json", raw).path


class TestMemoryPlan(unittest.TestCase):
    def test_plan_separates_models_and_refuses_beyond_device_or_host(self) -> None:
        shape = Shapes(GEOMETRY, 20, 30, 64, 32, 4, 3, 1, True)
        steps = {step.name: step for step in plan(shape, True)}
        frozen, trainable = he4_elements(GEOMETRY, 20)
        self.assertEqual(steps["candidate load"].device, (frozen + trainable) * 4)
        self.assertEqual(steps["baseline load"].device, baseline_elements(GEOMETRY, 30) * 4)
        self.assertGreater(steps["candidate restore"].device, steps["candidate load"].device)
        self.assertGreater(steps["baseline rounds"].device, steps["baseline load"].device)
        self.assertEqual(
            list(steps),
            [
                "candidate load",
                "candidate restore",
                "candidate rounds",
                "candidate windows",
                "candidate exports",
                "baseline load",
                "baseline rounds",
            ],
        )
        largest = max(step.device for step in steps.values())
        hungriest = max(step.host for step in steps.values())
        self.assertIsNone(refusal(tuple(steps.values()), largest + DEVICE_MARGIN, hungriest))
        self.assertIn(
            "GPU", refusal(tuple(steps.values()), largest + DEVICE_MARGIN - 1, hungriest) or ""
        )
        self.assertIn(
            "host", refusal(tuple(steps.values()), largest + DEVICE_MARGIN, hungriest - 1) or ""
        )
        self.assertTrue(all(step.device == 0 for step in plan(shape, False)))
        full = {
            step.name: step
            for step in plan(Shapes(Geometry(), 248320, 248320, 16384, 16384, 32, 3, 3, True), True)
        }
        self.assertLess(max(step.device for step in full.values()), 15 * GIB)

    def test_guard_stops_on_reported_out_of_memory_and_host_limit(self) -> None:
        Guard(False, 10, lambda: 0, lambda: 5).check()
        with self.assertRaisesRegex(FatalMemory, "out-of-memory"):
            Guard(False, 10, lambda: 1, lambda: 5).check()
        with self.assertRaisesRegex(FatalMemory, "host memory"):
            Guard(False, 10, lambda: 0, lambda: 11).check()


class TestDiagnosticsMemory(unittest.TestCase):
    def setUp(self) -> None:
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.addCleanup(quiet.close)

    def test_up_front_refusal_loads_nothing(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-memory-") as directory:
            root = Path(directory)
            spec = specification(root, GIB, True)
            with (
                mock.patch.object(round_diagnostics, "load_baseline") as baseline,
                mock.patch.object(round_diagnostics, "load_he4") as candidate,
            ):
                with self.assertRaisesRegex(ValueError, "memory plan refused: .* host memory"):
                    round_diagnostics.diagnose(spec, root / "out", GEOMETRY)
                baseline.assert_not_called()
                candidate.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_out_of_memory_is_fatal_and_models_never_coexist(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-memory-") as directory:
            root = Path(directory)
            spec = specification(root, 4 * GIB, True)
            candidates: list[weakref.ref[HE4]] = []
            seen: list[bool] = []

            def candidate(
                source: Artifact,
                vocabulary: Artifact,
                memory: int,
                geometry: Geometry,
                initialization: Artifact | None,
                frozen_format: Literal["Q4_0", "baseline-k"],
                frozen: Artifact | None,
            ) -> HE4:
                model = load_he4(
                    source, vocabulary, memory, geometry, initialization, frozen_format, frozen
                )
                candidates.append(weakref.ref(model))
                return model

            def baseline(
                drafter: Artifact, target: Artifact, memory: int, geometry: Geometry, device: str
            ) -> HE4:
                gc.collect()
                seen.append(all(reference_() is None for reference_ in candidates))
                return load_baseline(drafter, target, memory, geometry, device)

            with (
                mock.patch.object(round_diagnostics, "load_he4", candidate),
                mock.patch.object(round_diagnostics, "load_baseline", baseline),
            ):
                self.assertEqual(round_diagnostics.run_diagnostics(spec, root / "ok", GEOMETRY), 0)
            self.assertEqual((len(candidates), seen), (1, [True]))
            report = json.loads((root / "ok" / "report.json").read_text())
            self.assertEqual(
                [row["step"] for row in report["memory"]["measured"]],
                [
                    "candidate load",
                    "candidate restore",
                    "candidate rounds",
                    "baseline load",
                    "baseline rounds",
                ],
            )
            self.assertIn("baseline", report["rounds"])
            failure = torch.cuda.OutOfMemoryError("CUDA out of memory")
            with mock.patch.object(round_diagnostics, "propose", side_effect=failure) as propose:
                code = round_diagnostics.run_diagnostics(spec, root / "oom", GEOMETRY)
            self.assertEqual(code, round_diagnostics.FATAL_MEMORY_EXIT)
            self.assertEqual(propose.call_count, 1)
            self.assertFalse((root / "oom" / "report.json").exists())
            swallowed: Callable[[bool, int], Guard] = lambda cuda, limit: Guard(
                cuda, limit, lambda: 1
            )
            with mock.patch.object(round_diagnostics, "Guard", swallowed):
                code = round_diagnostics.run_diagnostics(spec, root / "swallowed", GEOMETRY)
            self.assertEqual(code, round_diagnostics.FATAL_MEMORY_EXIT)
            self.assertFalse((root / "swallowed" / "report.json").exists())
            with mock.patch.object(
                round_diagnostics, "propose", side_effect=RuntimeError("unrelated")
            ):
                with self.assertRaisesRegex(RuntimeError, "unrelated"):
                    round_diagnostics.run_diagnostics(spec, root / "other", GEOMETRY)

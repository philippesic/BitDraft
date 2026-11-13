import gc
import hashlib
import os
import signal
import tempfile
import threading
import unittest
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Literal, cast
from unittest import mock

import numpy as np
import torch

from tests.numerical.test_model import gguf_fixture, tiny_model
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact, Supervision
from w1a1_eagle.identity import source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.model import Geometry
from w1a1_eagle.numerical.captures import Capture, CaptureCache, CaptureSpec, load_capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.entry import run_leased
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, reference
from w1a1_eagle.training import Budget, Grouping

BUDGET = 32 * 1024 * 1024
LENGTHS = (24, 40, 32, 24, 40)


def authenticated(path: Path) -> Artifact:
    return Artifact(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)


def sequences(root: Path, hidden: int, lengths: tuple[int, ...] = LENGTHS) -> Artifact:
    records: list[Record] = []
    generator = np.random.default_rng(23)
    for index, length in enumerate(lengths):
        folder = root / f"sequence-{index}"
        folder.mkdir()
        half = index % 2 == 1
        suffix = "f16" if half else "f32"
        files = []
        for tap in (2, 6, 10, 14, 18, 22, 26, 30):
            path = folder / f"layer-{tap}.{suffix}"
            values = generator.standard_normal((length, hidden)).astype("<f2" if half else "<f4")
            path.write_bytes(values.tobytes())
            files.append(reference(authenticated(path)))
        metadata = folder / "capture.json"
        fields: Record = {
            "taps": [2, 6, 10, 14, 18, 22, 26, 30],
            "feature_semantics": "zero_based_pre_block_input_all_tokens",
            "input_tokens": [(index + position) % 18 for position in range(length)],
            "prompt_tokens": 8,
            "logit_positions": list(range(8, length)),
            "greedy_ids": [(index + position) % 18 for position in range(9, length)] + [19],
            "eog_ids": [19],
        }
        if half:
            fields["feature_dtype"] = "float16"
        metadata.write_bytes(canonical(fields))
        records.append(
            {
                "id": f"sequence-{index}",
                "source": "synthetic-fixture",
                "group": f"group-{index}",
                "topic": None,
                "aliases": [],
                "role": "train",
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
                "expected_train": len(lengths),
                "records": records,
            }
        )
    )
    return authenticated(manifest)


class Synchronous:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]:
        row, anchors = self.corpus.locate(cursor)
        capture = load_capture(row.capture, BUDGET, self.corpus.shape.hidden)
        by_anchor = {window.anchor: window for window in capture.windows}
        return capture, tuple(by_anchor[anchor] for anchor in anchors)


def trained_state(worker: TensorWorker) -> dict[str, torch.Tensor]:
    state = {f"model.{k}": v.detach().clone() for k, v in worker.model.named_parameters()}
    for key, values in worker.optimizer.state_dict()["state"].items():
        for name, value in values.items():
            state[f"optimizer.{key}.{name}"] = value.clone()
    return state


def loads_by_thread() -> tuple[
    list[tuple[str, str]], Callable[[CaptureCache, CaptureSpec], Capture]
]:
    seen: list[tuple[str, str]] = []
    original = CaptureCache.load

    def load(cache: CaptureCache, spec: CaptureSpec) -> Capture:
        seen.append((threading.current_thread().name, spec.identity))
        return original(cache, spec)

    return seen, load


class TestPrefetch(unittest.TestCase):
    def corpus(self, manifest: Artifact) -> Corpus:
        return Corpus(manifest, 3, BUDGET, 4, grouping=Grouping(3), shape=tiny_model()[0].geometry)

    def train(self, worker: TensorWorker, updates: int) -> list[float]:
        losses = []
        for target in range(worker.updates + 1, updates + 1):
            worker.run(Monotonic(), Budget(target, 60, 1000), lambda: False)
            losses.append(cast(float, worker.progress.last_loss))
        return losses

    def assert_same_state(self, left: TensorWorker, right: TensorWorker) -> None:
        expected = trained_state(right)
        actual = trained_state(left)
        self.assertEqual(set(actual), set(expected))
        for name, value in actual.items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertEqual((left.updates, left.cursor), (right.updates, right.cursor))
        self.assertEqual(left.accounting.observed_labels, right.accounting.observed_labels)

    def test_prefetched_training_matches_synchronous_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = sequences(Path(directory), 4)
            corpus = self.corpus(manifest)
            prefetched = TensorWorker(tiny_model()[0], corpus, "prefetch")
            synchronous = TensorWorker(
                tiny_model()[0], Synchronous(self.corpus(manifest)), "prefetch"
            )
            switches = 0
            seen, load = loads_by_thread()
            with mock.patch.object(CaptureCache, "load", load):
                for cursor in range(12):
                    expected = self.train(synchronous, cursor + 1)
                    self.assertEqual(self.train(prefetched, cursor + 1), expected)
                    upcoming = corpus.locate(cursor + 1)[0].capture
                    if upcoming.identity != corpus.locate(cursor)[0].capture.identity:
                        switches += 1
                        pending = corpus.cache.pending
                        self.assertIsNotNone(pending)
                        self.assertEqual(
                            cast(tuple[str, object], pending)[0], CaptureCache.key(upcoming)
                        )
            self.assertGreaterEqual(switches, 4)
            self.assertGreaterEqual(sum(name.startswith("capture-prefetch") for name, _ in seen), 4)
            self.assert_same_state(prefetched, synchronous)
            self.assertEqual(
                prefetched.accounting.observed_labels, synchronous.accounting.observed_labels
            )

    def test_resume_branch_and_continue_across_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = sequences(root, 4)
            original = TensorWorker(tiny_model()[0], self.corpus(manifest), "recipe")
            self.train(original, 3)
            corpus = cast(Corpus, original.samples)
            self.assertIsNotNone(corpus.cache.pending)
            parent = original.checkpoint(Store(root / "states"), 4 * 1024 * 1024)
            expected = self.train(original, 8)
            resumed = TensorWorker(tiny_model()[0], self.corpus(manifest), "recipe")
            resumed.restore(parent, 8 * 1024 * 1024)
            self.assertEqual(self.train(resumed, 8), expected)
            self.assert_same_state(resumed, original)
            pending = cast(tuple[str, Future[Capture]], corpus.cache.pending)
            self.assertNotEqual(pending[0], CaptureCache.key(corpus.locate(3)[0].capture))
            original.restore(parent, 8 * 1024 * 1024)
            self.assertEqual(self.train(original, 8), expected)
            repeated = trained_state(original)
            for name, value in trained_state(resumed).items():
                self.assertTrue(torch.equal(repeated[name], value), name)
            modes: tuple[tuple[Literal["branch", "continue"], str], ...] = (
                ("branch", "branched"),
                ("continue", "continued"),
            )
            for mode, identity in modes:
                child = TensorWorker(tiny_model()[0], self.corpus(manifest), identity)
                reference_child = TensorWorker(
                    tiny_model()[0], Synchronous(self.corpus(manifest)), identity
                )
                child.restore(parent, 8 * 1024 * 1024, mode=mode)
                reference_child.restore(parent, 8 * 1024 * 1024, mode=mode)
                self.assertEqual(self.train(child, 7), self.train(reference_child, 7))
                self.assert_same_state(child, reference_child)

    def test_prefetch_memory_is_released_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self.corpus(sequences(Path(directory), 4))
            cursor = next(
                cursor
                for cursor in range(20)
                if corpus.locate(cursor)[0].prompt.identity
                != corpus.locate(cursor + 1)[0].prompt.identity
            )
            capture, _ = corpus.sample(cursor)
            pending = cast(tuple[str, object], corpus.cache.pending)
            self.assertEqual(pending[0], CaptureCache.key(corpus.locate(cursor + 1)[0].capture))
            self.assertEqual(len(corpus.cache.entries), 1)
            del capture, pending
            corpus.cache.clear()
            gc.collect()
            self.assertIsNone(corpus.cache.pending)
            self.assertEqual(len(corpus.cache.entries), 0)
            self.assertFalse(any(type(item) is Capture for item in gc.get_objects()))

    def test_authentication_refuses_changes_with_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = self.corpus(sequences(Path(directory), 4))
            spec = corpus.rows[1].capture
            cache = CaptureCache(BUDGET, 4)
            first = cache.get(spec)
            self.assertEqual(first.features.dtype, torch.float16)
            feature = next(ref for ref in spec.files if ref.path.suffix == ".f16")
            data = bytearray(feature.path.read_bytes())
            data[0] ^= 1
            feature.path.write_bytes(bytes(data))
            self.assertIs(cache.get(spec), first)
            cache.clear()
            with self.assertRaisesRegex(ValueError, "content"):
                cache.get(spec)
            cache.prefetch(spec)
            failed = cast(tuple[str, Future[Capture]], cache.pending)[1]
            self.assertIsInstance(failed.exception(), ValueError)
            data[0] ^= 1
            feature.path.write_bytes(bytes(data))
            self.assertTrue(torch.equal(cache.get(spec).features, first.features))
            data[0] ^= 1
            feature.path.write_bytes(bytes(data))
            cache.clear()
            cache.prefetch(spec)
            with self.assertRaisesRegex(ValueError, "content"):
                cache.get(spec)
            feature.path.write_bytes(bytes(data[:-2]))
            cache.clear()
            with self.assertRaisesRegex(ValueError, "geometry|size"):
                cache.get(spec)
            data[0] ^= 1
            feature.path.write_bytes(bytes(data))
            self.assertTrue(torch.equal(cache.get(spec).features, first.features))
            metadata = spec.metadata.path.read_bytes()
            spec.metadata.path.write_bytes(
                metadata.replace(b'"prompt_tokens":8', b'"prompt_tokens":9')
            )
            cache.clear()
            with self.assertRaisesRegex(ValueError, "content"):
                cache.get(spec)
            spec.metadata.path.write_bytes(metadata)
            cache.clear()
            self.assertTrue(torch.equal(cache.get(spec).features, first.features))


class TestPrefetchEntry(unittest.TestCase):
    def prepare(self, root: Path) -> Path:
        cfg = Geometry(hidden=256, intermediate=256, heads=4, kv_heads=1, head_dim=8, layers=1)
        source, vocabulary = gguf_fixture(root, cfg)
        corpus = sequences(root, 256)
        intent: Record = {
            "schema": 2,
            "objective": "exercise prefetch through the worker entry",
            "checkpoint_updates": 1,
            "checkpoint_seconds": 30,
            "hypothesis": "prefetching leaves the journal and states unchanged",
            "decision": "CPU plumbing qualification only",
            "risk": "synthetic source cannot establish native parity or learning usefulness",
            "authority_scope": "disposable-CPU-test",
            "source": reference(source),
            "vocabulary": reference(vocabulary),
            "corpus": reference(corpus),
            "parent": None,
            "mode": "fresh",
            "budget": {"updates": 5, "wall_seconds": 60, "attempts": 6, "retries": 0},
            "memory_bytes": 64 * 1024 * 1024,
            "state_bytes": 8 * 1024 * 1024,
            "disk_reserve_bytes": 1,
            "backing_available_bytes": 256 * 1024 * 1024,
            "seed": 4,
            "device": "cpu",
            "grouping": {"max_windows": 4, "activation_bytes": 1 << 30, "device_bytes": 1 << 30},
            "science_policy": {
                "id": "synthetic-fixture-only",
                "min_unique_positions": 16,
                "max_mean_visits": 100,
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
        path = root / "intent.json"
        path.write_bytes(canonical(intent))
        return path

    def execute(self, intent: Path, output: Path) -> None:
        admitted = Intent.load(intent)
        receipt = admitted.seal(output / "intents")
        journal = Journal(output)
        journal.append(
            {
                "kind": "started",
                "attempt": "test",
                "intent": admitted.sha256,
                "controller": cast(Record, asdict(ProcessIdentity.read(os.getppid()))),
            }
        )
        journal.append(
            {
                "kind": "process",
                "attempt": "test",
                "owner": cast(Record, asdict(ProcessIdentity.read(os.getpid()))),
            }
        )
        handlers = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
        try:
            run_leased(receipt, output, "test")
        finally:
            for value, handler in handlers.items():
                signal.signal(value, handler)

    def test_failure_journals_no_prefetched_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent = self.prepare(root)
            original = TensorWorker.update

            def failing(
                worker: TensorWorker, capture: Capture, windows: tuple[Supervision, ...]
            ) -> float:
                if worker.updates == 2:
                    raise ValueError("injected update failure")
                return original(worker, capture, windows)

            outcomes = []
            for prefetch in (True, False):
                output = root / ("prefetch" if prefetch else "synchronous")
                seen, load = loads_by_thread()
                with (
                    mock.patch.object(TensorWorker, "update", failing),
                    mock.patch.object(CaptureCache, "load", load),
                    mock.patch.object(
                        CaptureCache,
                        "prefetch",
                        CaptureCache.prefetch if prefetch else lambda cache, spec: None,
                    ),
                    self.assertRaisesRegex(ValueError, "injected"),
                ):
                    self.execute(intent, output)
                events = Journal(output).events()
                journaled = [
                    (
                        row["kind"],
                        row.get("cursor"),
                        row.get("trajectory"),
                        row.get("anchors"),
                        row.get("labels"),
                    )
                    for row in events
                    if row["kind"] in {"update-attempt", "exposure", "checkpoint"}
                ]
                progress = [
                    {
                        key: record(row["progress"])[key]
                        for key in ("updates", "cursor", "attempts", "failed", "needs_recovery")
                    }
                    for row in events
                    if row["kind"] == "worker-failure"
                ]
                latest = Store(output / "states").latest(Intent.load(intent).recipe)
                state = torch.load(latest.artifact.path, map_location="cpu", weights_only=True)
                prefetched = [
                    identity for name, identity in seen if name.startswith("capture-prefetch")
                ]
                outcomes.append((journaled, progress, latest.updates, state["model"], prefetched))
            with_prefetch, without = outcomes
            self.assertEqual(with_prefetch[0], without[0])
            self.assertEqual(with_prefetch[1], without[1])
            self.assertEqual(with_prefetch[1][0]["updates"], 2)
            self.assertEqual(with_prefetch[2], without[2])
            for name, value in with_prefetch[3].items():
                self.assertTrue(torch.equal(value, without[3][name]), name)
            journaled_cursors = {row[1] for row in with_prefetch[0] if row[0] == "update-attempt"}
            self.assertEqual(journaled_cursors, {0, 1, 2})
            self.assertTrue(with_prefetch[4])
            self.assertFalse(without[4])

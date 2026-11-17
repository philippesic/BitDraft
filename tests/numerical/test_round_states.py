import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

from tests.numerical.test_features import seeded, snapshot, worker
from tests.numerical.test_model import (
    LOADER_GEOMETRY,
    OMITTED_RECIPE,
    authenticated,
    execute_leased,
    loader_fixture,
    row_optimization,
    tiny_model,
)
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, number, record, sequence
from w1a1_eagle.data import Artifact, Supervision, TailWindow, Window
from w1a1_eagle.drafting import generated_anchors, simulate
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.numerical.captures import Capture, load_capture
from w1a1_eagle.numerical.entry import admission, read_corpus, round_state_weighting
from w1a1_eagle.numerical.round_diagnostics import propose
from w1a1_eagle.numerical.round_state_map import (
    SPEC_SCHEMA,
    build,
    propose_first,
    propose_folded,
)
from w1a1_eagle.numerical.worker import TensorWorker, label_weights
from w1a1_eagle.records import artifact, canonical, reference
from w1a1_eagle.round_states import (
    MAP_SCHEMA,
    RoundStateMap,
    RoundStateWeighting,
    covered_windows,
    load_map,
    needed_spans,
    parse_map,
    round_starts,
    window_shares,
)
from w1a1_eagle.training import Budget

Rows = dict[str, tuple[list[tuple[int, int]], list[int]]]


def map_record(rows: Rows, corpus: tuple[str, int], target: str = "a" * 64) -> Record:
    return {
        "schema": MAP_SCHEMA,
        "corpus": {"sha256": corpus[0], "bytes": corpus[1], "format": None},
        "target_sha256": target,
        "depth": 3,
        "scope": {"kind": "full"},
        "drafter": {"state": {"sha256": "d" * 64, "bytes": 1}},
        "code_sha256": "0" * 64,
        "rows": [
            {"id": name, "covered": [list(span) for span in spans], "starts": starts}
            for name, (spans, starts) in rows.items()
        ],
        "totals": {
            "rows": len(rows),
            "covered_windows": sum(
                last - first + 1 for spans, _ in rows.values() for first, last in spans
            ),
            "proposals": 0,
            "round_starts": sum(len(starts) for _, starts in rows.values()),
        },
    }


def written_map(path: Path, rows: Rows, corpus: tuple[str, int]) -> Artifact:
    path.write_bytes(canonical(map_record(rows, corpus)))
    return authenticated(path)


def fixture_map() -> RoundStateMap:
    raw = map_record({"fixture": ([(0, 19)], [4, 17])}, ("c" * 64, 1))
    return parse_map(raw, "e" * 64, 1)


class TestRoundStartMap(unittest.TestCase):
    def test_round_starts_follow_accepted_runs_exactly(self) -> None:
        labels = {
            anchor: tuple(100 + anchor + 1 + slot for slot in range(3)) for anchor in range(20)
        }
        accepted = {1: 3, 5: 0, 6: 1, 8: 2, 11: 3, 15: 3, 19: 0}
        proposals: dict[int, tuple[tuple[int], ...]] = {}
        for anchor in range(1, 20):
            run = accepted.get(anchor, anchor % 4)
            proposals[anchor] = tuple(
                (labels[anchor][slot] if slot < run else 7,) for slot in range(3)
            )
        windows = {anchor: labels[anchor] for anchor in range(20)}
        found = round_starts(windows, proposals, 3)
        self.assertEqual(found, (1, 5, 6, 8, 11, 15, 19))
        for first, second in zip(found, found[1:]):
            self.assertEqual(second, first + accepted[first] + 1)
        spans = needed_spans(tuple(range(20)), (3, 12))
        self.assertEqual(spans, ((0, 12),))
        prefix = round_starts(covered_windows(windows, spans), proposals, 3)
        self.assertEqual(prefix, (1, 5, 6, 8, 11))
        self.assertEqual(generated_anchors(covered_windows(windows, spans)), tuple(range(1, 13)))

    def test_spans_follow_segments_and_refuse_foreign_anchors(self) -> None:
        anchors = (2, 3, 4, 5, 9, 10, 11, 20)
        self.assertEqual(needed_spans(anchors), ((2, 5), (9, 11), (20, 20)))
        self.assertEqual(needed_spans(anchors, (3, 10, 11)), ((2, 3), (9, 11)))
        self.assertEqual(needed_spans(anchors, ()), ())
        with self.assertRaisesRegex(ValueError, "outside"):
            needed_spans(anchors, (6,))

    def test_session_spans_start_after_each_turn_prompt(self) -> None:
        windows = {anchor: (anchor + 1,) * 3 for anchor in (*range(0, 6), *range(10, 14))}
        proposals = {anchor: ((anchor + 1,), (anchor + 1,), (anchor + 1,)) for anchor in windows}
        rounds = simulate(windows, proposals, 3)
        self.assertEqual([item.anchor for item in rounds], [1, 5, 11])

    def test_map_lookup_and_refusals(self) -> None:
        states = fixture_map()
        windows: tuple[Supervision, ...] = (
            Window("fixture", 2, tuple(range(3, 18))),
            Window("fixture", 4, tuple(range(5, 20))),
            TailWindow("fixture", 17, (18, 19)),
        )
        self.assertEqual(states.flags(windows), (False, True, True))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            states.flags((Window("other", 2, tuple(range(3, 18))),))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            states.flags((TailWindow("fixture", 20, (1,)),))
        bad: list[tuple[Rows, str]] = [
            ({"x": ([(3, 9)], [3])}, "follow a covered anchor"),
            ({"x": ([(3, 9)], [10])}, "follow a covered anchor"),
            ({"x": ([(3, 9)], [6, 5])}, "ordered"),
            ({"x": ([(3, 9), (10, 12)], [])}, "ordered and separate"),
        ]
        for rows, message in bad:
            with self.assertRaisesRegex(ValueError, message):
                parse_map(map_record(rows, ("c" * 64, 1)), "e" * 64, 1)
        broken = map_record({"x": ([(3, 9)], [4])}, ("c" * 64, 1))
        broken["totals"] = {**record(broken["totals"]), "round_starts": 2}
        with self.assertRaisesRegex(ValueError, "totals"):
            parse_map(broken, "e" * 64, 1)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_map({**broken, "schema": "other"}, "e" * 64, 1)

    def test_precomputed_contexts_match_the_grouped_path(self) -> None:
        model, features = tiny_model(window=3)
        tokens = tuple(range(20))
        anchors = (3, 4, 9, 15, 18)
        with torch.no_grad():
            contexts = model.group_contexts(features, tokens, 19)
            given = model.group_hidden(features, tokens, anchors, 4, False, contexts)
            plain = model.group_hidden(features, tokens, anchors, 4)
            self.assertTrue(torch.allclose(given, plain, rtol=0, atol=1e-6))
            with self.assertRaisesRegex(ValueError, "do not cover"):
                model.group_hidden(features, tokens, anchors, 4, False, contexts[:-1])
        capture = Capture(tokens, features, ())
        first = propose_first(model, capture, anchors, 3, 2)
        ranked = propose(model, capture, anchors, 3, 1, 2)
        self.assertEqual(first, ranked)
        for chunk in (1, 2, 5):
            self.assertEqual(propose_folded(model, capture, anchors, 3, chunk), first)
        self.assertEqual(propose_folded(model, capture, (), 3, 2), {})
        with torch.no_grad():
            for anchor in anchors:
                single = model.forward(features, tokens, anchor, 4)[1:].argmax(-1).tolist()
                self.assertEqual(first[anchor], tuple((token,) for token in single))


class TestWeighting(unittest.TestCase):
    def test_shares_keep_the_update_total_and_ratio(self) -> None:
        shares = window_shares((True, False, False, True), 2.0)
        self.assertAlmostEqual(sum(shares), 1.0, places=12)
        self.assertAlmostEqual(shares[0] / shares[1], 2.0, places=12)
        self.assertEqual(window_shares((False, False), 2.0), (0.5, 0.5))
        self.assertEqual(window_shares((True, True, True), 3.0), (1 / 3, 1 / 3, 1 / 3))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            window_shares((True,), 0.0)
        lengths = (15, 3, 2)
        weights = label_weights(lengths, None, (0.5, 0.25, 0.25))
        self.assertAlmostEqual(float(weights[:15].sum()), 0.5, places=6)
        self.assertAlmostEqual(float(weights[15:18].sum()), 0.25, places=6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        expected = []
        for length in lengths:
            decay = torch.exp(-torch.arange(length) / 7)
            expected.append(decay / decay.sum() / len(lengths))
        self.assertTrue(torch.equal(label_weights(lengths), torch.cat(expected)))
        with self.assertRaisesRegex(ValueError, "one window share"):
            label_weights(lengths, None, (1.0,))

    def test_update_loss_is_the_share_weighted_window_loss(self) -> None:
        weighted = worker("weighted")
        weighted.round_states = RoundStateWeighting(fixture_map(), 2.0)
        capture, windows = weighted.samples.sample(0)
        loss = float(weighted.group_loss(capture, windows, 16))
        singles = [float(worker("plain").group_loss(capture, (window,), 16)) for window in windows]
        shares = window_shares((False, True, True), 2.0)
        self.assertAlmostEqual(loss, sum(s * v for s, v in zip(shares, singles)), places=5)
        plain = float(worker("plain").group_loss(capture, windows, 16))
        self.assertAlmostEqual(plain, sum(singles) / 3, places=5)
        self.assertNotAlmostEqual(loss, plain, places=4)

    def test_progress_records_count_trained_round_starts(self) -> None:
        target = worker("weighted")
        target.round_states = RoundStateWeighting(fixture_map(), 2.0)
        target.run(Monotonic(), Budget(3, 60, 3), lambda: False)
        self.assertEqual(
            target.round_state_record(),
            {"windows": 8, "round_starts": 6, "share": 0.75, "weight": 2.0},
        )
        self.assertEqual(
            target.round_state_record(),
            {"windows": 0, "round_starts": 0, "share": None, "weight": 2.0},
        )
        self.assertIsNone(worker("plain").round_state_record())

    def test_resume_with_the_setting_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-round-resume-") as directory:
            weighting = RoundStateWeighting(fixture_map(), 2.0)
            original = worker("weighted")
            original.round_states = weighting
            original.run(Monotonic(), Budget(1, 60, 1), lambda: False)
            parent = original.checkpoint(Store(Path(directory)), 4 * 1024 * 1024)
            original.run(Monotonic(), Budget(3, 60, 3), lambda: False)
            resumed = worker("weighted")
            resumed.round_states = weighting
            resumed.restore(parent, 8 * 1024 * 1024)
            resumed.run(Monotonic(), Budget(3, 60, 3), lambda: False)
            self.assertEqual(resumed.progress.last_loss, original.progress.last_loss)
            expected = snapshot(original.model)
            for name, value in resumed.model.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]), name)
            control = worker("weighted")
            control.run(Monotonic(), Budget(3, 60, 3), lambda: False)
            self.assertNotEqual(control.progress.last_loss, original.progress.last_loss)


class TestIntentSetting(unittest.TestCase):
    def test_omitted_setting_keeps_the_recipe_and_values_enter_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-round-intent-") as directory:
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
            map_ref = {"path": str(root / "map.json"), "sha256": "5" * 64, "bytes": 9}

            def parse(changes: Record) -> Intent:
                path.write_bytes(canonical({**base, **changes}))
                return Intent.load(path)

            omitted = parse({})
            self.assertIsNone(omitted.round_states)
            self.assertEqual(omitted.recipe, OMITTED_RECIPE)
            two = parse({"round_states": {"map": map_ref, "weight": 2.0}})
            integral = parse({"round_states": {"map": map_ref, "weight": 2}})
            three = parse({"round_states": {"map": map_ref, "weight": 3.0}})
            other = parse({"round_states": {"map": {**map_ref, "sha256": "6" * 64}, "weight": 2.0}})
            assert two.round_states is not None
            self.assertEqual(two.round_states.weight, 2.0)
            self.assertEqual(two.round_states.map.sha256, "5" * 64)
            self.assertEqual(two.recipe, integral.recipe)
            self.assertEqual(len({OMITTED_RECIPE, two.recipe, three.recipe, other.recipe}), 4)
            refusals: list[tuple[object, str]] = [
                ({"map": map_ref}, "map and weight"),
                ({"map": map_ref, "weight": 2.0, "extra": 1}, "map and weight"),
                ({"map": map_ref, "weight": 0}, "positive finite"),
                ({"map": map_ref, "weight": -1.5}, "positive finite"),
            ]
            for value, message in refusals:
                with self.assertRaisesRegex(ValueError, message):
                    parse({"round_states": value})

    def test_admission_refuses_another_corpus_or_uncovered_windows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-round-admit-") as directory:
            root = Path(directory)
            loaded = loader_fixture(root)
            size = loaded.corpus.size
            path = root / "intent.json"

            def intent_with(rows: Rows, corpus: tuple[str, int]) -> Intent:
                states = written_map(root / "map.json", rows, corpus)
                weighted = {
                    **loaded.intent,
                    "round_states": {"map": reference(states), "weight": 2.0},
                }
                path.write_bytes(canonical(weighted))
                return Intent.load(path)

            path.write_bytes(canonical(loaded.intent))
            corpus = read_corpus(Intent.load(path))
            row = corpus.rows[0]
            name = row.capture.identity
            whole = list(needed_spans(row.anchors))
            starts = [whole[0][0] + 1]
            accepted = intent_with({name: (whole, starts)}, (loaded.corpus.sha256, size))
            result = record(admission(accepted, read_corpus(accepted))["round_states"])
            sampled = corpus.locate(0)[1]
            self.assertEqual(integer(result["planned_windows"]), len(sampled))
            self.assertEqual(integer(result["planned_round_starts"]), int(starts[0] in sampled))
            self.assertEqual(number(result["weight"]), 2.0)
            refusals: list[tuple[Rows, tuple[str, int], str]] = [
                ({name: (whole, starts)}, ("f" * 64, size), "another corpus"),
                ({name: (whole, starts)}, (loaded.corpus.sha256, size + 1), "another corpus"),
                (
                    {name: ([(whole[0][0], whole[0][0])], [])},
                    (loaded.corpus.sha256, size),
                    "does not cover the window",
                ),
            ]
            for rows, identity, message in refusals:
                refused = intent_with(rows, identity)
                with self.assertRaisesRegex(ValueError, message):
                    admission(refused, read_corpus(refused))
            plain = Intent.load(_plain(root, loaded.intent))
            self.assertNotIn("round_states", admission(plain, read_corpus(plain)))
            self.assertIsNone(round_state_weighting(plain, read_corpus(plain)))


def _plain(root: Path, intent: Record) -> Path:
    path = root / "plain-intent.json"
    path.write_bytes(canonical(intent))
    return path


class TestMapBuild(unittest.TestCase):
    def test_folded_scoring_writes_the_same_map(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-round-folded-") as directory:
            root = Path(directory)
            corpus, model, base = seeded(root)
            state = TensorWorker(model, corpus, "parent").checkpoint(
                Store(root / "states"), 1 << 26
            )
            spec: Record = {
                "schema": SPEC_SCHEMA,
                "memory_bytes": 4 << 30,
                "activation_bytes": 64 << 20,
                "drafter": {
                    "source": base["source"],
                    "vocabulary": base["vocabulary"],
                    "state": reference(state.artifact),
                    "mode": "continue",
                },
                "corpus": base["corpus"],
                "chunk": 3,
            }
            reports: dict[str, Record] = {}
            for name, changes in (
                ("default", {}),
                ("batched", {"scoring": "batched"}),
                ("folded", {"scoring": "folded"}),
            ):
                path = root / (name + ".json")
                path.write_bytes(canonical({**spec, **changes}))
                with contextlib.redirect_stdout(io.StringIO()):
                    reports[name] = build(path, root / name, LOADER_GEOMETRY)
            maps = {
                name: load_map(artifact(report["map"], root)) for name, report in reports.items()
            }
            self.assertEqual(reports["default"]["scoring"], "batched")
            self.assertEqual(reports["folded"]["scoring"], "folded")
            self.assertEqual(maps["default"].sha256, maps["batched"].sha256)
            self.assertEqual(maps["folded"].sha256, maps["default"].sha256)
            self.assertTrue(any(maps["folded"].starts.values()))
            self.assertEqual(reports["folded"]["proposals"], reports["default"]["proposals"])
            covered = integer(reports["folded"]["covered_windows"])
            compute = number(record(reports["folded"]["seconds"])["compute"])
            self.assertAlmostEqual(
                number(reports["folded"]["windows_per_compute_second"]), covered / compute
            )
            bad = root / "bad.json"
            bad.write_bytes(canonical({**spec, "scoring": "fast"}))
            with self.assertRaisesRegex(ValueError, "scoring must be one of"):
                build(bad, root / "bad", LOADER_GEOMETRY)

    def test_full_and_restricted_maps_match_an_independent_simulation_and_train(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="binary-round-map-") as directory:
            root = Path(directory)
            corpus, model, base = seeded(root)
            state = TensorWorker(model, corpus, "parent").checkpoint(
                Store(root / "states"), 1 << 26
            )
            drafter: Record = {
                "source": base["source"],
                "vocabulary": base["vocabulary"],
                "state": reference(state.artifact),
                "mode": "continue",
            }
            spec: Record = {
                "schema": SPEC_SCHEMA,
                "memory_bytes": 4 << 30,
                "activation_bytes": 64 << 20,
                "drafter": drafter,
                "corpus": base["corpus"],
                "depth": 3,
                "chunk": 2,
            }
            spec_path = root / "map-spec.json"
            spec_path.write_bytes(canonical(spec))
            with contextlib.redirect_stdout(io.StringIO()):
                report = build(spec_path, root / "full", LOADER_GEOMETRY)
            full = load_map(artifact(report["map"], root))
            self.assertEqual(full.scope, {"kind": "full"})
            self.assertEqual(full.corpus_sha256, record(base["corpus"])["sha256"])
            self.assertEqual(record(full.drafter["state"])["sha256"], state.artifact.sha256)
            row = corpus.rows[0]
            capture = load_capture(row.capture, 1 << 30, LOADER_GEOMETRY.hidden)
            windows = {window.anchor: window.labels for window in capture.windows}
            anchors = generated_anchors(windows)
            with torch.no_grad():
                ranked = propose(model, capture, anchors, 3, 1, 5)
            expected = tuple(item.anchor for item in simulate(windows, ranked, 3))
            self.assertEqual(tuple(sorted(full.starts[row.capture.identity])), expected)
            self.assertTrue(expected)
            self.assertEqual(report["proposals"], len(anchors))
            self.assertEqual(report["full_corpus_proposals"], len(anchors))
            self.assertIsNotNone(report["forecast_full_compute_seconds"])
            intent: Record = {
                **base,
                "optimization": row_optimization(),
                "budget": {"updates": 2, "wall_seconds": 60, "attempts": 3, "retries": 0},
                "science_policy": {
                    "id": "synthetic-fixture-only",
                    "min_unique_positions": 16,
                    "max_mean_visits": 8,
                },
            }
            control_path = root / "control.json"
            control_path.write_bytes(canonical(intent))
            spec_path.write_bytes(
                canonical({**spec, "restrict": reference(authenticated(control_path))})
            )
            with contextlib.redirect_stdout(io.StringIO()):
                restricted_report = build(spec_path, root / "restricted", LOADER_GEOMETRY)
            restricted = load_map(artifact(restricted_report["map"], root))
            self.assertEqual(restricted.scope["kind"], "intent")
            self.assertEqual(restricted.scope["cursors"], 2)
            control = Intent.load(control_path)
            sampled = {
                anchor for cursor in range(2) for anchor in read_corpus(control).locate(cursor)[1]
            }
            spans = restricted.covered[row.capture.identity]
            self.assertEqual(spans[-1][1], max(sampled))
            self.assertEqual(
                restricted.starts[row.capture.identity],
                frozenset(anchor for anchor in expected if anchor <= max(sampled)),
            )
            weighted_path = root / "weighted.json"
            weighted: Record = {
                **intent,
                "round_states": {"map": restricted_report["map"], "weight": 2.0},
            }
            weighted_path.write_bytes(canonical(weighted))
            admitted = Intent.load(weighted_path)
            self.assertNotEqual(admitted.recipe, control.recipe)
            execute_leased(weighted_path, root / "run")
            events = list(Journal(root / "run").events())
            granted = next(event for event in events if event.get("kind") == "admission")
            planned = record(granted["round_states"])
            self.assertEqual(planned["sha256"], restricted.sha256)
            self.assertEqual(planned["planned_cursors"], 2)
            progress = [event for event in events if event.get("kind") == "training-progress"]
            self.assertTrue(progress)
            self.assertTrue(all("round_states" in event for event in progress))
            exposures = [event for event in events if event.get("kind") == "exposure"]
            self.assertEqual(len(exposures), 2)
            for exposure in exposures:
                for anchor in sequence(exposure["anchors"]):
                    self.assertTrue(restricted.covers(row.capture.identity, integer(anchor)))

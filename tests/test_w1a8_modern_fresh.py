"""Synthetic temporary selection freezes only; never actual confirmation data."""

import copy
import importlib.util
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization/modern_fresh.py"
SPEC = importlib.util.spec_from_file_location("modern_fresh", PATH)
fresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fresh)


class ModernFreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-modern-fresh-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.freeze_path = self.root / "selected-synthetic.json"
        source = self.root / "source-freeze.json"
        source.write_text('{"scope":"synthetic fixture, no models or GPU"}\n')
        exclusions = []
        for role in sorted(fresh.ROLES):
            path = self.root / f"{role}.json"
            path.write_bytes(
                fresh.encoded(
                    {
                        "schema": "modern_fresh_exclusion_index_v1",
                        "records": [fresh.identities("Previously used synthetic " + role)],
                    }
                )
            )
            exclusions.append(
                {
                    "role": role,
                    "format": "hash_index_v1",
                    "file": fresh.reference(path),
                    "previously_used": True,
                    "sealed": False,
                    "complete": True,
                }
            )
        self.freeze = {
            "schema": fresh.SCHEMA,
            "selection_frozen": True,
            "selected_arm_id": "synthetic_selected",
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "candidate": {
                "path": str(self.root / "NEVER-OPEN-model.gguf"),
                "bytes": 100,
                "sha256": "a" * 64,
            },
            "runtime": {
                "path": str(self.root / "NEVER-OPEN-server"),
                "bytes": 100,
                "sha256": "b" * 64,
            },
            "source_freeze": fresh.reference(source),
            "generator": fresh.reference(PATH),
            "judge": fresh.reference(fresh.TRUSTED_JUDGE),
            "settings": {"output_limit": 256},
            "exclusions": exclusions,
        }
        source.write_bytes(
            fresh.encoded(
                {
                    "arms": [
                        {
                            "id": "synthetic_selected",
                            "kind": "w1a8",
                            "model": self.freeze["candidate"],
                            "runtime": {
                                "server": self.freeze["runtime"],
                                "libraries": [self.freeze["runtime"]],
                                "environment": {"fixture": "not executed"},
                                "source_pins": [fresh.reference(PATH)],
                            },
                        }
                    ]
                }
            )
        )
        self.freeze["source_freeze"] = fresh.reference(source)
        self.write_freeze()
        self.judge = fresh.load_judge(self.freeze["judge"])

    def write_freeze(self):
        self.freeze_path.write_bytes(fresh.encoded(self.freeze))

    def read(self, folder, name):
        return json.loads((folder / name).read_text())

    def test_postfreeze_prepare_is_deterministic_bounded_and_private(self):
        first, second = self.root / "first", self.root / "second"
        reads = []
        original = Path.read_bytes

        def spy(path):
            reads.append(str(path))
            return original(path)

        with patch.object(Path, "read_bytes", spy):
            receipt = fresh.prepare(self.freeze_path, first)
        fresh.prepare(self.freeze_path, second)
        self.assertTrue(receipt["admitted"])
        self.assertEqual(receipt["performance_prompts"], 24)
        self.assertEqual(receipt["quality_tasks"], 24)
        self.assertEqual(receipt["quality_correct_controls_passed"], 24)
        self.assertEqual(receipt["quality_known_wrong_controls_rejected"], 24)
        self.assertEqual(
            receipt["manifest_sha256"], fresh.reference(first / "manifest.json")["sha256"]
        )
        for name in [
            "requests.json",
            "quality-cases.json",
            "quality-controls.json",
            "manifest.json",
        ]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertLess(sum(p.stat().st_size for p in first.iterdir()), 2 * 1024**2)
        cases = self.read(first, "quality-cases.json")
        self.assertEqual({c["kind"] for c in cases}, {"python", "json", "sql"})
        self.assertEqual(
            len({g["group_id"] for g in self.read(first, "manifest.json")["groups"]}), 48
        )
        for row in self.read(first, "requests.json") + cases:
            self.assertEqual(
                set(row["request"]),
                {
                    "prompt",
                    "n_predict",
                    "temperature",
                    "seed",
                    "cache_prompt",
                    "return_tokens",
                    "stream",
                },
            )
            self.assertTrue(row["request"]["prompt"].endswith("<think>\n\n</think>\n\n"))
            self.assertEqual(row["request"]["n_predict"], 256)
        self.assertFalse(any("NEVER-OPEN" in path for path in reads))
        with self.assertRaisesRegex(ValueError, "replace"):
            fresh.prepare(self.freeze_path, first)

    def test_no_instance_creation_before_complete_selected_freeze(self):
        mutations = [
            lambda f: f.__setitem__("selection_frozen", False),
            lambda f: f.__setitem__("evaluation_inputs_generated", True),
            lambda f: f.__setitem__("selected_at_utc", "2999-01-01T00:00:00+00:00"),
            lambda f: f["settings"].__setitem__("output_limit", 256.0),
            lambda f: f["settings"].__setitem__("output_limit", 80),
            lambda f: f["exclusions"].pop(),
            lambda f: f["generator"].__setitem__("sha256", "0" * 64),
            lambda f: f["judge"].__setitem__("sha256", "0" * 64),
            lambda f: f["candidate"].__setitem__("sha256", "invalid"),
            lambda f: f["candidate"].__setitem__("sha256", "c" * 64),
            lambda f: f.__setitem__("selected_arm_id", "missing"),
        ]
        original = copy.deepcopy(self.freeze)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.freeze = copy.deepcopy(original)
                mutation(self.freeze)
                self.write_freeze()
                out = self.root / "must-not-exist"
                with patch.object(
                    fresh, "_instance", side_effect=AssertionError("generation forbidden")
                ):
                    with self.assertRaises(ValueError):
                        fresh.prepare(self.freeze_path, out)
                self.assertFalse(out.exists())

    def test_all_twelve_families_have_semantic_discriminating_controls(self):
        for domain, families in fresh.FAMILIES.items():
            for family in families:
                with self.subTest(family=family):
                    case, positive, negatives = fresh._instance(
                        "synthetic-fixture-only", domain, family, "quality", 0
                    )
                    self.assertTrue(self.judge.judge(case, positive)["passed"])
                    self.assertGreaterEqual(len(negatives), 2)
                    self.assertNotEqual(negatives[0], negatives[1])
                    self.assertTrue(all(not self.judge.judge(case, n)["passed"] for n in negatives))
                    self.assertNotIn("Explain the result", case["prompt"])

    def test_json_semantics_format_equivalence_and_malformed_rejection(self):
        case, positive, _ = fresh._instance("synthetic", "reasoning", "finite_sets", "quality", 0)
        answer = json.loads(positive)
        equivalent = json.dumps(dict(reversed(list(answer.items()))), indent=3)
        self.assertTrue(self.judge.judge(case, "```json\n" + equivalent + "\n```")["passed"])
        for text in [
            '{"union":NaN}',
            '{"union":1,"union":1}',
            "[]",
            "prose " + positive,
            fresh._wrong_json(answer, 0),
        ]:
            self.assertFalse(self.judge.judge(case, text)["passed"])
        boolean_case = {"kind": "json", "expected": {"value": 1}}
        self.assertFalse(self.judge.judge(boolean_case, '{"value":true}')["passed"])

    def test_python_sql_semantic_equivalence_and_existing_restrictions(self):
        case, positive, _ = fresh._instance("synthetic", "code", "prefix_sums", "quality", 0)
        alternative = (
            f"def {case['function']}(x):\n    return [sum(x[:i+1]) for i in range(len(x))]\n"
        )
        self.assertNotEqual(positive, alternative)
        self.assertTrue(self.judge.judge(case, alternative)["passed"])
        self.assertFalse(self.judge.judge(case, "import os\n" + positive)["passed"])
        self.assertFalse(self.judge.judge(case, positive + "\nx=1\n")["passed"])
        sql, good, wrong = fresh._instance("synthetic", "code", "sql_unblocked", "quality", 0)
        self.assertTrue(any(row[1] is None for row in sql["data"]))
        self.assertIn("NOT EXISTS", good)
        self.assertTrue(
            self.judge.judge(sql, good.replace("MIN(total)", "MIN(total) AS smallest"))["passed"]
        )
        self.assertTrue(all(not self.judge.judge(sql, n)["passed"] for n in wrong))
        for text in ["DELETE FROM orders", good + "; DELETE FROM orders"]:
            self.assertFalse(self.judge.judge(sql, text)["passed"])

    def test_prior_prompt_wrapper_inventory_is_not_silently_omitted(self):
        document = {
            "requests": [{"prompt": "alpha"}],
            "prompts": ["beta", {"rendered_prompt": "gamma"}],
            "bindings": [{"rendered_text": "delta", "private_answer": "ignore"}],
        }
        self.assertEqual(list(fresh._texts(document)), ["alpha", "beta", "gamma", "delta"])

    def test_overlap_is_normalized_and_preserved_as_inconclusive(self):
        previous = "Previously used synthetic actual_train"
        index_hashes, _ = fresh.load_exclusions(self.freeze["exclusions"])
        self.assertTrue(set(fresh.identities(previous).values()) & index_hashes)
        self.assertTrue(
            set(fresh.identities("  " + previous.replace(" ", "   ") + "\n").values())
            & index_hashes
        )
        make = fresh._instance

        def collide(*args):
            case, good, bad = make(*args)
            if args[3] == "performance" and args[4] == 0:
                case["prompt"] = previous
                case["identities"] = fresh.identities(previous)
            return case, good, bad

        with patch.object(fresh, "_instance", side_effect=collide):
            receipt = fresh.prepare(self.freeze_path, self.root / "overlap")
        self.assertFalse(receipt["admitted"])
        self.assertGreater(receipt["input_overlap_count"], 0)
        self.assertEqual(len(self.read(self.root / "overlap", "manifest.json")["groups"]), 48)
        self.assertEqual(len(self.read(self.root / "overlap", "requests.json")), 24)
        self.assertEqual(receipt["model_requests_executed"], 0)

    def test_control_failure_preserves_all_cases_and_invalidates_packet(self):
        judge = self.judge
        calls = 0

        class FailedOracle:
            @staticmethod
            def judge(case, text):
                nonlocal calls
                calls += 1
                result = judge.judge(case, text)
                return (
                    {"passed": False, "reason": "synthetic oracle failure"}
                    if calls == 1
                    else result
                )

        with patch.object(fresh, "load_judge", return_value=FailedOracle()):
            receipt = fresh.prepare(self.freeze_path, self.root / "inconclusive")
        self.assertEqual(receipt["status"], "INCONCLUSIVE_INPUTS_OR_ORACLE")
        self.assertFalse(receipt["admitted"])
        self.assertEqual(len(self.read(self.root / "inconclusive", "quality-cases.json")), 24)
        self.assertEqual(receipt["quality_correct_controls_passed"], 23)

    def test_sealed_or_changed_exclusions_and_bad_index_reject_without_output(self):
        original = copy.deepcopy(self.freeze)
        self.freeze["exclusions"][0]["sealed"] = True
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "unsealed"):
            fresh.prepare(self.freeze_path, self.root / "sealed")
        self.assertFalse((self.root / "sealed").exists())
        self.freeze = copy.deepcopy(original)
        file = Path(self.freeze["exclusions"][0]["file"]["path"])
        file.write_text('{"schema":"wrong","records":[]}')
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "changed"):
            fresh.prepare(self.freeze_path, self.root / "changed")
        self.freeze["exclusions"][0]["file"] = fresh.reference(file)
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "schema"):
            fresh.prepare(self.freeze_path, self.root / "bad-index")
        self.assertFalse((self.root / "bad-index").exists())


if __name__ == "__main__":
    unittest.main()

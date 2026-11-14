import tempfile
import unittest
from pathlib import Path

import torch

from tests.numerical.test_model import fixture_index, scenario_intent
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, number, record
from w1a1_eagle.identity import source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.entry import admission
from w1a1_eagle.records import canonical, reference
from w1a1_eagle.supervisor import execute


class TestRefusedLaunch(unittest.TestCase):
    def test_refused_launches_journal_attempts_and_recover_on_retry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-loader-") as directory:
            root = Path(directory)
            loaded, intent, intent_path = scenario_intent(root)
            corpus = Corpus(loaded.corpus, 17, 64 * 1024 * 1024, 256)
            with self.assertRaisesRegex(ValueError, "scope"):
                execute(intent_path, root / "not-authorized", "unrelated-task")
            self.assertFalse((root / "not-authorized").exists())
            invalid_policy = dict(intent)
            invalid_policy["science_policy"] = {
                "id": "deliberate-refusal",
                "min_unique_positions": 17,
                "max_mean_visits": 4,
            }
            intent_path.write_bytes(canonical(invalid_policy))
            with self.assertRaisesRegex(ValueError, "distinct-data"):
                admission(Intent.load(intent_path), corpus)
            refused = execute(intent_path, root / "refused-retry", "disposable-CPU-test")
            self.assertEqual(record(refused["latest"])["outcome"], "refused")
            self.assertIsNone(refused["worker"])
            self.assertFalse((root / "refused-retry/states/latest.json").exists())
            worker_refused = execute(
                intent_path,
                root / "worker-admission-refused",
                "disposable-CPU-test",
                worker_preflight=True,
            )
            self.assertEqual(record(worker_refused["latest"])["outcome"], "failed")
            self.assertIsNone(worker_refused["worker"])
            self.assertFalse((root / "worker-admission-refused/states/latest.json").exists())
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


class TestSupervisedResume(unittest.TestCase):
    def test_sealed_attempt_completes_and_resumes_in_its_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-loader-") as directory:
            root = Path(directory)
            _, intent, intent_path = scenario_intent(root)

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
            self.assertEqual(record(record(report["worker"])["accounting"])["observed_labels"], 30)
            exposures = [
                row for row in Journal(root / "attempt").events() if row.get("kind") == "exposure"
            ]
            self.assertEqual(
                [(row["anchors"], row["labels"]) for row in exposures], [([8, 9], [15, 15])]
            )
            self.assertIn("unknown", str(report["scientific"]))
            latest = Store(root / "attempt/states").latest(
                str(record(report["admission"])["recipe"])
            )
            self.assertEqual(latest.updates, 1)
            saved_latent = torch.load(latest.artifact.path, weights_only=True)["model"][
                "blocks.0.gate.latent_sign"
            ]
            self.assertLess(float((saved_latent - 1 / 3).abs().max()), 2e-3)
            intent["mode"] = "resume"
            intent["parent"] = reference(latest.artifact)
            intent["budget"] = {"updates": 3, "wall_seconds": 60, "attempts": 4, "retries": 0}
            intent["science_policy"] = {
                "id": "synthetic-fixture-only",
                "min_unique_positions": 16,
                "max_mean_visits": 8,
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


class TestIndexedLaunch(unittest.TestCase):
    def test_indexed_corpus_attempt_completes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="binary-loader-") as directory:
            root = Path(directory)
            loaded, intent, intent_path = scenario_intent(root)
            intent["science_policy"] = {
                "id": "synthetic-fixture-only",
                "min_unique_positions": 16,
                "max_mean_visits": 8,
            }
            index_ref, adapter = fixture_index(root, loaded.corpus)
            intent.update(corpus=reference(index_ref), input_adapter=adapter)
            intent_path.write_bytes(canonical(intent))
            indexed_report = execute(intent_path, root / "indexed-attempt", "disposable-CPU-test")
            self.assertEqual(
                record(indexed_report["latest"])["outcome"], "complete", indexed_report
            )
            self.assertEqual(record(indexed_report["worker"])["updates"], 1)

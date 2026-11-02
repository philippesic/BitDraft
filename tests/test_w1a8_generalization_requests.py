"""Real file pins and raw response joins protect the historical request gate."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization/requests.py"
SPEC = importlib.util.spec_from_file_location("generalization_requests", PATH)
requests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(requests)


class RequestGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def file(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        requests.write(path, value)
        return requests.pin(path)

    def fixture(self):
        target = self.file("target.gguf", {"target": 1})
        parent = self.file("parent.gguf", {"parent": 1})
        checkpoint = self.file("checkpoint.pt", {"state": "preserved"})
        model = self.file("model.gguf", {"export": 1})
        protocol = {
            "body_rows": 7,
            "draft_depth": 5,
            "markov": "full",
            "seed": 42,
            "target_kv": "F16",
            "target_sha256": target["sha256"],
            "target_weights": "F16",
            "target_width_mask": 3,
        }
        precision = {"activation_bits": 8, "tf32": False}
        base = {
            "parent": {"model": parent, "checkpoint": checkpoint},
            "scientific_protocol": protocol,
            "precision": precision,
        }
        spec = {
            "schema": "w1a8_generalization_v1",
            "base_spec": self.file("base.json", base),
            "policy": "occupancy",
            "horizon_updates": 19200,
            "retention_coefficient": 0.1,
            "scientific_protocol": protocol,
            "precision": precision,
            "sources": [],
        }
        finish = {
            "status": "EXPOSURE_COMPLETE",
            "local_updates": 1920,
            "additional_TRAIN_seconds": 200.0,
            "spec": self.file("spec.json", spec),
            "checkpoint": checkpoint,
            "parent": base["parent"],
            "native_evaluation_required": True,
            "backlog_screening_started": False,
            "export": {
                "output": model,
                "parent": parent,
                "serialization_audit_passed": True,
                "typed_metadata_unchanged": True,
                "all_other_payloads_unchanged": True,
                "permitted_changed_tensors": [str(i) for i in range(48)],
                "actual_changed_tensors": ["0"],
            },
        }
        baseline = {"target": target, "models": {"parent3504": parent}}
        return finish, spec, baseline

    def validate(self, finish, baseline):
        path = self.root / "finish.json"
        path.write_text(json.dumps(finish))
        return requests.validate_candidate(path, 1920, baseline)

    def test_admits_completed_native_export_and_preserved_checkpoint(self):
        finish, _, baseline = self.fixture()
        result = self.validate(finish, baseline)
        self.assertEqual(result["model"], finish["export"]["output"])
        self.assertEqual(result["local_updates"], 1920)

    def test_rejects_wrong_model_bytes_and_native_export_scope(self):
        finish, _, baseline = self.fixture()
        Path(finish["export"]["output"]["path"]).write_text("wrong native model")
        with self.assertRaisesRegex(ValueError, "input pin changed"):
            self.validate(finish, baseline)
        finish["export"]["output"] = requests.pin(finish["export"]["output"]["path"])
        finish["export"]["actual_changed_tensors"] = ["unexpected_head"]
        with self.assertRaisesRegex(ValueError, "tensor scope"):
            self.validate(finish, baseline)

    def test_rejects_shortened_paused_and_unmeasured_exposure(self):
        finish, _, baseline = self.fixture()
        for key, value in [
            ("local_updates", 96),
            ("status", "PAUSED"),
            ("additional_TRAIN_seconds", 0),
            ("additional_TRAIN_seconds", float("nan")),
        ]:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(finish)
                changed[key] = value
                with self.assertRaises(ValueError):
                    self.validate(changed, baseline)

    def test_rejects_changed_spec_pin_and_matching_but_wrong_protocol(self):
        finish, spec, baseline = self.fixture()
        spec["scientific_protocol"] = {**spec["scientific_protocol"], "draft_depth": 7}
        Path(finish["spec"]["path"]).write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, "input pin changed"):
            self.validate(finish, baseline)
        finish["spec"] = requests.pin(finish["spec"]["path"])
        with self.assertRaisesRegex(ValueError, "protocol/precision changed"):
            self.validate(finish, baseline)

    def test_frozen_configuration_preserves_all_controls_and_settings(self):
        baseline = {
            "models": {name: {"path": name} for name in requests.CONTROLS},
            "panels": {"development": {}},
            "repetitions": 5,
            "prompt_limit": 24,
            "mode": "claim",
            "balanced_screen": False,
            "port": 18297,
        }
        config = copy.deepcopy(baseline)
        config["models"]["occupancy"] = {"path": "new"}
        requests.validate_config(config, baseline)
        for key in ["port", "control"]:
            wrong = copy.deepcopy(config)
            if key == "port":
                wrong["port"] += 1
            else:
                wrong["models"]["parent3504"]["path"] = "substitute"
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                requests.validate_config(wrong, baseline)

    def response(
        self,
        text="answer",
        stop="stop",
        tokens=None,
        truncated=False,
        stop_type="eos",
        stopping_word="",
    ):
        return {
            "choices": [{"message": {"content": text}, "finish_reason": stop}],
            "usage": {"completion_tokens": 2},
            "__verbose": {
                "tokens": tokens or [2, 3],
                "truncated": truncated,
                "stop_type": stop_type,
                "stopping_word": stopping_word,
            },
        }

    def audit_fixture(self, changed=None):
        config = {"models": {"target_only": {}, "candidate": {}}}
        rows = []
        for arm in config["models"]:
            for rep in range(5):
                response = self.response()
                if arm == "candidate" and rep == 3 and changed:
                    response = self.response(**changed)
                ids, digest, stop = requests.raw_signature(response)[:3]
                cell = self.root / "development" / f"rep-{rep:02d}" / arm
                self.file(str(cell.relative_to(self.root) / "prompt-0000/response.json"), response)
                self.file(
                    str(cell.relative_to(self.root) / "release.json"), {"owned_server_reaped": True}
                )
                rows.append(
                    {
                        "panel": "development",
                        "arm": arm,
                        "repetition": rep,
                        "prompt_id": "one",
                        "result": {
                            "generated_token_ids": ids,
                            "completion_sha256": digest,
                            "finish_reason": stop,
                            "completion_tokens": 2,
                        },
                    }
                )
        (self.root / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.file("summary.json", {"status": "COMPLETE", "diagnostic": False})
        return config, rows

    def test_raw_token_text_stop_join_and_repeated_target_comparisons(self):
        config, _ = self.audit_fixture()
        report = requests.audit(self.root, config, ["one"])
        self.assertEqual(report["status"], "TOKEN_TEXT_STOP_PARITY_PASS")
        self.assertEqual(report["requests"], 10)

    def test_rejects_text_stop_and_token_mismatches_even_with_matching_metrics(self):
        for changed in [
            {"text": "wrong answer"},
            {"stop": "length"},
            {"tokens": [3, 2]},
            {"truncated": True},
            {"stop_type": "limit"},
            {"stopping_word": "END"},
        ]:
            with self.subTest(changed=changed):
                with tempfile.TemporaryDirectory() as temp:
                    old = self.root
                    self.root = Path(temp)
                    config, _ = self.audit_fixture(changed)
                    with self.assertRaisesRegex(ValueError, "current target token/text/stop"):
                        requests.audit(self.root, config, ["one"])
                    self.root = old

    def test_rejects_raw_response_metrics_disagreement_and_missing_requests(self):
        config, rows = self.audit_fixture()
        response_path = self.root / "development/rep-00/candidate/prompt-0000/response.json"
        response_path.write_text(json.dumps(self.response(text="changed raw response")))
        with self.assertRaisesRegex(ValueError, "raw response and recorded"):
            requests.audit(self.root, config, ["one"])
        response_path.write_text(json.dumps(self.response()))
        (self.root / "requests.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows[:-1])
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            requests.audit(self.root, config, ["one"])


if __name__ == "__main__":
    unittest.main()

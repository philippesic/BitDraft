"""Temporary synthetic packets only; no real confirmation cohort, models or CUDA."""

import argparse
import copy
import importlib.util
import json
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("modern_confirm", SCRIPTS / "modern_confirm.py")
confirm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(confirm)


class Judge:
    def __init__(self):
        self.calls = []

    def judge(self, case, text):
        self.calls.append((case["id"], text))
        return {"passed": text == "GOOD", "reason": "synthetic oracle"}


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-confirm-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "out"
        self.output.mkdir()
        self.judge = Judge()
        opaque = {
            "path": str(self.root / "NEVER-OPEN-model.gguf"),
            "sha256": "a" * 64,
            "bytes": 100,
        }
        runtime = {
            "server": {**opaque, "path": str(self.root / "NEVER-OPEN-server")},
            "libraries": [opaque],
            "environment": {"mode": "synthetic"},
            "source_pins": [opaque],
        }
        self.models = {
            "target": opaque,
            "baseline": opaque,
            "arms": [
                {
                    "id": "target",
                    "kind": "target",
                    "model": None,
                    "runtime": copy.deepcopy(runtime),
                },
                {
                    "id": "dflash",
                    "kind": "dflash",
                    "model": opaque,
                    "runtime": copy.deepcopy(runtime),
                },
                {
                    "id": "initial",
                    "kind": "w1a8",
                    "model": opaque,
                    "runtime": copy.deepcopy(runtime),
                },
                {
                    "id": "selected",
                    "kind": "w1a8",
                    "model": opaque,
                    "runtime": copy.deepcopy(runtime),
                },
            ],
        }
        source = self.put("source.json", self.models)
        self.selection = {
            "candidate": opaque,
            "runtime": runtime["server"],
            "source_freeze": source,
            "generator": confirm.pin(SCRIPTS / "modern_fresh.py"),
            "judge": confirm.pin(SCRIPTS / "modern_fresh.py"),
            "settings": {"output_limit": 256},
            "selected_arm_id": "selected",
            "exclusions": [],
        }
        selection = self.put("selection.json", self.selection)
        seed = confirm.fresh.digest(
            Path(selection["path"]).read_bytes() + b"modern4b-confirmation-v1"
        )
        groups, requests, cases, controls = [], [], [], []
        for domain, families in confirm.fresh.FAMILIES.items():
            for family in families:
                for phase in ("performance", "quality"):
                    for i in range(2):
                        text = f"synthetic fixture {phase} {family} {i}"
                        cid = f"{phase}-{family}-{i}"
                        meta = {
                            "id": cid,
                            "domain": domain,
                            "family": family,
                            "phase": phase,
                            "group_id": cid,
                            "source": "handwritten-test-fixture",
                            "instance_seed": cid,
                            "identities": confirm.fresh.identities(text),
                        }
                        payload = self.payload(confirm.fresh.render(text), 256)
                        groups.append(meta)
                        if phase == "performance":
                            requests.append(
                                {
                                    **meta,
                                    "prompt": payload["prompt"],
                                    "sha256": confirm.fresh.digest(payload["prompt"].encode()),
                                    "request": payload,
                                }
                            )
                        else:
                            cases.append(
                                {
                                    **meta,
                                    "prompt": text,
                                    "request": payload,
                                    "kind": "json",
                                    "expected": {"private_oracle": 1},
                                }
                            )
                            controls.append(
                                {
                                    "case_id": cid,
                                    "correct_control_passed": True,
                                    "known_wrong_control_rejected": True,
                                    "positive": {"answer": "GOOD", "judgement": {"passed": True}},
                                    "semantic_negatives": [
                                        {"answer": answer, "judgement": {"passed": False}}
                                        for answer in ("BAD0", "BAD1")
                                    ],
                                }
                            )
        manifest = {
            "schema": "modern_fresh_confirmation_inputs_v1",
            "status": "PREPARED_NO_MODEL_EVALUATION",
            "admitted": True,
            "freeze": selection,
            "seed": seed,
            "exclusions": [],
            "groups": groups,
            "overlap_case_ids": [],
            "model_requests_executed": 0,
            "tokenization_performed": False,
            **{
                k: self.selection[k]
                for k in ("candidate", "runtime", "source_freeze", "generator", "judge", "settings")
            },
        }
        self.inputs = {
            "selection": selection,
            "manifest": self.put("manifest.json", manifest),
            "requests": self.put("requests.json", requests),
            "quality_cases": self.put("quality-cases.json", cases),
            "quality_controls": self.put("quality-controls.json", controls),
        }
        receipt = {
            "schema": "modern_fresh_confirmation_receipt_v1",
            "status": "PREPARED_NO_MODEL_EVALUATION",
            "admitted": True,
            "freeze": selection,
            "seed": seed,
            "model_requests_executed_before_freeze": 0,
            "model_requests_executed": 0,
            "input_overlap_count": 0,
            "performance_prompts": 24,
            "quality_tasks": 24,
            "quality_templates": 12,
            "quality_correct_controls_passed": 24,
            "quality_known_wrong_controls_rejected": 24,
        }
        self.inputs["receipt"] = self.put("receipt.json", receipt)
        self.refresh_receipt()
        warmups = []
        old_rows = []
        for i in range(2):
            text = "old fit fixture " + str(i)
            warmups.append(
                {
                    "id": "fit-fixture-" + str(i),
                    "role": "fit",
                    "request": self.put(f"old-request-{i}.json", self.payload(text, 80)),
                    "reference": self.put(f"old-response-{i}.json", self.response()),
                }
            )
            old_rows.append(
                {
                    "id": warmups[-1]["id"],
                    "role": "fit",
                    "rendered_prompt_sha256": confirm.fresh.digest(text.encode()),
                }
            )
        old_roster = self.put(
            "old-roster.json",
            {"schema": "w1a8_modern_phase1_generalization_roster_v2", "rows": old_rows},
        )
        self.inputs["warmups"] = warmups
        self.inputs["warmup_source"] = self.put(
            "old-inputs.json", {"roster": old_roster, "warmups": warmups}
        )

    def put(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return confirm.pin(path)

    def refresh_receipt(self):
        receipt = confirm.read(self.inputs["receipt"]["path"])
        receipt["files"] = {
            filename: self.inputs[key]["sha256"] for key, filename in confirm.PACKET_FILES.items()
        }
        receipt["manifest_sha256"] = self.inputs["manifest"]["sha256"]
        self.inputs["receipt"] = self.put("receipt.json", receipt)

    def runner(self, allow=True):
        args = argparse.Namespace(
            models=self.root / "models.json",
            inputs=self.root / "inputs.json",
            output=self.output,
            repetitions=5,
            allow_cuda=allow,
        )
        runner = confirm.Confirmation(args)
        runner.inputs = copy.deepcopy(self.inputs)
        runner.models = copy.deepcopy(self.models)
        runner.arms = runner.models["arms"]
        runner.records = []
        runner.guard = Mock()
        return runner

    def authenticate(self, runner):
        with (
            patch.object(
                confirm.fresh,
                "validate_freeze",
                return_value=(self.selection, self.inputs["selection"], self.judge),
            ),
            patch.object(confirm.fresh, "load_exclusions", return_value=(set(), [])),
        ):
            runner.authenticate_inputs()

    @staticmethod
    def payload(prompt, count):
        return {
            "prompt": prompt,
            "n_predict": count,
            "temperature": 0,
            "seed": 1234,
            "cache_prompt": False,
            "return_tokens": True,
            "stream": False,
        }

    @staticmethod
    def response():
        return {
            "tokens": [123],
            "content": "GOOD",
            "stop_type": "eos",
            "stopping_word": "",
            "truncated": False,
            "stop": True,
            "tokens_predicted": 1,
            "timings": {
                "draft_n": 2,
                "draft_n_accepted": 1,
                "predicted_n": 1,
                "cache_n": 0,
                "prompt_n": 3,
                "prompt_ms": 1.0,
                "predicted_ms": 2.0,
            },
        }

    def test_invalid_native_protocol_cannot_create_reference(self):
        payload = self.payload("fixture", 256)
        changes = [
            lambda d: d.pop("stop"),
            lambda d: d.update(tokens_predicted=2),
            lambda d: d["timings"].update(predicted_n=2),
            lambda d: d["timings"].update(cache_n=1),
            lambda d: d["timings"].update(prompt_n=0),
            lambda d: d["timings"].update(predicted_ms=float("nan")),
        ]
        for change in changes:
            response = self.response()
            change(response)
            with self.assertRaises(ValueError):
                confirm.validate_response(response, payload)
        empty = self.response()
        empty.update(tokens=[], content="", tokens_predicted=0)
        empty["timings"].update(predicted_n=0, predicted_ms=0)
        self.assertEqual(confirm.validate_response(empty, payload)["tokens"], [])

    def test_requires_operator_authorization_before_base_auth(self):
        runner = self.runner(allow=False)
        with self.assertRaisesRegex(ValueError, "allow-cuda"):
            runner.authenticate()

    def test_admission_rechecks_all_controls_before_inference(self):
        runner = self.runner()
        runner.request = Mock(side_effect=AssertionError("no serving during admission"))
        self.authenticate(runner)
        self.assertEqual(len(self.judge.calls), 72)
        self.assertEqual(len(runner.rows), 24)
        self.assertEqual(len(runner.quality_rows), 24)
        self.assertEqual(set(runner.gold), {w["id"] for w in runner.warmups})
        self.assertIsNone(runner.geometry)
        self.assertGreaterEqual(
            confirm.read(self.output / "authenticated-controls.json")["cpu_wall_seconds"], 0
        )
        runner.request.assert_not_called()

    def test_actual_generator_packet_admitted_with_actual_trusted_judge(self):
        exclusions = []
        for role in sorted(confirm.fresh.ROLES):
            ref = self.put(
                "integration-excluded-" + role + ".json",
                {
                    "schema": "modern_fresh_exclusion_index_v1",
                    "records": [confirm.fresh.identities("test-only prior synthetic " + role)],
                },
            )
            exclusions.append(
                {
                    "role": role,
                    "format": "hash_index_v1",
                    "file": ref,
                    "previously_used": True,
                    "sealed": False,
                    "complete": True,
                }
            )
        selection = {
            **self.selection,
            "schema": confirm.fresh.SCHEMA,
            "selection_frozen": True,
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "evaluation_inputs_generated": False,
            "scope": "TEST ONLY: temporary fake model/source pins; never a project confirmation cohort",
            "exclusions": exclusions,
            "judge": confirm.fresh.reference(confirm.fresh.TRUSTED_JUDGE),
        }
        selection_ref = self.put("integration-selection.json", selection)
        generated = self.root / "integration-generated-synthetic"
        runner = self.runner()
        runner.request = Mock(side_effect=AssertionError("no model/network requests permitted"))
        original_load = confirm.fresh.load_judge
        oracle_calls = []

        def load_actual_counted(ref):
            module = original_load(ref)
            actual = module.judge

            def judge(case, text):
                oracle_calls.append(case["id"])
                return actual(case, text)

            module.judge = judge
            return module

        # Source and metadata access is real. Fake model/runtime paths must never
        # be opened or statted. Only the trusted judge's isolated CPU workers may
        # spawn; neither a server nor a network connection is permitted.
        original_open, original_stat, original_popen = Path.open, Path.stat, subprocess.Popen
        forbidden_accesses = []

        def checked_open(path, *args, **kwargs):
            if "NEVER-OPEN" in str(path):
                forbidden_accesses.append(str(path))
                raise AssertionError("opaque model/runtime accessed")
            return original_open(path, *args, **kwargs)

        def checked_stat(path, *args, **kwargs):
            if "NEVER-OPEN" in str(path):
                forbidden_accesses.append(str(path))
                raise AssertionError("opaque model/runtime statted")
            return original_stat(path, *args, **kwargs)

        worker_commands = []

        def cpu_worker_only(argv, *args, **kwargs):
            self.assertEqual(argv[:4], [sys.executable, "-I", "-S", "-B"])
            self.assertEqual(Path(argv[4]).resolve(), confirm.fresh.TRUSTED_JUDGE.resolve())
            self.assertEqual(argv[5], "--worker")
            self.assertIn(argv[6], ("python", "sql"))
            self.assertEqual(len(argv), 7)
            worker_commands.append(argv)
            return original_popen(argv, *args, **kwargs)

        with (
            patch.object(confirm.fresh, "load_judge", side_effect=load_actual_counted),
            patch.object(Path, "open", checked_open),
            patch.object(Path, "stat", checked_stat),
            patch.object(subprocess, "Popen", side_effect=cpu_worker_only),
            patch.object(socket, "socket", side_effect=AssertionError("network forbidden")),
        ):
            receipt = confirm.fresh.prepare(Path(selection_ref["path"]), generated)
            generated_calls = len(oracle_calls)
            self.assertTrue(receipt["admitted"])
            self.assertEqual(receipt["quality_correct_controls_passed"], 24)
            self.assertEqual(receipt["quality_known_wrong_controls_rejected"], 24)
            self.assertGreater(generated_calls, 72)  # SQL has three semantic negatives.
            runner.inputs = {
                key: confirm.pin(generated / filename)
                for key, filename in confirm.PACKET_FILES.items()
            }
            runner.inputs.update(
                selection=selection_ref,
                receipt=confirm.pin(generated / "receipt.json"),
                warmups=copy.deepcopy(self.inputs["warmups"]),
                warmup_source=self.inputs["warmup_source"],
            )
            runner.authenticate_inputs()  # Real generator/receipt/judge integration, no base model auth.
        self.assertEqual(len(oracle_calls), 2 * generated_calls)
        self.assertEqual(len(set(oracle_calls[generated_calls:])), 24)
        self.assertEqual(len(runner.rows), 24)
        self.assertEqual(len(runner.quality_rows), 24)
        controls = confirm.read(self.output / "authenticated-controls.json")
        self.assertEqual(len(controls["records"]), 24)
        self.assertTrue(all(r["positive"]["passed"] for r in controls["records"]))
        self.assertTrue(all(not n["passed"] for r in controls["records"] for n in r["negatives"]))
        self.assertTrue(any(len(r["negatives"]) == 3 for r in controls["records"]))
        self.assertTrue(worker_commands)
        self.assertEqual(forbidden_accesses, [])
        runner.request.assert_not_called()

    def test_forged_control_flags_do_not_override_wrong_positive(self):
        controls = confirm.read(self.inputs["quality_controls"]["path"])
        controls[0]["positive"]["answer"] = "BAD0"
        self.inputs["quality_controls"] = self.put("quality-controls.json", controls)
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError, "no longer discriminate"):
            self.authenticate(self.runner())

    def test_corrupt_packet_pin_and_inconclusive_packet_reject(self):
        runner = self.runner()
        runner.inputs["requests"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pin changed"):
            self.authenticate(runner)
        manifest = confirm.read(self.inputs["manifest"]["path"])
        manifest["admitted"] = False
        self.inputs["manifest"] = self.put("manifest.json", manifest)
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError, "admitted"):
            self.authenticate(self.runner())

    def test_selected_full_runtime_and_model_cannot_change(self):
        for field in ("environment", "libraries", "model"):
            with self.subTest(field=field):
                runner = self.runner()
                selected = runner.arms[-1]
                if field == "environment":
                    selected["runtime"][field]["mode"] = "other"
                elif field == "libraries":
                    selected["runtime"][field] = []
                else:
                    selected[field] = {**selected[field], "sha256": "b" * 64}
                with self.assertRaisesRegex(ValueError, "changed after selection"):
                    self.authenticate(runner)

    def test_warmup_role_flag_cannot_override_original_fit_binding(self):
        runner = self.runner()
        runner.inputs["warmups"][0]["reference"] = runner.inputs["warmups"][1]["reference"]
        with self.assertRaisesRegex(ValueError, "warmup ancestry"):
            self.authenticate(runner)

    def test_tokenization_freezes_all48_without_private_oracles(self):
        runner = self.runner()
        self.authenticate(runner)
        runner.request = Mock(return_value={"tokens": [1, 2, 3]})
        cell = self.output / "token-cell"
        cell.mkdir()
        runner.prepare_cell(123, cell, runner.arms[0], 0)
        self.assertEqual(runner.request.call_count, 48)
        for call in runner.request.call_args_list:
            self.assertEqual(call.args[1], "tokenize")
            self.assertEqual(
                set(call.args[2]), {"content", "add_special", "parse_special", "with_pieces"}
            )
            self.assertTrue(call.args[2]["add_special"])
            self.assertNotIn("private_oracle", json.dumps(call.args[2]))
        self.assertTrue(confirm.read(self.output / "native-geometry.json")["all_tasks_fit"])

    def test_one_overflow_preserves_all_geometry_and_rejects_whole_packet(self):
        runner = self.runner()
        self.authenticate(runner)
        responses = [{"tokens": [1] * (4096 - 256 + 1)}] + [{"tokens": [1]}] * 47
        runner.request = Mock(side_effect=responses)
        cell = self.output / "token-cell"
        cell.mkdir()
        with self.assertRaisesRegex(ValueError, "no truncation/drop"):
            runner.prepare_cell(123, cell, runner.arms[0], 0)
        geometry = confirm.read(self.output / "native-geometry.json")
        self.assertEqual(len(geometry["rows"]), 48)
        self.assertFalse(geometry["all_tasks_fit"])
        self.assertIsNone(runner.geometry)

    def test_target_first_gold_phase_and_completion_privacy(self):
        runner = self.runner()
        self.authenticate(runner)
        runner.geometry = [{"id": runner.rows[0]["id"], "prompt_tokens": 3, "fits": True}]
        target_response = self.response()
        target_response["timings"].update(draft_n=0, draft_n_accepted=0)
        runner.request = Mock(return_value=target_response)
        item = runner.rows[0]
        cell = self.output / "completion-cell"
        cell.mkdir()
        with self.assertRaisesRegex(ValueError, "only first frozen target"):
            runner.completion(123, item, cell, 0, False, runner.arms[-1], 0)
        runner.request.assert_not_called()
        result = runner.completion(123, item, cell, 0, False, runner.arms[0], 0)
        self.assertTrue(result["target_gold_created"])
        self.assertEqual(result["phase"], "performance")
        payload = runner.request.call_args.args[2]
        self.assertEqual(set(payload), confirm.PAYLOAD_KEYS)
        self.assertNotIn("private_oracle", json.dumps(payload))
        other = self.output / "other-cell"
        other.mkdir()
        runner.request = Mock(return_value={**self.response(), "tokens": [124]})
        with self.assertRaisesRegex(ValueError, "parity failed"):
            runner.completion(123, item, other, 0, False, runner.arms[-1], 0)
        self.assertEqual(runner.gold[item["id"]]["tokens"], [123])

    def test_stop_interrupts_blocked_http_and_reaps_owned_process(self):
        runner = self.runner()
        runner.permission = Mock()
        runner.stopping = False
        source = (
            "import time\nfrom http.server import HTTPServer,BaseHTTPRequestHandler\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_POST(self): time.sleep(300)\n"
            " def log_message(self,*args): pass\n"
            "server=HTTPServer(('127.0.0.1',0),Handler)\n"
            "print(server.server_port,flush=True)\nserver.serve_forever()\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        runner.active = process
        requests_module = sys.modules["modern_requests"]
        runner.active_identity = requests_module.process_identity(process)
        previous = signal.signal(signal.SIGALRM, runner.request_stop)
        try:
            port = int(process.stdout.readline())
            signal.setitimer(signal.ITIMER_REAL, 0.05)
            with self.assertRaises(requests_module.StopRequested):
                runner.request(port, "completion", {"fixture": "no models"}, timeout=90)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            runner.reap(self.output / "cpu-server-release.json")
            process.stdout.close()
        receipt = confirm.read(self.output / "cpu-server-release.json")
        self.assertTrue(receipt["owned_process_reaped"])
        self.assertTrue(receipt["owned_process_group_absent"])
        self.assertTrue(runner.stopping)

    def test_speculative_counter_drift_rejected(self):
        for proposed, accepted in [(None, None), (-1, 0), (1, 2), (2, 2)]:
            response = self.response()
            response["timings"].update(draft_n=proposed, draft_n_accepted=accepted)
            with self.assertRaises(ValueError):
                confirm.draft_counts(response, "w1a8")

    def test_five_performance_rounds_then_quality_once_and_judge_after_serving(self):
        runner = self.runner()
        runner.authenticate = lambda: self.authenticate(runner)
        response = self.put("synthetic-served-response.json", self.response())
        stages = []

        def cell(rep, arm):
            # Only admission controls may have run while any serving is ongoing.
            self.assertEqual(len(self.judge.calls), 72)
            stages.append((runner.current_phase, rep, arm["id"]))
            for item in runner.rows:
                runner.records.append(
                    {
                        "arm": arm["id"],
                        "phase": runner.current_phase,
                        "repetition": rep,
                        "id": item["id"],
                        "warmup": False,
                        "completion_tokens": 1,
                        "full_http_wall_seconds": 0.01,
                        "draft_proposed": 2,
                        "draft_accepted": 1,
                        "response": response,
                    }
                )
            runner.records.extend(
                {"arm": arm["id"], "phase": "warmup", "warmup": True} for _ in range(2)
            )
            if not (self.output / "native-geometry.json").exists():
                confirm.atomic(self.output / "native-geometry.json", {"synthetic": True})

        runner.cell = cell
        with patch.object(confirm.signal, "signal"):
            runner.run()
        result = confirm.read(self.output / "result.json")
        self.assertEqual(result["performance_requests"], 480)
        self.assertEqual(result["quality_requests"], 96)
        self.assertEqual(result["excluded_warmups"], 48)
        self.assertEqual(result["completed_startups"], 24)
        expected = runner.freeze_counts()
        self.assertEqual(expected["expected_startups"], len(stages))
        self.assertEqual(expected["expected_excluded_warmups"], result["excluded_warmups"])
        self.assertEqual(expected["expected_quality_requests"], result["quality_requests"])
        self.assertTrue(all(s[0] == "performance" for s in stages[:20]))
        self.assertTrue(all(s[0] == "quality" and s[1] == 5 for s in stages[20:]))
        self.assertEqual(stages[0][2], "target")
        self.assertEqual(stages[4][2], "selected")  # Odd repetitions reverse arms.
        self.assertEqual(stages[20][2], "target")
        self.assertEqual(len(self.judge.calls), 72 + 96)
        self.assertTrue(all(q["passed"] == 24 for q in result["independent_quality"]))


if __name__ == "__main__":
    unittest.main()

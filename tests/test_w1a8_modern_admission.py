"""Synthetic CPU joins, never evidence of native capture or model fidelity."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization/modern_admission.py"
SPEC = importlib.util.spec_from_file_location("modern_admission", PATH)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class NativeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.packet = self.fixture()
        self.addCleanup(patch.stopall)
        patch.object(gate, "PROJECT", self.root).start()
        patch.object(gate, "TRUSTED_BASELINE", self.packet["baseline"]).start()

    def pin(self, path):
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    def file(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) if not isinstance(data, str) else data)
        return self.pin(path)

    def array(self, name, array):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        array.astype("<f4").tofile(path)
        return self.pin(path)

    def rewrite(self, ref, change):
        path = Path(ref["path"])
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))
        ref.clear()
        ref.update(self.pin(path))

    def fixture(self):
        target = self.file("target.gguf", "synthetic immutable target")
        cuda = self.file("runtime/libggml-cuda.so.0.28.0", "frozen CUDA fixture, not executable")
        libraries = [
            self.file("runtime/" + name, "frozen " + name)
            for name in [
                "libllama.so.0.8.0",
                "libggml.so.0.28.0",
                "libggml-base.so.0.28.0",
                "libggml-cpu.so.0.28.0",
            ]
        ]
        server = self.file("runtime/llama-server", "server fixture")
        frozen = self.file(
            "frozen/final-freeze.json", {"server": server, "cuda": cuda, "pins": libraries + [cuda]}
        )
        baseline = self.file(
            "baseline.json",
            {
                "cuda": cuda,
                "sources": [frozen],
                "rows": [{"target": gate.TARGET_ID, "target_pin": target}],
            },
        )
        helper = self.file("build/modern_capture", "native helper fixture")
        env = {"GGML_CUDA_DECODE_CONSISTENT": "32", "LD_LIBRARY_PATH": str(self.root / "runtime")}
        build = self.file(
            "build/build.json",
            {
                "status": "compiled_linked_cpu_checks_only",
                "baseline": baseline,
                "environment": env,
                "source": self.file("build/source/capture.cpp", "source snapshot"),
                "builder": self.file("build/source/build.py", "builder snapshot"),
                "compiler": self.file("compiler", "compiler fixture"),
                "executable": helper,
                "runtime_sources": [self.file("runtime/source.cpp", "runtime source")],
                "included_local_headers": [self.file("runtime/header.h", "header")],
                "runtime_libraries": libraries,
                "cuda_overlay": cuda,
                "source_origins": {
                    "capture": {"path": "/retired-worktree/no-longer-exists", "sha256": "f" * 64}
                },
            },
        )
        raw = self.file("prompt.txt", "synthetic native prompt")
        ids = self.file("tokens/tokens.ids", "1\n2\n")
        token_meta = self.file(
            "tokens/tokenization.json",
            {
                "status": "native_tokenization_complete",
                "model": target["path"],
                "model_bytes": target["bytes"],
                "text_file": raw["path"],
                "text_bytes": raw["bytes"],
                "token_count": 2,
                "token_ids_file": "tokens.ids",
                "add_special": True,
                "parse_special": True,
                "chat_template_applied": False,
                "vocab_only": True,
                "backend_initialized": False,
                "weights_loaded": False,
            },
        )
        token_command = self.file(
            "token-command.json",
            {
                "argv": [
                    helper["path"],
                    "--tokenize",
                    "--model",
                    target["path"],
                    "--text-file",
                    raw["path"],
                    "--out",
                    str(self.root / "tokens"),
                ],
                "environment": env,
            },
        )
        fixed = self.file("continuation.ids", "1\n")
        logits = np.zeros((2, gate.VOCAB), dtype=np.float32)
        logits[0, 1] = 2
        logits[1, 5], logits[1, 4] = 1, 0.95
        arrays = {"logits.f32": self.array("capture/logits.f32", logits)}
        for tap in [0, 2, 16, 29]:
            values = np.ones((3, gate.WIDTH), dtype=np.float32) * 0.000123456
            values[1] *= 2
            arrays[f"layer-{tap}.f32"] = self.array(f"capture/layer-{tap}.f32", values)
        capture = self.file(
            "capture/capture.json",
            {
                "status": "native_capture_complete",
                "model": target["path"],
                "model_bytes": target["bytes"],
                "dtype": "little_endian_float32",
                "feature_semantics": "zero_based_pre_block_input_all_tokens",
                "continuation_mode": "fixed_token_ids",
                "taps": [0, 2, 16, 29],
                "settings": gate.SETTINGS,
                "cpu_library": libraries[-1]["path"],
                "cuda_library": cuda["path"],
                "gpu_description": "NVIDIA GeForce RTX 5080",
                "input_tokens": [1, 2, 1],
                "prompt_tokens": 2,
                "feature_shape": [3, gate.WIDTH],
                "logits_shape": [2, gate.VOCAB],
                "logit_positions": [1, 2],
                "greedy_ids": [1, 5],
                "prefill_chunk": 512,
                "decode_chunk": 1,
                "output_cap": 3,
                "threads": 4,
                "array_bytes": sum(x["bytes"] for x in arrays.values()),
            },
        )
        capture_command = self.file(
            "capture-command.json",
            {
                "argv": [
                    helper["path"],
                    "--model",
                    target["path"],
                    "--tokens",
                    ids["path"],
                    "--out",
                    str(self.root / "capture"),
                    "--cpu-library",
                    libraries[-1]["path"],
                    "--cuda-library",
                    cuda["path"],
                    "--decode-tokens",
                    fixed["path"],
                    "--prefill-chunk",
                    "512",
                    "--decode-chunk",
                    "1",
                    "--output-cap",
                    "3",
                    "--threads",
                    "4",
                    "--taps",
                    "0,2,16,29",
                    "--max-output-mib",
                    "128",
                ],
                "environment": env,
            },
        )
        maps = self.file(
            "maps.txt", "\n".join("0-1 r-xp 0 00:00 1 " + x["path"] for x in libraries + [cuda])
        )
        server_command = [
            server["path"],
            "-m",
            target["path"],
            "-ngl",
            "99",
            "-c",
            "4096",
            "-np",
            "1",
            "-b",
            "512",
            "-ub",
            "512",
            "-t",
            "4",
            "-fa",
            "on",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--host",
            "127.0.0.1",
            "--port",
            "12345",
            "--metrics",
            "--no-webui",
            "--no-warmup",
            "--spec-type",
            "none",
        ]
        freeze = self.file(
            "serving/freeze.json",
            {
                "source": self.file("serving/probe.py", "server launcher"),
                "server": server,
                "target": target,
                "cuda": cuda,
                "command_ancestry": self.file("serving/old-command.json", {}),
                "command": server_command,
                "environment": env,
            },
        )
        request = self.file(
            "serving/request.json",
            {
                "prompt": "synthetic native prompt",
                "n_predict": 2,
                "temperature": 0,
                "seed": 1234,
                "cache_prompt": False,
                "return_tokens": True,
                "stream": False,
            },
        )
        response = self.file(
            "serving/response.json",
            {
                "tokens": [1, 4],
                "tokens_predicted": 2,
                "tokens_evaluated": 2,
                "prompt": "synthetic native prompt",
                "model": target["path"],
                "truncated": False,
                "stop_type": "limit",
                "stopping_word": "",
                "generation_settings": {
                    "temperature": 0,
                    "seed": 1234,
                    "repeat_penalty": 1,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "dry_multiplier": 0,
                    "logit_bias": [],
                    "grammar": "",
                    "lora": [],
                    "n_predict": 2,
                    "ignore_eos": False,
                    "backend_sampling": False,
                },
            },
        )
        return {
            "schema": "w1a8_modern_capture_packet_v1",
            "baseline": baseline,
            "target": target,
            "build": build,
            "capture": capture,
            "capture_command": capture_command,
            "capture_library_maps": maps,
            "continuation_tokens": fixed,
            "arrays": arrays,
            "tokenizer": {
                "build": build,
                "metadata": token_meta,
                "command": token_command,
                "tokens": ids,
                "text": raw,
            },
            "serving": {
                "freeze": freeze,
                "request": request,
                "response": response,
                "library_maps": maps,
            },
        }

    def test_exact_join_excludes_disagreeing_row_without_claiming_parent_success(self):
        result = gate.admit_packet(self.packet)
        self.assertEqual(result["eligible_row_ids"], [0])
        self.assertEqual(result["excluded_row_ids"], [1])
        self.assertEqual(result["rows"][0]["input_position"], 1)
        self.assertEqual(result["rows"][0]["serving_token_index"], 0)
        self.assertEqual(result["rows"][1]["serving_token_index"], 1)
        self.assertEqual(result["rows"][1]["serving_label"], 4)
        self.assertEqual(result["rows"][1]["native_argmax"], 5)
        self.assertFalse(result["parent_success_protection_proven"])
        self.assertFalse(any(x["parent_success_protection_proven"] for x in result["rows"]))
        self.assertEqual(result["native_embedding_first_row_by_token"], {"1": 0, "2": 1})
        self.assertEqual(
            len(gate.require_eligible_rows(result, [0], purpose="protection_numeric")), 1
        )

    def test_no_caller_or_receipt_override_can_enable_a_disagreeing_row(self):
        packet = copy.deepcopy(self.packet)
        packet["protection_row_override"] = [1]
        with self.assertRaisesRegex(ValueError, "overrides"):
            gate.admit_packet(packet)
        result = gate.admit_packet(self.packet)
        for purpose in ["supervision", "protection_numeric"]:
            with self.assertRaisesRegex(ValueError, "excluded row"):
                gate.require_eligible_rows(result, [1], purpose=purpose)
        result["rows"][1]["supervision_eligible"] = True
        result["rows"][1]["protection_numeric_eligible"] = True
        result["eligible_row_ids"].append(1)
        result["excluded_row_ids"].clear()
        with self.assertRaisesRegex(ValueError, "excluded row"):
            gate.require_eligible_rows(result, [1], purpose="protection_numeric")

    def test_changed_pin_is_rejected_before_numeric_use(self):
        Path(self.packet["arrays"]["logits.f32"]["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "pinned size|file pin changed"):
            gate.admit_packet(self.packet)

    def test_tokenizer_may_explicitly_hide_cuda_but_no_other_environment_changes(self):
        ref = self.packet["tokenizer"]["command"]
        self.rewrite(ref, lambda value: value["environment"].update(CUDA_VISIBLE_DEVICES=""))
        gate.admit_packet(self.packet)
        self.rewrite(ref, lambda value: value["environment"].update(CUDA_VISIBLE_DEVICES="0"))
        with self.assertRaisesRegex(ValueError, "tokenizer command differs"):
            gate.admit_packet(self.packet)
        self.rewrite(ref, lambda value: value["environment"].update(CUDA_VISIBLE_DEVICES="", LD_PRELOAD="other.so"))
        with self.assertRaisesRegex(ValueError, "tokenizer command differs"):
            gate.admit_packet(self.packet)

    def test_wrong_target_and_untrusted_baseline_are_rejected(self):
        packet = copy.deepcopy(self.packet)
        packet["target"] = self.file("wrong-target.gguf", "different target")
        with self.assertRaisesRegex(ValueError, "capture target differs"):
            gate.admit_packet(packet)
        packet = copy.deepcopy(self.packet)
        packet["baseline"] = self.file("different-baseline.json", {})
        with self.assertRaisesRegex(ValueError, "untrusted baseline"):
            gate.admit_packet(packet)

    def test_runtime_and_loaded_library_substitution_are_rejected(self):
        self.rewrite(
            self.packet["build"], lambda b: b["environment"].update(GGML_CUDA_DECODE_CONSISTENT="1")
        )
        with self.assertRaisesRegex(ValueError, "correctness mode"):
            gate.admit_packet(self.packet)

    def test_unfrozen_loaded_cuda_is_rejected(self):
        maps = self.file("wrong-maps.txt", "0-1 r-xp 0 0 1 /unfrozen/libggml-cuda.so.0.28.0")
        self.packet["capture_library_maps"] = maps
        with self.assertRaisesRegex(ValueError, "unfrozen runtime library"):
            gate.admit_packet(self.packet)

    def test_prompt_and_fixed_history_mismatches_are_rejected(self):
        self.rewrite(
            self.packet["serving"]["request"], lambda x: x.update(prompt="different prompt")
        )
        with self.assertRaisesRegex(ValueError, "request or raw prompt"):
            gate.admit_packet(self.packet)
        self.rewrite(
            self.packet["serving"]["request"], lambda x: x.update(prompt="synthetic native prompt")
        )
        path = Path(self.packet["continuation_tokens"]["path"])
        path.write_text("3\n")
        self.packet["continuation_tokens"] = self.pin(path)
        with self.assertRaisesRegex(ValueError, "fixed continuation"):
            gate.admit_packet(self.packet)

    def test_next_token_off_by_one_and_shape_are_rejected(self):
        self.rewrite(self.packet["capture"], lambda x: x.update(logit_positions=[0, 1]))
        with self.assertRaisesRegex(ValueError, "next-token positions"):
            gate.admit_packet(self.packet)
        self.rewrite(
            self.packet["capture"],
            lambda x: x.update(logit_positions=[1, 2], feature_shape=[3, 2559]),
        )
        with self.assertRaisesRegex(ValueError, "capture shape"):
            gate.admit_packet(self.packet)

    def test_array_size_nonfinite_and_false_argmax_metadata_are_rejected(self):
        self.rewrite(self.packet["capture"], lambda x: x.update(greedy_ids=[1, 4]))
        with self.assertRaisesRegex(ValueError, "raw argmax"):
            gate.admit_packet(self.packet)
        self.rewrite(self.packet["capture"], lambda x: x.update(greedy_ids=[1, 5]))
        ref = self.packet["arrays"]["layer-2.f32"]
        data = np.fromfile(ref["path"], dtype="<f4")
        data[0] = np.nan
        data.tofile(ref["path"])
        ref.update(self.pin(Path(ref["path"])))
        with self.assertRaisesRegex(ValueError, "nonfinite native array"):
            gate.admit_packet(self.packet)

    def test_token_embeddings_must_be_same_for_repeated_token_ids(self):
        ref = self.packet["arrays"]["layer-0.f32"]
        data = np.fromfile(ref["path"], dtype="<f4").reshape(3, gate.WIDTH)
        data[2] += 0.1
        data.tofile(ref["path"])
        ref.update(self.pin(Path(ref["path"])))
        with self.assertRaisesRegex(ValueError, "embedding differs"):
            gate.admit_packet(self.packet)

    def test_sampling_changes_and_missing_evidence_fail_closed(self):
        self.rewrite(
            self.packet["serving"]["response"],
            lambda x: x["generation_settings"].update(presence_penalty=0.1),
        )
        with self.assertRaisesRegex(ValueError, "sampling semantics"):
            gate.admit_packet(self.packet)
        del self.packet["serving"]["request"]
        with self.assertRaisesRegex(ValueError, "exact serving evidence"):
            gate.admit_packet(self.packet)

    def test_receipt_publication_is_exclusive_and_keeps_arrays_shared(self):
        packet = self.root / "packet.json"
        packet.write_text(json.dumps(self.packet))
        output = self.root / "admission.json"
        before = Path(self.packet["arrays"]["logits.f32"]["path"]).stat()
        result = gate.prepare(packet, output)
        self.assertEqual(json.loads(output.read_text()), result)
        self.assertEqual(
            Path(self.packet["arrays"]["logits.f32"]["path"]).stat().st_ino, before.st_ino
        )
        with self.assertRaisesRegex(ValueError, "refuse to overwrite"):
            gate.prepare(packet, output)

    def four_teacher_rows(self):
        """Create a complete authenticated fixture with an excluded middle row."""
        labels, native = [1, 4, 7, 8], [1, 5, 7, 8]
        inputs = [1, 2] + labels[:3]
        logits = np.zeros((4, gate.VOCAB), dtype=np.float32)
        for index, (label, proposal) in enumerate(zip(labels, native)):
            logits[index, label] = 0.95
            logits[index, proposal] = 1
        self.packet["arrays"]["logits.f32"] = self.array("capture/logits.f32", logits)
        for tap in [0, 2, 16, 29]:
            values = np.array(inputs, dtype=np.float32)[:, None].repeat(gate.WIDTH, 1)
            values *= 0.000123456
            name = f"layer-{tap}.f32"
            self.packet["arrays"][name] = self.array(f"capture/{name}", values)
        fixed = Path(self.packet["continuation_tokens"]["path"])
        fixed.write_text("1\n4\n7\n")
        self.packet["continuation_tokens"] = self.pin(fixed)
        self.rewrite(
            self.packet["capture"],
            lambda value: value.update(
                input_tokens=inputs,
                feature_shape=[5, gate.WIDTH],
                logits_shape=[4, gate.VOCAB],
                logit_positions=[1, 2, 3, 4],
                greedy_ids=native,
                array_bytes=sum(ref["bytes"] for ref in self.packet["arrays"].values()),
            ),
        )
        self.rewrite(self.packet["serving"]["request"], lambda value: value.update(n_predict=4))

        def response(value):
            value.update(tokens=labels, tokens_predicted=4)
            value["generation_settings"]["n_predict"] = 4

        self.rewrite(self.packet["serving"]["response"], response)
        return gate.admit_packet(self.packet)

    def test_actor_alignment_uses_next_teacher_label_and_unshifted_cache_position(self):
        ref = self.packet["arrays"]["logits.f32"]
        logits = np.fromfile(ref["path"], dtype="<f4").reshape(2, gate.VOCAB)
        logits[1, 4], logits[1, 5] = 1, 0.95
        logits.tofile(ref["path"])
        ref.update(self.pin(Path(ref["path"])))
        self.rewrite(self.packet["capture"], lambda value: value.update(greedy_ids=[1, 4]))
        receipt = gate.admit_packet(self.packet)
        steps = gate.build_actor_steps(receipt)
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertEqual(step["source_teacher_row_id"], 0)
        self.assertEqual(step["label_teacher_row_id"], 1)
        self.assertEqual(step["feature_input_position"], 1)
        self.assertEqual(step["decoder_position"], 1)
        self.assertEqual(step["input_position"], 2)
        self.assertEqual(step["label_position"], 3)
        self.assertEqual(step["input_token_id"], 1)
        self.assertEqual(step["label_token_id"], 4)
        self.assertNotEqual(step["label_token_id"], receipt["rows"][0]["serving_label"])
        self.assertTrue(step["supervision_eligible"])
        self.assertFalse(step["parent_success_protection_proven"])
        self.assertFalse(step["draft_vocab_membership_proven"])
        self.assertFalse(step["actor_native_qualification_proven"])
        self.assertEqual(gate.require_actor_steps(receipt, [0]), steps)
        with self.assertRaisesRegex(ValueError, "final teacher boundary"):
            gate.require_actor_steps(receipt, [1])

    def test_selected_later_window_keeps_serving_offset_and_full_prefix(self):
        self.four_teacher_rows()
        ref = self.packet["arrays"]["logits.f32"]
        logits = np.fromfile(ref["path"], dtype="<f4").reshape(4, gate.VOCAB)[2:].copy()
        logits.tofile(ref["path"])
        ref.update(self.pin(Path(ref["path"])))
        self.rewrite(
            self.packet["capture"],
            lambda value: value.update(
                logits_shape=[2, gate.VOCAB], logit_positions=[3, 4], greedy_ids=[7, 8],
                array_bytes=sum(item["bytes"] for item in self.packet["arrays"].values()),
            ),
        )
        with self.assertRaisesRegex(ValueError, "explicit capture command"):
            gate.admit_packet(self.packet)
        self.rewrite(
            self.packet["capture_command"],
            lambda value: value["argv"].extend(["--logit-positions", "3,4"]),
        )
        receipt = gate.admit_packet(self.packet)
        self.assertEqual([row["serving_token_index"] for row in receipt["rows"]], [2, 3])
        step, = gate.build_actor_steps(receipt)
        self.assertEqual((step["decoder_position"], step["input_position"], step["label_position"]), (3, 4, 5))
        self.assertEqual((step["input_token_id"], step["label_token_id"]), (7, 8))
        self.assertTrue(step["supervision_eligible"])
        self.rewrite(
            self.packet["capture_command"], lambda value: value["argv"].__setitem__(-1, "2,3")
        )
        with self.assertRaisesRegex(ValueError, "command/build binding"):
            gate.admit_packet(self.packet)

    def test_excluded_middle_teacher_row_blocks_both_adjacent_actor_transitions(self):
        receipt = self.four_teacher_rows()
        self.assertEqual(receipt["excluded_row_ids"], [1])
        steps = gate.build_actor_steps(receipt)
        self.assertEqual([step["actor_step_id"] for step in steps], [0, 1, 2])
        self.assertEqual([step["supervision_eligible"] for step in steps], [False, False, True])
        self.assertEqual(steps[0]["excluded_teacher_row_ids"]["supervision"], [1])
        self.assertEqual(steps[1]["excluded_teacher_row_ids"]["supervision"], [1])
        for purpose in ("supervision", "protection_numeric"):
            for step_id in (0, 1):
                with self.assertRaisesRegex(ValueError, "excluded row"):
                    gate.require_actor_steps(receipt, [step_id], purpose=purpose)
            accepted = gate.require_actor_steps(receipt, [2], purpose=purpose)
            self.assertEqual(accepted[0]["input_token_id"], 7)
            self.assertEqual(accepted[0]["label_token_id"], 8)
            self.assertEqual(accepted[0]["decoder_position"], 3)
            self.assertEqual(accepted[0]["input_position"], 4)
            self.assertEqual(accepted[0]["label_position"], 5)

    def test_actor_consumer_recomputes_gates_after_forged_flags(self):
        receipt = self.four_teacher_rows()
        steps = gate.build_actor_steps(receipt)
        for step in steps:
            step["supervision_eligible"] = True
            step["protection_numeric_eligible"] = True
        # Neither a fabricated output flag nor fabricated teacher eligibility
        # replaces actual native/serving agreement.
        receipt["eligible_row_ids"] = [0, 1, 2, 3]
        receipt["excluded_row_ids"] = []
        receipt["rows"][1]["supervision_eligible"] = True
        receipt["rows"][1]["protection_numeric_eligible"] = True
        for purpose in ("supervision", "protection_numeric"):
            with self.assertRaisesRegex(ValueError, "excluded row"):
                gate.require_actor_steps(receipt, [0, 1], purpose=purpose)

    def test_noncontiguous_teacher_positions_ids_and_indices_are_rejected(self):
        receipt = self.four_teacher_rows()
        for field, value in (("input_position", 4), ("row_id", 3), ("serving_token_index", 3)):
            changed = copy.deepcopy(receipt)
            changed["rows"][2][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contiguous"):
                gate.build_actor_steps(changed)
        with self.assertRaisesRegex(ValueError, "distinct actor step IDs"):
            gate.require_actor_steps(receipt, [2, 2])
        with self.assertRaisesRegex(ValueError, "distinct actor step IDs"):
            gate.require_actor_steps(receipt, [True])

    def test_final_teacher_row_alone_has_no_actor_step(self):
        receipt = gate.admit_packet(self.packet)
        receipt["rows"] = receipt["rows"][:1]
        receipt["excluded_row_ids"] = []
        self.assertEqual(gate.build_actor_steps(receipt), [])
        self.assertEqual(gate.require_actor_steps(receipt, []), [])
        with self.assertRaisesRegex(ValueError, "final teacher boundary"):
            gate.require_actor_steps(receipt, [0])


if __name__ == "__main__":
    unittest.main()

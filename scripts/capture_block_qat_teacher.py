#!/usr/bin/env python3
"""Persistent native exact-prefix five-tap/full-vocabulary TRAIN teacher client.

Native helper loads target once, resets F16 KV for each request, and writes raw
F32 row-major files. `NativeTeacher.capture_prefix(tokens,tap_ids,logits_mode)`
serves final-prefix full logits for live DSpark L1, dense rows by default, or
opt-in indexed full-vocabulary rows for TRAIN sequence capture. Indexed output
selects storage only after identical dense-head decoding; features remain complete.
Teacher target/precision/hardware ancestry is recorded;
no optimizer or drafter is present. GPU use belongs to the sole operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_block_binary import sha256
from gguf import GGMLQuantizationType as Type
from gguf import GGUFReader

from w1a1_eagle.block_data import (
    BLOCK_LOGITS_SELECTION,
    PARTIAL_LABEL_POLICY,
    TAPS,
    block_teacher_indices,
    generated_block_anchors,
    validate_logits_indices,
)

ALIGNED_PROMPTS = {
    "source-00-0569": (
        "train",
        "magicoder:line-000854-index-39750",
        "520fc196ca6a59159c9c9a373655e248c140301d25db8ef0df0e265d96aa0a14",
    ),
    "source-03-0539": (
        "train",
        "gsm8k:train-000763",
        "b787bd36a3a713b8cfb22ae7b7baee6c246d40a6eb647466d1e9c79fe832aeaf",
    ),
    "source-00-0199": (
        "reference",
        "magicoder:line-015359-index-34538",
        "ef45524781dcf21ad9edf8a6c31952cce8da867593605d64fed3a6ea53fa5926",
    ),
    "source-00-0274": (
        "reference",
        "dolly:line-009016",
        "d17b02801e776686934aed3cd4f26879daf27ff2eee7b648de823136834ffc4c",
    ),
}
ALIGNED_TARGET = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
ALIGNED_TEMPLATE = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def validate_deployment_rendering(rendering):
    """A named diagnostic contract; flags never substitute for native tokens."""
    messages = rendering.get("original_messages")
    tokens = rendering.get("prompt_token_ids")
    text = rendering.get("rendered_prompt")
    if (
        rendering.get("schema") != "dspark_deployment_rendering_v1"
        or rendering.get("target", {}).get("sha256") != ALIGNED_TARGET
        or rendering.get("template_sha256") != ALIGNED_TEMPLATE
        or rendering.get("chat_template_kwargs") != {"enable_thinking": False}
        or rendering.get("tokenizer") != {"add_special": True, "parse_special": True}
        or not isinstance(messages, list)
        or not messages
        or any(
            set(m) != {"role", "content"} or not all(isinstance(v, str) for v in m.values())
            for m in messages
        )
        or not isinstance(text, str)
        or not text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        or not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not int or t < 0 for t in tokens)
        or rendering.get("prompt_token_count") != len(tokens)
    ):
        raise ValueError("explicit deployment render/template/token boundary differs")
    for role in (
        "renderer_binary",
        "target",
        "render_request",
        "render_response",
        "tokenize_request",
        "tokenize_response",
    ):
        record = rendering.get(role, {})
        if (
            set(record) != {"path", "sha256"}
            or not Path(record["path"]).is_absolute()
            or len(record["sha256"]) != 64
        ):
            raise ValueError("actual deployment renderer source/response locators required")
    return rendering


def prepare_aligned_requests(rows):
    """Join saved source messages and native rendering artifacts, without model IO."""
    if len(rows) != 4 or [r.get("chain_id") for r in rows] != list(ALIGNED_PROMPTS):
        raise ValueError("exact two TRAIN and two separate reference prompts required")
    requests = []
    for row in rows:
        source = row["original_source"]
        rendering_pin = row["deployment_rendering"]
        for record in (source, rendering_pin):
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("original source or deployment rendering changed")
        original = json.loads(Path(source["path"]).read_text())
        rendering = validate_deployment_rendering(
            json.loads(Path(rendering_pin["path"]).read_text())
        )
        role, prompt_id, prompt_sha = ALIGNED_PROMPTS[row["chain_id"]]
        ancestry = original["chain_ancestry"]
        if (
            original.get("prompt", {}).get("messages") != rendering["original_messages"]
            or ancestry.get("prompt_id") != prompt_id
            or ancestry.get("prompt_sha256") != prompt_sha
            or original.get("target_sha256") != ALIGNED_TARGET
            or original.get("target_chat_template_sha256") != ALIGNED_TEMPLATE
        ):
            raise ValueError("aligned capture original message/source identity differs")
        for name in ("render_request", "render_response", "tokenize_request", "tokenize_response"):
            record = rendering[name]
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("native rendering/tokenization artifact changed")
        render_request = json.loads(Path(rendering["render_request"]["path"]).read_text())
        render_response = json.loads(Path(rendering["render_response"]["path"]).read_text())
        tokenize_request = json.loads(Path(rendering["tokenize_request"]["path"]).read_text())
        tokenize_response = json.loads(Path(rendering["tokenize_response"]["path"]).read_text())
        if (
            render_request.get("messages") != rendering["original_messages"]
            or render_request.get("chat_template_kwargs") != {"enable_thinking": False}
            or render_response.get("prompt") != rendering["rendered_prompt"]
            or tokenize_request
            != {"content": rendering["rendered_prompt"], "add_special": True, "parse_special": True}
            or tokenize_response.get("tokens") != rendering["prompt_token_ids"]
        ):
            raise ValueError("actual native render/token response join differs")
        requests.append(
            {
                "template_mode": "raw_text",
                "prompt_text": rendering["rendered_prompt"],
                "max_new_tokens": 32,
                "max_prompt_tokens": 2016,
                "tap_ids": list(TAPS),
                "logits_mode": "all",
                "deployment_rendering": rendering,
                "alignment_source": {
                    "chain_id": row["chain_id"],
                    "role": role,
                    "original_source": source,
                    "rendering_receipt": rendering_pin,
                    "original_prompt_sha256": prompt_sha,
                },
                "chain_ancestry": {
                    **{k: ancestry[k] for k in ("prompt_id", "domain", "source_split")},
                    "prompt_sha256": hashlib.sha256(
                        rendering["rendered_prompt"].encode()
                    ).hexdigest(),
                    "prompt_length": len(rendering["prompt_token_ids"]),
                },
            }
        )
    return requests


def render_aligned_inputs(plan, base_url, output):
    """Use the existing live native renderer/tokenizer; never generate tokens."""
    from benchmark_native_eagle import request_json

    if (
        plan.get("schema") != "dspark_mode_aligned_capture_inputs_v1"
        or plan.get("preparation_only") is not True
        or [r.get("chain_id") for r in plan["rows"]] != list(ALIGNED_PROMPTS)
        or not base_url.startswith("http://127.0.0.1:")
        or output.exists()
    ):
        raise ValueError("fixed four-message preparation and fresh native renderer output required")
    sources = []
    for row in plan["rows"]:
        source = row["original_source"]
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("saved original source changed before rendering")
        old = json.loads(Path(source["path"]).read_text())
        expected = ALIGNED_PROMPTS[row["chain_id"]]
        if (
            old["chain_ancestry"]["prompt_id"] != expected[1]
            or old["chain_ancestry"]["prompt_sha256"] != expected[2]
        ):
            raise ValueError("fixed original source messages differ")
        sources.append(old)
    output.mkdir(parents=True, exist_ok=False)
    updated = json.loads(json.dumps(plan))
    for row, old in zip(updated["rows"], sources, strict=True):
        directory = output / row["chain_id"]
        directory.mkdir()
        render = {
            "messages": old["prompt"]["messages"],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
        }
        records = {}
        for name, endpoint, body in (("render", "/apply-template", render),):
            req = directory / (name + "-request.json")
            req.write_text(json.dumps(body, sort_keys=True) + "\n")
            status, raw = request_json(base_url + endpoint, body, 30)
            resp = directory / (name + "-response.json")
            resp.write_text(raw)
            if status != 200:
                raise RuntimeError("native deployment rendering failed")
            records[name + "_request"] = {"path": str(req.resolve()), "sha256": sha256(req)}
            records[name + "_response"] = {"path": str(resp.resolve()), "sha256": sha256(resp)}
        prompt = json.loads(raw)["prompt"]
        body = {"content": prompt, "add_special": True, "parse_special": True}
        req = directory / "tokenize-request.json"
        req.write_text(json.dumps(body, sort_keys=True) + "\n")
        status, raw = request_json(base_url + "/tokenize", body, 30)
        resp = directory / "tokenize-response.json"
        resp.write_text(raw)
        if status != 200:
            raise RuntimeError("native deployment tokenization failed")
        for name, path in (("tokenize_request", req), ("tokenize_response", resp)):
            records[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
        tokens = json.loads(raw)["tokens"]
        rendering = {
            "schema": "dspark_deployment_rendering_v1",
            **records,
            "renderer_binary": plan["renderer_binary"],
            "target": plan["target"],
            "template_sha256": ALIGNED_TEMPLATE,
            "original_messages": render["messages"],
            "chat_template_kwargs": render["chat_template_kwargs"],
            "rendered_prompt": prompt,
            "prompt_token_ids": tokens,
            "prompt_token_count": len(tokens),
            "tokenizer": {"add_special": True, "parse_special": True},
        }
        validate_deployment_rendering(rendering)
        path = directory / "deployment-rendering.json"
        path.write_text(json.dumps(rendering, indent=2, sort_keys=True) + "\n")
        row["deployment_rendering"] = {"path": str(path.resolve()), "sha256": sha256(path)}
    path = output / "rendered-inputs.json"
    path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    return path


class NativeTeacher:
    def generate_deployment_aligned(self, request):
        """Explicit raw-text capture, with native deployment rendering ancestry."""
        rendering = validate_deployment_rendering(request["deployment_rendering"])
        source = request.get("alignment_source", {})
        expected = ALIGNED_PROMPTS.get(source.get("chain_id"))
        if expected is None or source.get("role") != expected[0]:
            raise ValueError("fixed original TRAIN/reference role differs")
        if (
            request.get("template_mode") != "raw_text"
            or request.get("prompt_text") != rendering["rendered_prompt"]
            or request.get("max_new_tokens") != 32
            or request.get("tap_ids") != list(TAPS)
            or request.get("logits_mode") != "all"
            or self.ancestry["target_sha256"] != rendering["target"]["sha256"]
            or self.ancestry["target_chat_template_sha256"] != rendering["template_sha256"]
            or self.ancestry["target_precision"] != "F16"
            or self.ancestry["target_kv_type"] != "F16"
        ):
            raise ValueError("aligned raw-text capture target/method differs")
        receipt = self.generate_capture(
            prompt_text=request["prompt_text"],
            template_mode="raw_text",
            max_new_tokens=32,
            max_prompt_tokens=request["max_prompt_tokens"],
            tap_ids=request["tap_ids"],
            logits_mode="all",
            chain_ancestry=request["chain_ancestry"],
        )
        receipt["deployment_rendering"] = rendering
        receipt["alignment_source"] = request["alignment_source"]
        self._save_receipt(receipt)
        if (
            receipt["rendered_prompt"] != rendering["rendered_prompt"]
            or receipt["prompt_length"] != len(rendering["prompt_token_ids"])
            or receipt["tokens"][: receipt["prompt_length"]] != rendering["prompt_token_ids"]
        ):
            raise ValueError("deployment versus raw capture rendered-token equality failed")
        anchors = generated_block_anchors(receipt["prompt_length"], len(receipt["tokens"]))
        role = expected[0]
        if role not in {"train", "reference"}:
            raise ValueError("aligned capture must retain its fixed TRAIN/reference role")
        if role == "train" and len(anchors) < 4:
            raise ValueError("actual generated chain lacks complete blocks1-3; no padding allowed")
        receipt["mode_alignment_status"] = "EXACT_RENDERED_TOKEN_PREFIX_MATCH"
        receipt["complete_post_seed_blocks"] = [i for i in (1, 2, 3) if i < len(anchors)]
        receipt["reference_support"] = (
            ("complete_block_reader" if anchors else "native_prefix_only_no_complete_block")
            if role == "reference"
            else "not_reference"
        )
        self._save_receipt(receipt)
        return receipt

    def __init__(
        self,
        binary: Path,
        target: Path,
        output_root: Path,
        *,
        target_sha256: str,
        max_tokens: int,
        gpu_layers: int,
        producer_source_revision: str,
        timeout_seconds: float = 300,
    ):
        self.binary, self.target, self.root = map(Path, (binary, target, output_root))
        if (
            self.binary.resolve() == self.target.resolve()
            or not self.binary.is_file()
            or not self.target.is_file()
        ):
            raise ValueError("distinct native binary and frozen target files required")
        if (
            type(max_tokens) is not int
            or not 0 < max_tokens <= 32768
            or type(gpu_layers) is not int
            or not 0 <= gpu_layers <= 999
        ):
            raise ValueError("invalid token/GPU bounds")
        if (
            not isinstance(producer_source_revision, str)
            or len(producer_source_revision) != 40
            or any(c not in "0123456789abcdef" for c in producer_source_revision)
        ):
            raise ValueError("exact native producer source revision required")
        actual_target = sha256(self.target)
        if actual_target != target_sha256:
            raise ValueError("frozen target SHA256 mismatch")
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("invalid timeout")
        target_reader = GGUFReader(self.target)
        target_inventory = {}
        matrix_types = set()
        for tensor in target_reader.tensors:
            name = tensor.tensor_type.name
            target_inventory[name] = target_inventory.get(name, 0) + 1
            if len(tensor.shape) >= 2:
                matrix_types.add(tensor.tensor_type)
        target_precision = (
            "F16"
            if matrix_types == {Type.F16}
            else "F32"
            if matrix_types == {Type.F32}
            else "mixed"
        )
        tokenizer_metadata = {
            key: field.contents()
            for key, field in target_reader.fields.items()
            if key.startswith("tokenizer.")
        }
        tokenizer_hash = hashlib.sha256(
            json.dumps(
                tokenizer_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        template = tokenizer_metadata.get("tokenizer.chat_template", "")
        template_hash = hashlib.sha256(
            (
                template if isinstance(template, str) else json.dumps(template, sort_keys=True)
            ).encode()
        ).hexdigest()
        del target_reader
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.ancestry = {
            "target_sha256": actual_target,
            "producer_binary_sha256": sha256(self.binary),
            "producer_source_revision": producer_source_revision,
            "client_source_sha256": sha256(Path(__file__)),
            "gpu_layers": gpu_layers,
            "target_kv_type": "F16",
            "target_precision": target_precision,
            "target_tensor_type_inventory": target_inventory,
            "tokenizer_metadata_sha256": tokenizer_hash,
            "target_chat_template_sha256": template_hash,
            "producer_host": os.uname().sysname + " " + os.uname().machine,
        }
        self.log = open(self.root / f"producer-{uuid.uuid4().hex}.log", "w")
        self.process = subprocess.Popen(
            [str(self.binary), str(self.target), str(self.root), str(max_tokens), str(gpu_layers)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            # Stay inside the remote_job process group: a supervisor STOP must
            # signal the teacher alongside its Python owner. The native child
            # never forks, so local timeout escalation targets only its PID.
            start_new_session=False,
        )
        self.ancestry.update(
            producer_pid=self.process.pid,
            producer_pgid=os.getpgid(self.process.pid),
            producer_parent_pid=os.getpid(),
        )
        self.native_runtime_binding = None
        self.closed = False

    def capture_prefix(
        self,
        tokens,
        tap_ids,
        *,
        logits_mode="last",
        chain_ancestry=None,
        decode_history=None,
        logits_indices=None,
    ):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("native teacher process is not live")
        if (
            not isinstance(tokens, (list, tuple))
            or not tokens
            or len(tokens) > self.max_tokens
            or any(type(n) is not int or n < 0 for n in tokens)
        ):
            raise ValueError("exact nonempty bounded integer token prefix required")
        self._validate_taps(tap_ids)
        if logits_mode not in ("all", "last", "none", "indexed"):
            raise ValueError("unsupported logits mode")
        request = {
            "id": uuid.uuid4().hex,
            "tokens": list(tokens),
            "tap_ids": list(tap_ids),
            "logits_mode": logits_mode,
        }
        if logits_mode == "indexed":
            validate_logits_indices(logits_indices, len(tokens))
            if not logits_indices:
                raise ValueError("nonempty indexed teacher row map required for replay")
            request["logits_indices"] = list(logits_indices)
        elif logits_indices is not None:
            raise ValueError("logit selection requires indexed mode")
        if decode_history is not None:
            if not isinstance(decode_history, list) or not decode_history:
                raise ValueError("nonempty source decode history required")
            offset = 0
            for chunk in decode_history:
                if (
                    not isinstance(chunk, dict)
                    or set(chunk) != {"offset", "count", "phase", "kv_reused_from_same_chain"}
                    or type(chunk["offset"]) is not int
                    or chunk["offset"] != offset
                    or type(chunk["count"]) is not int
                    or not 0 < chunk["count"] <= 256
                    or chunk["phase"] not in ("prefill", "target_only_greedy")
                    or (chunk["phase"] == "target_only_greedy" and chunk["count"] != 1)
                    or type(chunk["kv_reused_from_same_chain"]) is not bool
                    or chunk["kv_reused_from_same_chain"] != (offset > 0)
                ):
                    raise ValueError("invalid exact source decode partition")
                offset += chunk["count"]
            if offset != len(tokens):
                raise ValueError("source decode partitions do not cover exact prefix")
            request["decode_history"] = decode_history
        return self._capture_request(request, chain_ancestry)

    @staticmethod
    def _validate_taps(tap_ids):
        if (
            not isinstance(tap_ids, (list, tuple))
            or len(tap_ids) not in (3, 5)
            or len(set(tap_ids)) != len(tap_ids)
            or any(type(n) is not int or n < 0 for n in tap_ids)
        ):
            raise ValueError("exact ordered three or five distinct native input taps required")

    def generate_capture(
        self,
        *,
        messages=None,
        prompt_text=None,
        template_mode,
        max_new_tokens,
        max_prompt_tokens=None,
        tap_ids,
        logits_mode="all",
        chain_ancestry=None,
        label_policy=None,
    ):
        if label_policy is not None:
            if label_policy != PARTIAL_LABEL_POLICY or logits_mode != "indexed":
                raise ValueError(
                    "partial-label generation requires all_available_v1 indexed replay"
                )
            # ecff native generated-index mode physically truncates incomplete tails.
            # Generate honestly without logits, then replay exact tokens and decode
            # partitions with an explicit map. Keep both original receipts/payloads.
            generated = self.generate_capture(
                messages=messages,
                prompt_text=prompt_text,
                template_mode=template_mode,
                max_new_tokens=max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
                tap_ids=tap_ids,
                logits_mode="none",
                chain_ancestry=chain_ancestry,
            )
            tokens = generated["tokens"]
            anchors = generated_block_anchors(generated["prompt_length"], len(tokens), partial=True)
            indices = block_teacher_indices(anchors, len(tokens))
            if not indices:
                raise ValueError("native generation has no observed loss-bearing decision")
            replay = self.capture_prefix(
                tokens,
                tap_ids,
                logits_mode="indexed",
                logits_indices=indices,
                chain_ancestry=generated["chain_ancestry"],
                decode_history=generated["decode_history"],
            )
            source = self.root / generated["id"] / "receipt.json"
            replay["generation_source"] = {"path": str(source.resolve()), "sha256": sha256(source)}
            self._save_receipt(replay)
            return replay
        self._validate_taps(tap_ids)
        if logits_mode not in ("all", "last", "none", "indexed"):
            raise ValueError("unsupported logits mode")
        if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= self.max_tokens:
            raise ValueError("invalid bounded native continuation")
        if max_prompt_tokens is None:
            max_prompt_tokens = self.max_tokens - max_new_tokens
        if (
            type(max_prompt_tokens) is not int
            or max_prompt_tokens <= 0
            or max_prompt_tokens + max_new_tokens > self.max_tokens
        ):
            raise ValueError("invalid bounded native prompt token cap")
        prompt = {
            "template_mode": template_mode,
            "max_new_tokens": max_new_tokens,
            "max_prompt_tokens": max_prompt_tokens,
        }
        if template_mode == "native_chat" and prompt_text is None:
            if (
                not isinstance(messages, list)
                or not messages
                or any(
                    not isinstance(item, dict)
                    or set(item) != {"role", "content"}
                    or any(not isinstance(value, str) for value in item.values())
                    for item in messages
                )
            ):
                raise ValueError("original role/content messages required")
            prompt["messages"] = messages
            source_text = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        elif template_mode == "raw_text" and messages is None and isinstance(prompt_text, str):
            if not prompt_text:
                raise ValueError("empty raw native prompt")
            prompt["text"] = prompt_text
            source_text = prompt_text
        else:
            raise ValueError("explicit native_chat or raw_text source required")
        request = {
            "id": uuid.uuid4().hex,
            "prompt": prompt,
            "tap_ids": list(tap_ids),
            "logits_mode": logits_mode,
        }
        if logits_mode == "indexed":
            request["logits_selection"] = dict(BLOCK_LOGITS_SELECTION)
        receipt = self._capture_request(request, chain_ancestry)
        receipt["prompt_source_sha256"] = hashlib.sha256(source_text.encode()).hexdigest()
        receipt["rendered_prompt_sha256"] = hashlib.sha256(
            receipt["rendered_prompt"].encode()
        ).hexdigest()
        receipt["chat_template_sha256"] = hashlib.sha256(
            receipt["chat_template"].encode()
        ).hexdigest()
        if (
            template_mode == "native_chat"
            and receipt["chat_template_sha256"] != self.ancestry["target_chat_template_sha256"]
        ):
            raise ValueError("native selected template differs from frozen GGUF template")
        if chain_ancestry is not None:
            if (
                "prompt_length" in chain_ancestry
                and chain_ancestry["prompt_length"] != receipt["prompt_length"]
            ):
                raise ValueError("caller prompt boundary differs from native tokenization")
            receipt["chain_ancestry"] = {
                **chain_ancestry,
                "prompt_length": receipt["prompt_length"],
            }
        self._save_receipt(receipt)
        return receipt

    def _bind_native_runtime(self):
        """Hash actual mapped project runtime libraries once, on Linux only."""
        if self.native_runtime_binding is not None:
            return self.native_runtime_binding
        maps = Path(f"/proc/{self.process.pid}/maps")
        if not maps.is_file():
            self.native_runtime_binding = {
                "scope": "unavailable_on_non_linux",
                "libraries": [],
                "actual_mapping_checked": False,
            }
            return self.native_runtime_binding
        libraries = {}
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            name = fields[5]
            basename = Path(name).name
            if not basename.startswith(("libllama", "libggml")) or ".so" not in basename:
                continue
            if name.endswith(" (deleted)"):
                raise ValueError("native mapped runtime library was replaced/deleted")
            path = Path(name)
            if path.stat().st_ino != int(fields[4]):
                raise ValueError("native mapped runtime library inode differs from disk")
            if name not in libraries:
                libraries[name] = {"path": name, "sha256": sha256(path)}
        self.native_runtime_binding = {
            "scope": "actual_linux_mapped_project_libraries",
            "actual_mapping_checked": True,
            "libraries": [libraries[name] for name in sorted(libraries)],
            "binary_sha256": self.ancestry["producer_binary_sha256"],
        }
        return self.native_runtime_binding

    def _save_receipt(self, receipt):
        directory = self.root / receipt["id"]
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )

    def _capture_request(self, request, chain_ancestry):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("native teacher process is not live")
        tokens = request.get("tokens")
        logits_mode = request["logits_mode"]
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        import selectors

        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(self.timeout):
                self.close()
                raise TimeoutError("native teacher request timed out; owned producer stopped")
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise RuntimeError(f"native teacher exited {self.process.poll()}; see {self.log.name}")
        receipt = json.loads(line)
        if (
            receipt.get("schema") != "block_native_teacher_request_v1"
            or not receipt.get("complete")
            or any(receipt.get(k) != v for k, v in request.items())
        ):
            raise ValueError("native teacher response differs from exact caller prefix")
        tokens = receipt["tokens"]
        if (
            not tokens
            or len(tokens) > self.max_tokens
            or any(type(n) is not int or n < 0 for n in tokens)
        ):
            raise ValueError("invalid native generated token chain")
        if "prompt" in request:
            generation = receipt.get("generation", {})
            boundary = receipt.get("prompt_length")
            if (
                type(boundary) is not int
                or not 0 < boundary <= len(tokens)
                or generation.get("mode") != "native_target_greedy"
                or generation.get("max_new_tokens") != request["prompt"]["max_new_tokens"]
                or generation.get("generated_tokens") != len(tokens) - boundary
                or generation.get("stop_eog") is not True
                or receipt.get("tokenizer")
                != {"add_special": True, "parse_special": True, "implementation": "llama_tokenize"}
            ):
                raise ValueError("invalid native generation/tokenizer contract")
        if logits_mode == "indexed":
            expected = (
                block_teacher_indices(
                    generated_block_anchors(receipt["prompt_length"], len(tokens))
                )
                if "prompt" in request
                else request["logits_indices"]
            )
            validate_logits_indices(receipt.get("logits_indices"), len(tokens), expected=expected)
        elif "logits_indices" in receipt or "logits_selection" in receipt:
            raise ValueError("unexpected indexed teacher row map in dense response")
        directory = self.root / request["id"]
        features_shape, logits_shape = receipt["features_shape"], receipt["logits_shape"]
        if (
            len(features_shape) != 3
            or features_shape[:2] != [len(tokens), len(request["tap_ids"])]
            or features_shape[2] <= 0
            or logits_shape[0]
            != (
                len(tokens)
                if logits_mode == "all"
                else 1
                if logits_mode == "last"
                else len(receipt["logits_indices"])
                if logits_mode == "indexed"
                else 0
            )
            or logits_shape[1] <= 0
        ):
            raise ValueError("native teacher response shape mismatch")
        files = {}
        descriptors = [("features", features_shape)]
        if logits_mode != "none":
            descriptors.append(("logits", logits_shape))
        for name, shape in descriptors:
            path = directory / (name + ".f32")
            if path.stat().st_size != 4 * int(np.prod(shape)):
                raise ValueError("native teacher file byte count mismatch")
            files[name] = {
                "path": str(path),
                "shape": shape,
                "dtype": "float32",
                "sha256": sha256(path),
            }
        receipt = {
            **receipt,
            **self.ancestry,
            "files": files,
            "native_runtime_binding": self._bind_native_runtime(),
            "chain_ancestry": chain_ancestry,
            "teacher_context_reset_between_requests": True,
            "prefix_freshness": "native_generated_chain"
            if "prompt" in request
            else "caller_current_student_prefix",
        }
        self._save_receipt(receipt)
        return receipt

    @staticmethod
    def array(receipt, name):
        item = receipt["files"][name]
        return np.memmap(item["path"], dtype=np.float32, mode="r", shape=tuple(item["shape"]))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            self.process.stdout.close()
            self.log.close()
        if self.process.returncode not in (0, -signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"native teacher failed with exit {self.process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def supervisor_stop(signum, _frame):
    """Unwind the CLI context so its exact native child is reaped on STOP."""
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    preparing = any(k in sys.argv for k in ("--prepare-aligned-inputs", "--render-aligned-inputs"))
    parser.add_argument("--prepare-aligned-inputs", type=Path)
    parser.add_argument("--render-aligned-inputs", type=Path)
    parser.add_argument("--renderer-url")
    parser.add_argument("--execute-aligned", action="store_true")
    for flag in ("binary", "target", "output-root", "requests"):
        parser.add_argument("--" + flag, type=Path, required=not preparing)
    parser.add_argument("--target-sha256", required=not preparing)
    parser.add_argument("--producer-source-revision", required=not preparing)
    parser.add_argument("--max-tokens", type=int, required=not preparing)
    parser.add_argument("--gpu-layers", type=int, required=not preparing)
    args = parser.parse_args()
    if args.render_aligned_inputs:
        if args.output_root is None or not str(args.output_root.resolve()).startswith("/mnt/c/"):
            raise ValueError("live native renderer artifacts require a C-backed output root")
        plan = json.loads(args.render_aligned_inputs.read_text())
        path = render_aligned_inputs(plan, args.renderer_url, args.output_root)
        print(
            json.dumps(
                {
                    "status": "DEPLOYMENT_RENDERED_ONLY_NO_CAPTURE",
                    "inputs": str(path),
                    "sha256": sha256(path),
                }
            )
        )
        return
    if preparing:
        plan = json.loads(args.prepare_aligned_inputs.read_text())
        if (
            plan.get("schema") != "dspark_mode_aligned_capture_inputs_v1"
            or plan.get("preparation_only") is not True
        ):
            raise ValueError("explicit preparation-only aligned input required")
        requests = prepare_aligned_requests(plan["rows"])
        if args.requests is None or args.requests.exists() or args.requests.is_symlink():
            raise ValueError("fresh aligned request output required")
        args.requests.parent.mkdir(parents=True, exist_ok=True)
        with args.requests.open("x") as stream:
            for request in requests:
                request["alignment_preparation_only"] = True
                stream.write(json.dumps(request, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": "ALIGNED_REQUESTS_PREPARED_NOT_CAPTURED",
                    "requests": str(args.requests),
                    "sha256": sha256(args.requests),
                    "count": 4,
                    "model_loaded": False,
                }
            )
        )
        return
    requests = [json.loads(line) for line in args.requests.read_text().splitlines()]
    aligned = any("deployment_rendering" in request for request in requests)
    if aligned:
        if (
            not args.execute_aligned
            or len(requests) != 4
            or [r.get("alignment_source", {}).get("chain_id") for r in requests]
            != list(ALIGNED_PROMPTS)
            or any(r.get("alignment_preparation_only") is not False for r in requests)
            or not str(args.output_root.resolve()).startswith("/mnt/c/")
            or args.output_root.exists()
            or args.output_root.is_symlink()
            or args.max_tokens != 2048
            or args.target_sha256 != ALIGNED_TARGET
        ):
            raise ValueError("new explicitly activated C-backed four-capture operation required")
        for variable in ("TMPDIR", "TMP", "TEMP", "CUDA_CACHE_PATH"):
            value = os.environ.get(variable)
            if value is None or not str(Path(value).resolve()).startswith("/mnt/c/"):
                raise ValueError(
                    "aligned capture temporary/cache paths must be C-backed: " + variable
                )
    signal.signal(signal.SIGTERM, supervisor_stop)
    signal.signal(signal.SIGINT, supervisor_stop)
    with NativeTeacher(
        args.binary,
        args.target,
        args.output_root,
        target_sha256=args.target_sha256,
        max_tokens=args.max_tokens,
        gpu_layers=args.gpu_layers,
        producer_source_revision=args.producer_source_revision,
    ) as teacher:
        for request in requests:
            if "deployment_rendering" in request:
                receipt = teacher.generate_deployment_aligned(request)
            elif "template_mode" in request:
                receipt = teacher.generate_capture(
                    messages=request.get("messages"),
                    prompt_text=request.get("prompt_text"),
                    template_mode=request["template_mode"],
                    max_new_tokens=request["max_new_tokens"],
                    max_prompt_tokens=request.get("max_prompt_tokens"),
                    tap_ids=request["tap_ids"],
                    logits_mode=request.get("logits_mode", "all"),
                    chain_ancestry=request.get("chain_ancestry"),
                )
            else:
                receipt = teacher.capture_prefix(
                    request["tokens"],
                    request["tap_ids"],
                    logits_mode=request.get("logits_mode", "all"),
                    chain_ancestry=request.get("chain_ancestry"),
                    decode_history=request.get("decode_history"),
                    logits_indices=request.get("logits_indices"),
                )
            print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

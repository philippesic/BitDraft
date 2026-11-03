"""Serve an admitted post-selection fresh packet without exposing private oracles.

CLI uses the frozen modern_requests models contract. --inputs JSON contains
manifest, receipt, requests, quality_cases, quality_controls, selection (all pins),
warmup_source (pin to prior development inputs JSON with roster pin), two old
fit warmups [{id,role:'fit',request:pin,reference:pin}], and optional source_pins.
The frozen target arm must be first. Five matched performance repetitions precede
one quality request per case/arm; judge workers run after all serving timings.
"""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import time
from pathlib import Path

import modern_fresh as fresh
from modern_requests import (
    PAYLOAD_KEYS,
    Requests,
    atomic,
    draft_counts,
    pin,
    read,
    require,
    signature,
)

PACKET_FILES = {
    "manifest": "manifest.json",
    "requests": "requests.json",
    "quality_cases": "quality-cases.json",
    "quality_controls": "quality-controls.json",
}
META = ("id", "domain", "family", "group_id", "source", "phase", "instance_seed", "identities")


def validate_payload(payload, output_limit):
    require(
        isinstance(payload, dict) and set(payload) == PAYLOAD_KEYS,
        "send only the exact public request body",
    )
    require(isinstance(payload["prompt"], str) and bool(payload["prompt"]), "raw prompt required")
    require(
        type(payload["n_predict"]) is int
        and payload["n_predict"] == output_limit
        and type(payload["seed"]) is int
        and payload["seed"] == 1234
        and type(payload["temperature"]) in (int, float)
        and payload["temperature"] == 0
        and payload["cache_prompt"] is False
        and payload["return_tokens"] is True
        and payload["stream"] is False,
        "frozen completion settings differ",
    )


def validate_response(response, payload):
    observed = signature(response)
    timings = response.get("timings", {})
    count = response.get("tokens_predicted")
    require(
        response.get("stop") is True
        and type(count) is int
        and count == len(observed["tokens"])
        and 0 <= count <= payload["n_predict"]
        and all(t < 248320 for t in observed["tokens"])
        and type(timings.get("predicted_n")) is int
        and timings["predicted_n"] == count,
        "unfinished response or inconsistent output token accounting",
    )
    require(
        type(timings.get("cache_n")) is int
        and timings["cache_n"] == 0
        and type(timings.get("prompt_n")) is int
        and timings["prompt_n"] > 0,
        "prompt cache or input token accounting differs",
    )
    for key in ("prompt_ms", "predicted_ms"):
        value = timings.get(key)
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid native timing: " + key,
        )
    require(count > 0 or observed["content"] == "", "empty token output has nonempty content")
    return observed


class Confirmation(Requests):
    def freeze_counts(self):
        arms = len(self.arms)
        return {
            "expected_startups": 6 * arms,
            "expected_measured_requests": 5 * 24 * arms,
            "expected_quality_requests": 24 * arms,
            "expected_excluded_warmups": 6 * len(self.warmups) * arms,
            "phase_plan": "five performance rounds, then one quality round; judge after serving",
        }

    def authenticate_inputs(self):
        require(self.arms[0]["kind"] == "target", "frozen target arm must be first")
        refs = self.inputs
        packet = {key: read(self.checked(refs[key])) for key in PACKET_FILES}
        receipt = read(self.checked(refs["receipt"]))
        manifest = packet["manifest"]
        require(
            receipt.get("schema") == "modern_fresh_confirmation_receipt_v1"
            and manifest.get("schema") == "modern_fresh_confirmation_inputs_v1"
            and receipt.get("admitted") is True
            and manifest.get("admitted") is True
            and receipt.get("status") == manifest.get("status") == "PREPARED_NO_MODEL_EVALUATION",
            "admitted unevaluated fresh packet required",
        )
        require(
            set(receipt.get("files", {})) == set(PACKET_FILES.values()), "incomplete packet hashes"
        )
        for key, filename in PACKET_FILES.items():
            require(
                refs[key]["sha256"] == receipt["files"][filename],
                "packet file hash differs: " + key,
            )
        require(
            receipt.get("manifest_sha256") == refs["manifest"]["sha256"],
            "manifest identity differs",
        )
        selection_path = self.checked(refs["selection"])
        selection, selection_pin, self.judge = fresh.validate_freeze(selection_path)
        require(
            receipt["freeze"] == manifest["freeze"] == selection_pin, "selection ancestry differs"
        )
        require(
            receipt.get("model_requests_executed_before_freeze") == 0
            and receipt.get("model_requests_executed") == 0
            and manifest.get("model_requests_executed") == 0
            and manifest.get("tokenization_performed") is False,
            "packet already exposed",
        )
        require(
            receipt.get("input_overlap_count") == 0 and manifest.get("overlap_case_ids") == [],
            "fresh packet has exclusion overlap",
        )
        require(receipt.get("seed") == manifest.get("seed"), "packet seed differs")
        require(
            receipt["seed"]
            == fresh.digest(selection_path.read_bytes() + b"modern4b-confirmation-v1"),
            "packet seed does not bind the selected freeze",
        )
        excluded, exclusion_receipts = fresh.load_exclusions(selection["exclusions"])
        require(
            manifest.get("exclusions") == exclusion_receipts, "packet exclusion inventories differ"
        )
        for key in ("candidate", "runtime", "source_freeze", "generator", "judge", "settings"):
            require(manifest.get(key) == selection[key], "manifest selection field differs: " + key)
        source = read(self.checked(selection["source_freeze"]))
        require(
            source.get("target") == self.models.get("target")
            and source.get("baseline") == self.models.get("baseline"),
            "selected target/baseline differs",
        )
        source_arms = source.get("arms", [])
        selected_id = selection["selected_arm_id"]
        require(sum(a["id"] == selected_id for a in self.arms) == 1, "selected arm missing")
        for arm in self.arms:
            matches = [a for a in source_arms if a.get("id") == arm["id"]]
            require(
                len(matches) == 1 and matches[0] == arm,
                "model or full runtime changed after selection: " + arm["id"],
            )
        selected = next(a for a in self.arms if a["id"] == selected_id)
        require(
            selected["kind"] == "w1a8"
            and selected["model"] == selection["candidate"]
            and selected["runtime"]["server"] == selection["runtime"],
            "selected candidate differs",
        )
        self.selected_id = selected_id
        self.output_limit = selection["settings"]["output_limit"]
        require(
            type(self.output_limit) is int and self.output_limit in (256, 512),
            "fresh output reserve differs",
        )
        groups = manifest.get("groups", [])
        require(
            len(groups) == 48
            and len({g["id"] for g in groups}) == 48
            and len({g["group_id"] for g in groups}) == 48,
            "48 unique fresh groups required",
        )
        by_id = {g["id"]: g for g in groups}
        self.rows = packet["requests"]
        quality_cases = packet["quality_cases"]
        require(
            len(self.rows) == len(quality_cases) == 24,
            "24 performance and24 quality cases required",
        )
        all_items = self.rows + quality_cases
        require(
            len({r["id"] for r in all_items}) == 48 and {r["id"] for r in all_items} == set(by_id),
            "fresh phase coverage differs",
        )
        self.payloads, self.gold = {}, {}
        observed = set()
        for phase, items in (("performance", self.rows), ("quality", quality_cases)):
            for domain, families in fresh.FAMILIES.items():
                for family in families:
                    require(
                        sum(r.get("domain") == domain and r.get("family") == family for r in items)
                        == 2,
                        "fresh family coverage differs",
                    )
            for item in items:
                require(
                    item.get("phase") == phase
                    and all(item.get(k) == by_id[item["id"]].get(k) for k in META),
                    "case metadata/phase differs",
                )
                require(item["id"].replace("-", "").replace("_", "").isalnum(), "unsafe case ID")
                payload = item["request"]
                validate_payload(payload, self.output_limit)
                require(
                    item["identities"] == fresh.identities(payload["prompt"]),
                    "prompt identity differs",
                )
                identity_values = set(item["identities"].values())
                require(
                    not identity_values.intersection(excluded | observed),
                    "fresh prompt overlaps exclusions or another packet case",
                )
                observed.update(identity_values)
                if phase == "performance":
                    require(
                        item["prompt"] == payload["prompt"]
                        and item["sha256"] == fresh.digest(payload["prompt"].encode()),
                        "performance body differs",
                    )
                else:
                    require(
                        payload["prompt"] == fresh.render(item["prompt"]),
                        "quality body exposes or changes task",
                    )
                self.payloads[item["id"]] = payload
        self.quality_cases = {c["id"]: c for c in quality_cases}
        self.quality_rows = [{k: c[k] for k in META} for c in quality_cases]
        controls = packet["quality_controls"]
        require(
            len(controls) == 24
            and len({c["case_id"] for c in controls}) == 24
            and {c["case_id"] for c in controls} == set(self.quality_cases),
            "quality control coverage differs",
        )
        control_results = []
        control_begin = time.perf_counter()
        for control in controls:
            require(
                control.get("correct_control_passed") is True
                and control.get("known_wrong_control_rejected") is True
                and control["positive"]["judgement"].get("passed") is True
                and len(control["semantic_negatives"]) >= 2
                and all(
                    n["judgement"].get("passed") is False for n in control["semantic_negatives"]
                ),
                "oracle controls not admitted",
            )
            case = self.quality_cases[control["case_id"]]
            positive = self.judge.judge(case, control["positive"]["answer"])
            negatives = [self.judge.judge(case, n["answer"]) for n in control["semantic_negatives"]]
            require(
                positive.get("passed") is True and all(n.get("passed") is False for n in negatives),
                "frozen oracle controls no longer discriminate",
            )
            control_results.append(
                {"case_id": case["id"], "positive": positive, "negatives": negatives}
            )
        require(
            receipt.get("performance_prompts")
            == receipt.get("quality_tasks")
            == receipt.get("quality_correct_controls_passed")
            == receipt.get("quality_known_wrong_controls_rejected")
            == 24
            and receipt.get("quality_templates") == 12,
            "receipt coverage differs",
        )
        control_seconds = time.perf_counter() - control_begin
        self.warmups = refs["warmups"]
        require(
            len(self.warmups) == 2 and len({w["id"] for w in self.warmups}) == 2,
            "two distinct old fit warmups required",
        )
        warmup_source = read(self.checked(refs["warmup_source"]))
        prior = warmup_source.get("warmups", [])
        old_roster_ref = warmup_source["roster"]
        old_roster = read(self.checked(old_roster_ref))
        require(
            old_roster.get("schema") == "w1a8_modern_phase1_generalization_roster_v2",
            "old warmup roster required",
        )
        old_rows = {r["id"]: r for r in old_roster["rows"]}
        for item in self.warmups:
            require(
                item.get("role") == "fit"
                and item["id"] not in by_id
                and item["id"].replace("-", "").replace("_", "").isalnum(),
                "warmups must be old fit inputs",
            )
            matches = [w for w in prior if w.get("id") == item["id"]]
            require(
                len(matches) == 1
                and all(matches[0].get(k) == item[k] for k in ("id", "request", "reference"))
                and old_rows.get(item["id"], {}).get("role") == "fit",
                "warmup ancestry differs from original fit inputs",
            )
            payload = read(self.checked(item["request"]))
            validate_payload(payload, 80)
            require(
                fresh.digest(payload["prompt"].encode())
                == old_rows[item["id"]]["rendered_prompt_sha256"],
                "warmup raw prompt differs from original fit roster",
            )
            self.payloads[item["id"]] = payload
            self.gold[item["id"]] = signature(read(self.checked(item["reference"])))
        self.manifest, self.receipt = manifest, receipt
        self.request_schema = "w1a8_modern_fresh_confirmation_requests_v1"
        self.request_scope = "post-selection synthetic instances; five performance reps then independent task quality; no broad prose or pretraining-novelty claim"
        self.input_source_refs = [
            pin(__file__),
            selection["generator"],
            selection["judge"],
            selection["source_freeze"],
            refs["selection"],
            refs["warmup_source"],
            old_roster_ref,
        ]
        self.input_source_refs += [refs[key] for key in PACKET_FILES] + [refs["receipt"]]
        self.input_source_refs += [item["file"] for item in selection["exclusions"]]
        self.current_phase = "performance"
        self.geometry = None
        (self.output / "gold").mkdir()
        atomic(
            self.output / "authenticated-controls.json",
            {
                "all_positive_passed": True,
                "all_negative_rejected": True,
                "cpu_wall_seconds": control_seconds,
                "records": control_results,
            },
        )

    def prepare_cell(self, port, cell, arm, rep):
        if self.geometry is not None:
            return
        require(
            arm["kind"] == "target" and rep == 0, "target must freeze tokenization before serving"
        )
        directory = cell / "tokenization"
        directory.mkdir()
        rows = []
        for item in self.rows + self.quality_rows:
            payload = {
                "content": self.payloads[item["id"]]["prompt"],
                "add_special": True,
                "parse_special": True,
                "with_pieces": False,
            }
            request_path = directory / (item["id"] + "-request.json")
            response_path = directory / (item["id"] + "-response.json")
            atomic(request_path, payload)
            response = self.request(port, "tokenize", payload)
            atomic(response_path, response)
            ids = response.get("tokens")
            require(
                isinstance(ids, list)
                and bool(ids)
                and all(type(t) is int and 0 <= t < 248320 for t in ids),
                "invalid native token IDs",
            )
            rows.append(
                {
                    "id": item["id"],
                    "prompt_tokens": len(ids),
                    "tokens": ids,
                    "request": pin(request_path),
                    "response": pin(response_path),
                    "fits": len(ids) + self.output_limit <= 4096,
                }
            )
        fits = all(r["fits"] for r in rows)
        atomic(
            self.output / "native-geometry.json",
            {
                "target": self.models["target"],
                "runtime": arm["runtime"],
                "context": 4096,
                "output_reserve": self.output_limit,
                "all_tasks_fit": fits,
                "rows": rows,
            },
        )
        require(
            fits,
            "whole fresh packet exceeds context plus output reserve; no truncation/drop allowed",
        )
        self.geometry = rows

    def completion(self, port, item, cell, index, warmup, arm, rep):
        self.guard()
        require(self.geometry is not None, "native geometry must precede every completion")
        allowed = self.warmups if warmup else self.rows
        require(item["id"] in {row["id"] for row in allowed}, "request outside current phase")
        created = item["id"] not in self.gold
        if created:
            require(
                not warmup
                and arm["kind"] == "target"
                and (
                    (self.current_phase == "performance" and rep == 0)
                    or (self.current_phase == "quality" and rep == 5)
                ),
                "only first frozen target response may create fresh gold",
            )
        payload = self.payloads[item["id"]]
        prefix = f"{index:02d}-{item['id']}"
        request_path = cell / (prefix + "-request.json")
        response_path = cell / (prefix + "-response.json")
        atomic(request_path, payload)
        begin = time.perf_counter()
        response = self.request(port, "completion", payload, timeout=180)
        seconds = time.perf_counter() - begin
        atomic(response_path, response)
        observed = validate_response(response, payload)
        proposed, accepted = draft_counts(response, arm["kind"])
        if not warmup:
            geometry = [r for r in self.geometry if r.get("id") == item["id"]]
            require(
                len(geometry) == 1
                and response["timings"]["prompt_n"] == geometry[0]["prompt_tokens"],
                "native prompt accounting differs from frozen tokenization",
            )
        require(observed["truncated"] is False, "fresh completion truncated context")
        if created:
            self.gold[item["id"]] = observed
            atomic(
                self.output / "gold" / (item["id"] + ".json"),
                {
                    "id": item["id"],
                    "phase": self.current_phase,
                    "response": pin(response_path),
                    "signature": observed,
                    "kind": "actual frozen target output; not a semantic oracle",
                },
            )
        require(
            observed == self.gold[item["id"]],
            "token/text/stop parity failed: " + arm["id"] + "/" + item["id"],
        )
        require(math.isfinite(seconds) and seconds > 0, "invalid HTTP wall")
        timings = response.get("timings", {})
        row = {
            "arm": arm["id"],
            "runtime_variant": arm["runtime"].get("variant", "original"),
            "repetition": rep,
            "id": item["id"],
            "phase": "warmup" if warmup else self.current_phase,
            "stage": self.current_phase,
            "warmup": warmup,
            "request": pin(request_path),
            "response": pin(response_path),
            "full_http_wall_seconds": seconds,
            "completion_tokens": len(observed["tokens"]),
            "zero_output": not observed["tokens"],
            "timings": timings,
            "draft_proposed": proposed,
            "draft_accepted": accepted,
            "target_gold_created": created,
            "token_text_stop_parity": True,
        }
        atomic(cell / (prefix + "-receipt.json"), row)
        self.records.append(row)
        return row

    def summarize(self):
        performance = []
        quality = []
        for arm in self.arms:
            reps = []
            for rep in range(5):
                rows = [
                    r
                    for r in self.records
                    if r["arm"] == arm["id"]
                    and r["phase"] == "performance"
                    and r["repetition"] == rep
                ]
                require(len(rows) == 24, "incomplete performance repetition")
                tokens = sum(r["completion_tokens"] for r in rows)
                wall = sum(r["full_http_wall_seconds"] for r in rows)
                require(
                    arm["kind"] == "target" or sum(r["draft_proposed"] for r in rows) > 0,
                    "speculative arm inactive across entire fresh repetition",
                )
                reps.append(
                    {
                        "repetition": rep,
                        "tokens": tokens,
                        "full_http_wall_seconds": wall,
                        "tokens_per_second": tokens / wall,
                        "draft_proposed": sum(r["draft_proposed"] or 0 for r in rows),
                        "draft_accepted": sum(r["draft_accepted"] or 0 for r in rows),
                    }
                )
            performance.append(
                {
                    "arm": arm["id"],
                    "kind": arm["kind"],
                    "runtime_variant": arm["runtime"].get("variant", "original"),
                    "variant_build": arm["runtime"].get("variant_build"),
                    "tokens_per_second": sum(r["tokens"] for r in reps)
                    / sum(r["full_http_wall_seconds"] for r in reps),
                    "zero_output_requests": sum(
                        r["completion_tokens"] == 0
                        for r in self.records
                        if r["arm"] == arm["id"] and r["phase"] == "performance"
                    ),
                    "repetitions": reps,
                }
            )
            rows = [r for r in self.records if r["arm"] == arm["id"] and r["phase"] == "quality"]
            require(
                len(rows) == 24 and len({r["id"] for r in rows}) == 24,
                "incomplete independent quality",
            )
            judged = []
            for row in rows:
                self.guard()
                case = self.quality_cases[row["id"]]
                response = read(self.checked(row["response"]))
                result = self.judge.judge(case, response["content"])
                require(type(result.get("passed")) is bool, "invalid quality judgement")
                judged.append(
                    {
                        "id": case["id"],
                        "domain": case["domain"],
                        "family": case["family"],
                        "response": row["response"],
                        "judgement": result,
                    }
                )
            quality.append(
                {
                    "arm": arm["id"],
                    "passed": sum(r["judgement"]["passed"] for r in judged),
                    "total": 24,
                    "zero_output_requests": sum(r["completion_tokens"] == 0 for r in rows),
                    "by_domain": {
                        domain: {
                            "passed": sum(
                                r["judgement"]["passed"] for r in judged if r["domain"] == domain
                            ),
                            "total": sum(r["domain"] == domain for r in judged),
                        }
                        for domain in fresh.FAMILIES
                    },
                    "cases": judged,
                }
            )
        baseline = next(p for p in performance if p["kind"] == "dflash")
        for p in performance:
            ratios = [
                r["tokens_per_second"] / b["tokens_per_second"]
                for r, b in zip(p["repetitions"], baseline["repetitions"])
            ]
            p["matched_DFlash_speedup_repetitions"] = ratios
            p["median_matched_DFlash_speedup"] = statistics.median(ratios)
        return performance, quality

    def run(self):
        self.records = []
        self.stopping = False

        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        try:
            self.authenticate()
            for rep in range(5):
                for arm in self.arms if rep % 2 == 0 else list(reversed(self.arms)):
                    self.cell(rep, arm)
            self.current_phase = "quality"
            self.rows = self.quality_rows
            for arm in self.arms:
                self.cell(5, arm)
            judge_begin = time.perf_counter()
            performance, quality = self.summarize()
            atomic(
                self.output / "result.json",
                {
                    "status": "FRESH_CONFIRMATION_EVALUATED",
                    "selected_arm_id": self.selected_id,
                    "token_text_stop_parity": True,
                    "performance_requests": sum(r["phase"] == "performance" for r in self.records),
                    "quality_requests": sum(r["phase"] == "quality" for r in self.records),
                    "excluded_warmups": sum(r["warmup"] for r in self.records),
                    "completed_startups": 6 * len(self.arms),
                    "performance": performance,
                    "independent_quality": quality,
                    "oracle_controls": pin(self.output / "authenticated-controls.json"),
                    "native_geometry": pin(self.output / "native-geometry.json"),
                    "records": self.records,
                    "judge_wall_seconds": time.perf_counter() - judge_begin,
                    "wall_seconds": time.perf_counter() - self.started,
                    "scope": self.request_scope,
                    "broad_prose_quality_qualified": False,
                    "predictor_validated": False,
                    "backlog_started": False,
                },
            )
        except BaseException as error:
            if self.created_output:
                atomic(
                    self.output / "failure.json",
                    {
                        "error": repr(error),
                        "completed_records": self.records,
                        "wall_seconds": time.perf_counter() - self.started,
                    },
                )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--allow-cuda", action="store_true")
    Confirmation(parser.parse_args()).run()


if __name__ == "__main__":
    main()

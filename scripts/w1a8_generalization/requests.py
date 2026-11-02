"""Frozen historical development requests; never an untouched release screen.

Preparation is CPU only. Execution requires the sole GPU operator's explicit
--allow-native and delegates to the retained, authenticated evaluator unchanged.
"""

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/home/philip/binary-eagle-decoding")
EVALUATOR = PROJECT / "runs/5080-broader-training-20261031/request-evaluator01.py"
EVALUATOR_SHA = "21e79e3b09149f2533a11c225525820e898c7661ff8761480e35ad4766fd16f2"
BASE_CONFIG = Path(
    "/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031/request-final-config02.json"
)
BASE_CONFIG_SHA = "d83cdb0403a40bb346f00499374f123cbe3e48b8f392a2c785cca29376b86255"
CONTROLS = ("q4_k_m", "parent3504", "previous_eal_best", "target_only")
SCOPE = (
    "Historical Qwen3-4B F16/F16-KV DSpark W1A8; reused 24-prompt development "
    "panel. No untouched confirmation, answer-quality, predictor-validation "
    "or backlog-screening claim."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def checked(ref):
    require(pin(ref["path"])["sha256"] == ref["sha256"], "input pin changed: " + ref["path"])
    return Path(ref["path"])


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def references(value):
    """Collect every pinned dependency, including nested panel and build pins."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def authenticate_all(value):
    seen = {}
    for ref in references(value):
        path = str(Path(ref["path"]).resolve())
        require(path not in seen or seen[path] == ref["sha256"], "conflicting pin: " + path)
        if path not in seen:
            checked(ref)
            seen[path] = ref["sha256"]


def validate_candidate(path, expected_updates, baseline):
    finish = read(path)
    require(
        finish.get("status") == "EXPOSURE_COMPLETE", "successful exposure-complete finish required"
    )
    require(
        type(expected_updates) is int
        and expected_updates > 0
        and type(finish.get("local_updates")) is int
        and finish["local_updates"] == expected_updates,
        "candidate exposure differs from matched updates",
    )
    seconds = finish.get("additional_TRAIN_seconds")
    require(
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(seconds)
        and seconds > 0,
        "positive measured training exposure required",
    )
    require(
        finish.get("native_evaluation_required") is True
        and finish.get("backlog_screening_started") is False,
        "generalization finish contract required",
    )
    spec = read(checked(finish["spec"]))
    require(spec.get("schema") == "w1a8_generalization_v1", "generalization training spec required")
    require(
        spec.get("policy") in ("legacy_errors", "occupancy")
        and expected_updates <= spec["horizon_updates"],
        "training policy or exposure horizon differs",
    )
    coefficient = spec.get("retention_coefficient")
    require(
        isinstance(coefficient, (int, float)) and math.isfinite(coefficient) and coefficient >= 0,
        "finite retention objective required",
    )
    base = read(checked(spec["base_spec"]))
    require(
        spec["scientific_protocol"] == base["scientific_protocol"]
        and spec["precision"] == base["precision"],
        "training spec protocol/precision changed",
    )
    protocol = spec["scientific_protocol"]
    require(
        protocol
        == {
            "body_rows": 7,
            "draft_depth": 5,
            "markov": "full",
            "seed": 42,
            "target_kv": "F16",
            "target_sha256": baseline["target"]["sha256"],
            "target_weights": "F16",
            "target_width_mask": 3,
        },
        "historical target/verifier contract changed",
    )
    require(spec["precision"]["activation_bits"] == 8, "W1A8 training precision required")
    require(
        finish["parent"] == base["parent"]
        and base["parent"]["model"]["sha256"] == baseline["models"]["parent3504"]["sha256"],
        "candidate parent differs from unchanged parent control",
    )
    exported = finish["export"]
    require(
        all(
            exported.get(key) is True
            for key in (
                "serialization_audit_passed",
                "typed_metadata_unchanged",
                "all_other_payloads_unchanged",
            )
        ),
        "successful native serialization audit required",
    )
    require(exported["parent"] == finish["parent"]["model"], "native export parent differs")
    permitted, changed = exported["permitted_changed_tensors"], exported["actual_changed_tensors"]
    require(
        len(permitted) == 48
        and len(set(permitted)) == 48
        and bool(changed)
        and set(changed) <= set(permitted),
        "native export tensor scope differs",
    )
    require(Path(exported["output"]["path"]).suffix == ".gguf", "native GGUF output required")
    require(finish.get("checkpoint") is not None, "preserved verified checkpoint required")
    authenticate_all(finish)
    authenticate_all(spec)
    return {
        "finish": pin(path),
        "spec": finish["spec"],
        "checkpoint": finish["checkpoint"],
        "model": exported["output"],
        "local_updates": expected_updates,
        "additional_TRAIN_seconds": seconds,
        "policy": spec["policy"],
        "retention_coefficient": coefficient,
    }


def validate_config(config, baseline):
    expected = copy.deepcopy(baseline)
    expected["models"] = {name: copy.deepcopy(baseline["models"][name]) for name in CONTROLS}
    candidates = {name: model for name, model in config["models"].items() if name not in CONTROLS}
    require(bool(candidates), "at least one admitted candidate required")
    expected["models"].update(candidates)
    require(config == expected, "frozen historical request configuration changed")
    require(
        set(config["panels"]) == {"development"}
        and config["repetitions"] == 5
        and config["prompt_limit"] == 24
        and config["mode"] == "claim"
        and config["balanced_screen"] is False,
        "full fixed development protocol required",
    )
    return candidates


def prepare(args):
    base_ref = {"path": str(BASE_CONFIG), "sha256": BASE_CONFIG_SHA}
    evaluator_ref = {"path": str(EVALUATOR), "sha256": EVALUATOR_SHA}
    baseline = read(checked(base_ref))
    checked(evaluator_ref)
    config = copy.deepcopy(baseline)
    config["models"] = {name: copy.deepcopy(baseline["models"][name]) for name in CONTROLS}
    candidates = {}
    for value in args.candidate:
        name, path = value.split("=", 1)
        require(
            name not in config["models"] and name.replace("_", "").replace("-", "").isalnum(),
            "unique safe candidate name required",
        )
        candidate = validate_candidate(Path(path), args.expected_updates, baseline)
        candidates[name] = candidate
        config["models"][name] = {
            **candidate["model"],
            "custom_ffn": True,
            "draft_limit": 5,
            "markov_mode": "full",
        }
    validate_config(config, baseline)
    authenticate_all(config)
    build = read(checked(config["build"]))
    require(
        build["server"] == config["binary"]
        and build["runtime_libraries"] == config["runtime_libraries"],
        "native build binding differs",
    )
    prompts = [
        json.loads(line)
        for line in checked(config["panels"]["development"]["prompts"]).read_text().splitlines()
        if line.strip()
    ]
    require(
        len(prompts) == 24 and len({row["id"] for row in prompts}) == 24,
        "24 unique development prompts required",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "config.json", config)
    write(
        args.output / "prepared.json",
        {
            "schema": "w1a8_generalization_requests_v1",
            "evaluator": evaluator_ref,
            "base_config": base_ref,
            "config": pin(args.output / "config.json"),
            "wrapper": pin(__file__),
            "candidates": candidates,
            "expected_updates": args.expected_updates,
            "prompt_ids": [row["id"] for row in prompts],
            "scope": SCOPE,
        },
    )


def raw_signature(response):
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "one raw completion required")
    first = choices[0]
    text = first.get("message", {}).get("content")
    stop = first.get("finish_reason")
    require(
        isinstance(text, str) and isinstance(stop, str) and stop,
        "raw text and finish_reason required",
    )
    verbose = response.get("__verbose", {})
    stopping = {key: verbose.get(key) for key in ("truncated", "stop_type", "stopping_word")}
    require(
        type(stopping["truncated"]) is bool
        and isinstance(stopping["stop_type"], str)
        and bool(stopping["stop_type"])
        and isinstance(stopping["stopping_word"], str),
        "raw truncation, stop_type and stopping_word required",
    )
    ids = response.get("generated_token_ids", response.get("token_ids"))
    if ids is None:
        ids = response.get("__verbose", {}).get("tokens")
    require(
        isinstance(ids, list)
        and bool(ids)
        and all(type(token) is int and token >= 0 for token in ids),
        "actual raw generated token IDs required",
    )
    require(
        response.get("usage", {}).get("completion_tokens") == len(ids),
        "raw completion token count differs",
    )
    return ids, hashlib.sha256(text.encode()).hexdigest(), stop, stopping


def audit(output, config, prompt_ids):
    """Independently join raw responses, metrics, target pairs and repetitions."""
    summary = read(output / "summary.json")
    require(
        summary.get("status") == "COMPLETE" and summary.get("diagnostic") is False,
        "complete clean request evaluation required",
    )
    rows = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = {
        (arm, rep, prompt) for arm in config["models"] for rep in range(5) for prompt in prompt_ids
    }
    indexed = {}
    for row in rows:
        key = row["arm"], row["repetition"], row["prompt_id"]
        require(
            row["panel"] == "development" and key in expected and key not in indexed,
            "unexpected or duplicate measured request",
        )
        index = prompt_ids.index(row["prompt_id"])
        cell = output / "development" / f"rep-{row['repetition']:02d}" / row["arm"]
        signature = raw_signature(read(cell / f"prompt-{index:04d}" / "response.json"))
        result = row["result"]
        require(
            signature[:3]
            == (
                result.get("generated_token_ids"),
                result.get("completion_sha256"),
                result.get("finish_reason"),
            ),
            "raw response and recorded token/text/stop differ",
        )
        require(
            result.get("completion_tokens") == len(signature[0]),
            "recorded completion count differs",
        )
        require(
            read(cell / "release.json").get("owned_server_reaped") is True,
            "owned server release receipt missing",
        )
        indexed[key] = signature
    require(set(indexed) == expected, "complete arm/repetition/prompt coverage required")
    for (arm, rep, prompt), signature in indexed.items():
        require(
            signature == indexed[("target_only", rep, prompt)],
            "candidate/control differs from current target token/text/stop/truncation",
        )
        require(
            signature == indexed[(arm, 0, prompt)],
            "output token/text/stop/truncation differs across repetitions",
        )
    return {
        "schema": "w1a8_generalization_request_audit_v1",
        "status": "TOKEN_TEXT_STOP_PARITY_PASS",
        "requests": len(rows),
        "prompt_count": len(prompt_ids),
        "repetitions": 5,
        "arms": list(config["models"]),
        "compared_fields": [
            "generated_token_ids",
            "completion_sha256",
            "finish_reason",
            "__verbose.truncated",
            "__verbose.stop_type",
            "__verbose.stopping_word",
        ],
        "scope": SCOPE,
        "same_repetition_target_comparisons": len(rows),
        "across_repetition_comparisons": len(rows),
        "summary": pin(output / "summary.json"),
        "requests_pin": pin(output / "requests.jsonl"),
        "runtime_release_verification_required": True,
    }


def run(args):
    require(args.allow_native, "sole GPU operator must explicitly pass --allow-native")
    prepared = read(args.prepared)
    require(
        prepared["schema"] == "w1a8_generalization_requests_v1",
        "prepared request contract required",
    )
    require(
        prepared["evaluator"] == {"path": str(EVALUATOR), "sha256": EVALUATOR_SHA}
        and prepared["base_config"] == {"path": str(BASE_CONFIG), "sha256": BASE_CONFIG_SHA},
        "exact retained evaluator/config required; source substitution prohibited",
    )
    authenticate_all(prepared)
    config = read(checked(prepared["config"]))
    baseline = read(checked(prepared["base_config"]))
    candidates = validate_config(config, baseline)
    require(set(candidates) == set(prepared["candidates"]), "candidate membership differs")
    for name, saved in prepared["candidates"].items():
        current = validate_candidate(
            checked(saved["finish"]), prepared["expected_updates"], baseline
        )
        require(
            saved == current
            and candidates[name]
            == {**current["model"], "custom_ffn": True, "draft_limit": 5, "markov_mode": "full"},
            "candidate export/spec/exposure differs",
        )
    authenticate_all(config)
    command = [
        sys.executable,
        "-B",
        str(checked(prepared["evaluator"])),
        "run",
        "--config",
        str(checked(prepared["config"])),
        "--output",
        str(args.output.resolve()),
        "--stop",
        str(args.stop.resolve()),
        "--allow-native",
    ]
    subprocess.run(command, cwd=PROJECT, check=True)
    try:
        report = audit(args.output, config, prepared["prompt_ids"])
    except (ValueError, KeyError, TypeError, OSError) as error:
        write(
            args.output / "independent-audit.json",
            {
                "schema": "w1a8_generalization_request_audit_v1",
                "status": "REJECTED",
                "reason": str(error),
                "preparation": pin(args.prepared),
                "scope": SCOPE,
            },
        )
        raise
    report["preparation"] = pin(args.prepared)
    write(args.output / "independent-audit.json", report)
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--candidate", action="append", required=True, help="name=/absolute/finish.json")
    p.add_argument(
        "--expected-updates", type=int, required=True, help="matched completed local exposure"
    )
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop", type=Path, required=True)
    p.add_argument("--allow-native", action="store_true")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()

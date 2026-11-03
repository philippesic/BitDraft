"""Deferred sole-owner FIT-only selected-head plumbing qualification.

Uses the trainer's exact parent migration, CE/replay sampler and update function.
Two unique updates plus a repeated second update check compact Adam/RNG resume.
Learned bit changes are measured, never forced; native admission stays separate.
Run only when the root operator explicitly owns the GPU lane.
"""

from __future__ import annotations

import copy
import json
import math
import resource
import time

import modern_train_selected_head as trainer
import torch

SCHEMA = "w1a8_selected_head_real_qualification_v1"


def bit_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(bit_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(bit_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def numerical_packet(state):
    """Compare trajectory state; physical duration/cache counters remain measured separately."""
    packet = {
        key: copy.deepcopy(state[key])
        for key in (
            "selected_state",
            "optimizer",
            "cursor",
            "rng",
            "cuda_rng",
            "python_rng",
            "exposure",
            "ce_stage_exposure",
        )
    }
    packet["head_stage"] = {
        key: copy.deepcopy(state["head_stage"][key])
        for key in ("global_start_cursor", "stage_updates", "stage_exposure", "ce_stage_exposure")
    }
    return packet


def fit_only_context(context):
    for key in ("ce_fit", "replay_windows"):
        windows = context[key]
        trainer.require(
            windows and all(w["role"] == "fit" for w in windows),
            "real head qualifier accepts FIT CE/replay windows only",
        )
    trainer.require(
        context["selected"].learn_signs, "real gradient qualification requires learn-signs true"
    )
    trainer.require(
        context["snapshot"]()["cursor"] == trainer.HEAD_START,
        "real head qualifier must start at the declared3840 parent",
    )


def validate_gradient(value, *, expected_rows=512):
    trainer.require(
        isinstance(value, dict)
        and len(value.get("shape", [])) == 2
        and value["shape"][0] == expected_rows
        and value.get("finite") is True
        and value.get("nonzero_elements", 0) > 0
        and value.get("norm", 0) > 0,
        "finite nonzero actual selected-row gradient required",
    )
    trainer.require(math.isfinite(value["norm"]), "selected-row gradient norm must be finite")


def validate_zero_export(actual, expected):
    trainer.require(
        all(actual[key] == expected[key] for key in ("bytes", "sha256")),
        "loaded-parent zero-head export differs from parent GGUF bytes",
    )
    return {
        "byte_identity": True,
        "model": copy.deepcopy(actual),
        "parent": copy.deepcopy(expected),
    }


def resume_two_updates(context):
    """Exercise the same fsynced checkpoints and strict restoration as real training."""
    first = context["update"](capture_gradient=True)
    validate_gradient(first["head_gradient"])
    first_pin = context["save"]("RECOVERY")
    first_identity = trainer.file_pin(first_pin)
    second = context["update"](capture_gradient=True)
    validate_gradient(second["head_gradient"])
    expected = numerical_packet(context["snapshot"]())
    # Persist both paid unique steps before rewinding to exercise the second one.
    second_pin = context["save"]("COMPLETE")
    saved = torch.load(first_pin, map_location="cpu", weights_only=False)
    trainer.require(trainer.file_pin(first_pin) == first_identity, "qualifier checkpoint changed")
    context["restore"](saved)
    repeated = context["update"](capture_gradient=True)
    actual = numerical_packet(context["snapshot"]())
    trainer.require(
        bit_equal(expected, actual) and second["loss"] == repeated["loss"],
        "head compact checkpoint repeated update2 differs in model/Adam/RNG/counters",
    )
    return {
        "physical_updates": 3,
        "unique_updates": 2,
        "exact_compact_Adam_RNG_resume": True,
        "checkpoints": [trainer.file_pin(first_pin), trainer.file_pin(second_pin)],
        "first": first,
        "second": second,
        "repeated_second": repeated,
        "physical_update_seconds": sum(row["update_seconds"] for row in (first, second, repeated)),
    }


def _qualify_loaded(context):
    fit_only_context(context)
    args, selected, guard, write = (context[key] for key in ("args", "selected", "guard", "write"))
    started = time.perf_counter()
    trainer.require(
        tuple(selected.latent.shape) == (512, 2560),
        "real qualification requires the production512x2560 selected masters",
    )
    source = trainer.file_pin(__file__)
    parent_model = trainer.file_pin(trainer.PARENT_GGUF)
    trainer.require(
        parent_model["sha256"] == trainer.PARENT_GGUF_SHA256
        and parent_model["bytes"] == trainer.PARENT_GGUF_BYTES,
        "fixed parent native GGUF changed",
    )
    guard.before_write(parent_model["bytes"])
    zero = selected.export(args.out / "zero-parent.gguf")
    zero_proof = validate_zero_export(zero["output"], parent_model)
    write(args.out / "zero-export.json", {"export": zero, "proof": zero_proof})
    context["save"]("RECOVERY")
    initial_metrics = trainer.sign_metrics(selected)
    trainer.require(
        initial_metrics["hard_sign_flips"] == 0 and initial_metrics["changed_latents"] == 0,
        "qualification did not start from initializer selected signs",
    )
    replay = resume_two_updates(context)
    selected.check(integrity=True)
    metrics = trainer.sign_metrics(selected)
    guard.before_write(parent_model["bytes"])
    learned = selected.export(args.out / "learned.gguf")
    write(args.out / "learned-export.json", learned)
    result = {
        "schema": SCHEMA,
        "status": "SELECTED_HEAD_REAL_GRADIENT_RESUME_EXPORT_PASS",
        "qualifier": source,
        "trainer": trainer.file_pin(trainer.__file__),
        "parent_checkpoint": context["parent_pin"],
        "contract": context["contract"],
        "zero_export": zero_proof,
        "actual_gradient": replay["first"]["head_gradient"],
        "resume": replay,
        "selected_metrics": metrics,
        "learned_export": learned["output"],
        "initial_selected_state_unchanged_bits": initial_metrics,
        "peak_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "qualification_body_seconds": time.perf_counter() - started,
        "whole_job_seconds": time.perf_counter() - context["started"],
        "resident_runtime": context["cache"].report(),
        "native_admission_passed": False,
        "scope": (
            "two FIT-only updates qualify plumbing; learned flips reported without forcing; "
            "no capacity or throughput success claim"
        ),
    }
    trainer.require(
        bit_equal(context["snapshot"]()["selected_state"], selected.state()),
        "export mutated selected model state",
    )
    trainer.require(trainer.file_pin(__file__) == source, "qualifier source changed")
    write(args.out / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "learned_hard_sign_flips": metrics["hard_sign_flips"],
                "unique_updates": 2,
                "native_admission_passed": False,
            }
        ),
        flush=True,
    )


def qualify_loaded(context):
    try:
        _qualify_loaded(context)
    except BaseException as error:
        checkpoint = context["save"]("INTERRUPTED")
        context["write"](
            context["args"].out / "failure.json",
            {
                "error": repr(error),
                "preserved_checkpoint": trainer.file_pin(checkpoint),
                "native_admission_passed": False,
            },
            recovery=True,
        )
        raise


def main():
    trainer.main(qualification_runner=qualify_loaded)


if __name__ == "__main__":
    main()

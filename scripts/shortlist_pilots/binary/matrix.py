#!/usr/bin/env python3
"""CPU-only command matrix and real-input preflight; never launches a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "runs/w1a8-algorithms-20261020"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pilot_matrix(run_root, parent_checkpoint, *, parent_binding=None, python="python3",
                 source_checkpoint=None, source_gguf=None, parent_model=None):
    common = [python, str(ROOT / "scripts/shortlist_pilots/binary/run_parent.py"),
              "--parent-checkpoint", str(parent_checkpoint), "--allow-cuda",
              "--max-train-seconds", "120", "--max-updates", "72", "--pairs", "18"]
    for flag, value in (("--parent-binding", parent_binding), ("--source-checkpoint", source_checkpoint),
                        ("--source-gguf", source_gguf), ("--parent-model", parent_model)):
        if value is not None:
            common += [flag, str(value)]
    techniques = ["control", "elastic-gradient", "trust-gradient", "selective-gradient",
                  "calibration-weight-mse", "calibration-a8", "mos-static", "mos",
                  "double-binary", "double-binary-rank1984", "ternary-a8"]
    pilots = []
    for technique in ["control-initial", *techniques]:
        actual = ("control" if technique == "control-initial" else
                  "double-binary" if technique == "double-binary-rank1984" else technique)
        portable = actual in {"mos", "mos-static", "double-binary", "ternary-a8"}
        command = common + ["--technique", actual, "--run-dir", str(Path(run_root) / technique)]
        if portable:
            command += ["--export-operators"]
        if technique == "control-initial":
            command += ["--initial-only"]
        storage = None
        if actual == "double-binary":
            rank = 1984 if technique == "double-binary-rank1984" else 32
            command += ["--rank", str(rank), "--projection", "blk.4.ffn_up"]
            m, n = 9728, 2560
            bits = (rank+32)*(m+n)
            storage = {"projection": "blk.4.ffn_up", "shape": [m, n], "rank": rank,
                       "binary_factor_bits": rank*(m+n), "float32_scale_values": m+n,
                       "total_storage_bits": bits, "original_weight_count": m*n,
                       "effective_bits_per_original_weight": bits/(m*n),
                       "storage_budget_max_bits_per_original_weight": 1 if rank == 1984 else None,
                       "precision": "F32 input/output scales, intermediate and factor arithmetic",
                       "purpose": "matched <=1BPW capacity screen" if rank == 1984 else "extreme compression diagnostic only"}
        pilots.append({"id": technique, "argv": command, "shell": shlex.join(command),
                       "max_train_seconds": 0 if technique == "control-initial" else 120,
                       "storage": storage,
                       "native_export": not portable and technique != "control-initial",
                       "status": "prepared; not executed",
                       "control": "same restored EAL48/Adam/TRAIN18pairs; share qualified objective control"})
    return {"schema": "shortlist_binary_matrix_v1", "pilots": pilots,
            "execution": "Root owner runs serially after fresh host/GPU/source/runtime qualification",
            "training_ceiling": "each pilot independently admits <=840 TRAIN seconds; matrix uses120",
            "control_reuse": "skip trained control if exact objective EAL48 control already qualified",
            "gate": "initial impact is recorded; flat initial anchors do not automatically reject a technique",
            "cost_scope": "portable arms need native operators before CUDA/full-request conclusions"}


def preflight(binding_path, parent_checkpoint, *, source_checkpoint=None, source_gguf=None, parent_model=None):
    def read(path):
        return json.loads(Path(path).read_text())
    parent = read(binding_path)["parent"]
    original = read(CAMPAIGN / "sign-eal01/bindings.json")["parent_checkpoint"]
    base = read(CAMPAIGN / "command-calibration-pilot01/joint1000.json")["model"]
    dependencies = {
        "current_full48_parent": {**parent["checkpoint"], "path": str(parent_checkpoint)},
        "current_native_parent": {**parent["model"], **({"path": str(parent_model)} if parent_model else {})},
        "original_sign_reference": {**original, **({"path": str(source_checkpoint)} if source_checkpoint else {})},
        "original_floating_source": {**base, **({"path": str(source_gguf)} if source_gguf else {})},
    }
    for name, pin in dependencies.items():
        path = Path(pin["path"])
        pin["exists"] = path.is_file()
        pin["digest_matches"] = pin["exists"] and digest(path) == pin["sha256"]
    required = [CAMPAIGN / p for p in (
        "training-source06/source-manifest.json", "loop2-answer-broad-data01/inputs.json",
        "loop2-answer-interleave-data01/order.json", "command-loop2-answer-data01/cohort.py",
        "command-loop2-firsterror-calib03/source_gate.py")]
    files = {str(p): p.is_file() for p in required}
    return {"dependencies": dependencies, "files_present": files,
            "ready_for_restore_attempt": all(p["digest_matches"] for p in dependencies.values()) and all(files.values()),
            "scope": "read-only input hash/existence check; actual dataset/source/zero-update gates run in loader"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--parent-binding", type=Path, default=CAMPAIGN / "loop2-answer-broad01-p1/bindings.json")
    p.add_argument("--parent-model", type=Path)
    p.add_argument("--source-checkpoint", type=Path)
    p.add_argument("--source-gguf", type=Path)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--python", default="python3")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--output", type=Path, help="exclusive-create JSON; defaults stdout")
    args = p.parse_args()
    matrix = pilot_matrix(args.run_root, args.parent_checkpoint, parent_binding=args.parent_binding,
                          python=args.python, source_checkpoint=args.source_checkpoint,
                          source_gguf=args.source_gguf, parent_model=args.parent_model)
    if args.preflight:
        matrix["preflight"] = preflight(args.parent_binding, args.parent_checkpoint,
                                       source_checkpoint=args.source_checkpoint, source_gguf=args.source_gguf,
                                       parent_model=args.parent_model)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(matrix, stream, indent=2)
    else:
        print(json.dumps(matrix, indent=2))
    if args.preflight and not matrix["preflight"]["ready_for_restore_attempt"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
